"""Замер на «новых» диктовках: которых модель НЕ видела при обучении.

Зачем отдельный инструмент. Обычная валидация (`postproc_val.jsonl`) — это 48 пар,
они малы и шумят. Главный вопрос практики другой: **не деградирует ли модель на
свежих фразах**, когда пользователь надиктует новое. Проверить это честно можно
только разбиением ПО ВРЕМЕНИ: берём последние N диктовок, обучаем модель без них
и смотрим на них.

Сценарий:
  1. `--prepare` — собрать датасет, исключив последние N диктовок (по id/времени),
     и напечатать, сколько ушло в «новое»;
  2. обучение делает обычный `train_postproc_lora.py` на этом датасете;
  3. `--eval` — сравнить на «новых» диктовках: сырой ASR / правила / 4B / кандидат.

Использование:
    python tools/eval_on_new_dictations.py --prepare --holdout 80
    ... обучение ...
    python tools/eval_on_new_dictations.py --eval --candidate models\\...gguf --holdout 80
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_postproc_dataset import finalize_clean, real_pairs, to_chat  # noqa: E402
from eval_postproc import distance, light_clean, words                  # noqa: E402
from myflow.config import SettingsStore                                 # noqa: E402
from myflow.tui import Supervisor, default_specs                        # noqa: E402

HOLDOUT = ROOT / "data" / "train" / "holdout_pairs.jsonl"
PORT_CANDIDATE = 8083


def split(holdout: int) -> tuple[list[dict], list[dict]]:
    """Последние `holdout` диктовок — «новые», остальные — для обучения."""
    pairs = real_pairs()
    pairs = sorted(pairs, key=lambda p: p["id"])
    if holdout >= len(pairs):
        raise SystemExit(f"диктовок всего {len(pairs)}, holdout {holdout} слишком велик")
    return pairs[:-holdout], pairs[-holdout:]


def prepare(holdout: int, out_dir: Path) -> int:
    train_pairs, new_pairs = split(holdout)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [to_chat(p["src"], p["dst"]) for p in train_pairs]
    (out_dir / "postproc_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    HOLDOUT.write_text("\n".join(json.dumps(
        {"src": p["src"], "dst": p["dst"]}, ensure_ascii=False) for p in new_pairs),
        encoding="utf-8")
    print(f"  для обучения: {len(rows)} пар -> {out_dir / 'postproc_train.jsonl'}")
    print(f"  «новые» (не для обучения): {len(new_pairs)} -> {HOLDOUT.relative_to(ROOT)}")
    print("\n  Обучение (пример):")
    print(f"    .venv-train\\Scripts\\python.exe tools\\train_postproc_lora.py "
          f"--base data\\train\\Qwen3-0.6B --train {out_dir / 'postproc_train.jsonl'} "
          f"--val {out_dir / 'postproc_train.jsonl'} --epochs 4 "
          f"--out {out_dir / 'lora'}")
    print("\n  Затем:")
    print(f"    .venv\\Scripts\\python.exe -X utf8 tools\\eval_on_new_dictations.py "
          f"--eval --holdout {holdout} --candidate <gguf>")
    return 0


def pct(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))]


def row(name: str, dists: list[float], lat: list[int] | None = None) -> str:
    exact = sum(1 for d in dists if d == 0) / max(1, len(dists)) * 100
    line = (f"  {name:<24} p50 {pct(dists, .5):.3f} | среднее {statistics.fmean(dists):.3f}"
            f" | точных {exact:>5.1f}%")
    if lat:
        line += f" | p50 {pct([float(x) for x in lat], .5):>4.0f} мс"
    return line


def wait_health(url: str, seconds: float = 120.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/models", timeout=1.5).ok:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False


def evaluate(holdout: int, candidate: Path | None) -> int:
    if not HOLDOUT.exists():
        print(f"нет {HOLDOUT}: сначала --prepare")
        return 1
    pairs = [json.loads(l) for l in HOLDOUT.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    print(f"  «новых» диктовок (модель их не видела): {len(pairs)}")
    print(row("сырой ASR", [distance(p["dst"], p["src"]) for p in pairs]))
    print(row("light_clean", [distance(p["dst"], light_clean(p["src"])[0]) for p in pairs]))

    from eval_postproc import run_llm
    sup = Supervisor(default_specs(), enabled=True)

    def gated(url: str, style: str) -> tuple[list[float], list[int], dict]:
        """Боевой путь: ворота быстрого пути -> правила -> модель.

        Без ворот модель правит и те фразы, которые трогать не надо, и картина
        получается искажённой (на свежих диктовках «ничего не делать» лучше всех,
        а модель «портит» 0.104). В бою 431 фраза из 624 до модели вообще не
        доходит — здесь воспроизводим ровно эту логику.
        """
        from myflow.config import SettingsStore as _Store
        from myflow.fastpath import fast_path_allowed, looks_wellformed
        from myflow.lightclean import light_clean, strip_inner_fillers
        from myflow.polish import PolishClient

        cfg = _Store().get().polish
        cfg.url = url
        cfg.enabled = True
        client = PolishClient(cfg)
        dists, lats, routes = [], [], {}
        for row in pairs:
            src = row["src"]
            fast_ok, _ = fast_path_allowed("gigaam", "plain", src)
            if fast_ok:
                out, route = src, "fast_path"
            else:
                cleaned, rules = light_clean(src)
                terminal = src.rstrip()[-1:] in ".!?…"
                if rules and terminal and looks_wellformed(cleaned)[0]:
                    out, route = cleaned, "local_polish"
                else:
                    started = time.perf_counter()
                    try:
                        res = client.polish(src, style=style, dictionary=[])
                        out = res.text or src
                        route = "llm"
                    except Exception:  # noqa: BLE001
                        out, route = src, "llm_error"
                    lats.append(int((time.perf_counter() - started) * 1000))
            cleaned_final, inner = strip_inner_fillers(out)
            if inner and cleaned_final:
                out = cleaned_final
            dists.append(distance(row["dst"], out))
            routes[route] = routes.get(route, 0) + 1
        return dists, lats, routes

    print("\n  --- боевой путь (с воротами быстрого пути) ---")
    llm = next(s for s in sup.specs if s.key == "llm")
    base_url = SettingsStore().get().polish.url
    started_llm = started_cand = False
    proc = None
    log = ROOT / "data" / "train" / "candidate_new.log"
    try:
        if not wait_health(base_url, 2.0):
            print("  поднимаю 4B...", flush=True)
            sup.start(llm)
            started_llm = sup.wait_ready(llm, 180.0)
        dists, lat, routes = gated(base_url, "plain")
        print(row("Qwen3-4B (боевой)", dists, lat) + f" | {routes}")

        if candidate is not None:
            path = candidate if candidate.is_absolute() else ROOT / candidate
            server = ROOT / "bin" / "llama-cpp" / "llama-server.exe"
            with open(log, "ab") as fh:
                proc = subprocess.Popen(
                    [str(server), "-m", str(path), "--host", "127.0.0.1",
                     "--port", str(PORT_CANDIDATE), "-c", "4096", "-np", "1",
                     "-ngl", "99", "-fa", "on", "--jinja"],
                    stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
            started_cand = wait_health(f"http://127.0.0.1:{PORT_CANDIDATE}/v1")
            if started_cand:
                from myflow.postproc_prompt import POSTPROC_STYLE
                dists, lat, routes = gated(f"http://127.0.0.1:{PORT_CANDIDATE}/v1",
                                           POSTPROC_STYLE)
                print(row("наша (боевой)", dists, lat) + f" | {routes}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
        if started_llm:
            sup.stop(llm)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Замер на новых диктовках")
    ap.add_argument("--holdout", type=int, default=80, help="сколько последних диктовок держим")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "new_split")
    ap.add_argument("--candidate", type=Path, default=ROOT / "models"
                    / "myflow-postproc-qwen3-06b-q8_0.gguf")
    args = ap.parse_args()
    if args.prepare:
        return prepare(args.holdout, args.out)
    if args.eval:
        return evaluate(args.holdout, args.candidate)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
