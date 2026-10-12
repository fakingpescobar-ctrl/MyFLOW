"""Проверка: мост стоит КОПИЕЙ (а не ссылкой) и всё видно снаружи."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

home = Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh")
print(f"  домашний каталог DSH: {home}")
for profile in ("desktop", "web"):
    dest = home / "profiles" / profile / "node_modules" / "dsh-plugin-myflow"
    if not dest.exists():
        print(f"  [{profile}] плагина нет")
        continue
    is_link = dest.is_symlink() or os.path.isjunction(dest)
    index = dest / "index.js"
    print(f"  [{profile}] {'ССЫЛКА' if is_link else 'КОПИЯ'}, "
          f"index.js: {'есть' if index.exists() else 'нет'}")
    manifest = home / "profiles" / profile / "package.json"
    if manifest.exists():
        import json
        data = json.loads(manifest.read_text(encoding="utf-8"))
        dep = (data.get("dependencies") or {}).get("dsh-plugin-myflow")
        bundles = ((data.get("dsh") or {}).get("profile") or {}).get("bundles") or []
        print(f"        зависимость: {dep}")
        print(f"        в bundles: {'да' if 'dsh-plugin-myflow' in bundles else 'нет'}")

from myflow.dsh_bridge import bridge_health, bridge_installed   # noqa: E402

health = bridge_health()
print(f"  мост готов: {bridge_installed()} | битые: {health['broken'] or '—'}")
