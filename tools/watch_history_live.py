"""Живой замер: как быстро новая фраза появляется в истории (её читает окно).

Запуск на 90 секунд, пока пользователь диктует. Печатает строку только когда
появляется новая запись — то есть видно, через сколько секунд после фразы она
попадает в базу, а значит и в подсветку окна чтения.
"""
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

DB = Path(sys.argv[1] if len(sys.argv) > 1 else
          Path(__file__).resolve().parent.parent / "data" / "myflow.sqlite")
SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 90


def newest() -> tuple[int, int]:
    if not DB.exists():
        return 0, 0
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        row = con.execute("select max(id), "
                          "length(coalesce(asr_text,'')) from history "
                          "where id = (select max(id) from history)").fetchone()
    finally:
        con.close()
    return (int(row[0] or 0), int(row[1] or 0)) if row else (0, 0)


def main() -> int:
    print(f"  слежу за {DB.name} — {SECONDS} секунд. Диктуй!")
    last_id, last_len = newest()
    print(f"  старт: id={last_id}, символов={last_len}, "
          f"время={datetime.now().strftime('%H:%M:%S')}")
    started = time.monotonic()
    seen = 0
    while time.monotonic() - started < SECONDS:
        record_id, length = newest()
        if record_id != last_id:
            print(f"  +[{time.strftime('%H:%M:%S')}] появилась запись id={record_id}, "
                  f"символов={length} (предыдущая была {last_id})", flush=True)
            last_id, last_len = record_id, length
            seen += 1
        time.sleep(0.5)
    print(f"  итог: новых записей за {SECONDS} с — {seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
