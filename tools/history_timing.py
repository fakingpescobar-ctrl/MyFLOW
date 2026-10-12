"""Почему подсветка отстаёт: когда именно запись попадает в историю.

Окно чтения берёт фразы из истории (SQLite), поэтому если запись появляется позже,
чем видно текст в Блокноте, подсветка будет отставать на эту задержку. Меряем:
  * разницу между created_at соседних записей (длительность фразы);
  * поля status/observation_end_reason — ждёт ли запись окончания наблюдения за правкой.
"""
import sqlite3
from pathlib import Path

DB = Path(r"data\myflow.sqlite")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

rows = list(con.execute(
    "select id, created_at, status, observation_end_reason, transcript_origin, "
    "length(asr_text) as n, duration_ms, e2e_latency_ms "
    "from history order by id desc limit 12"))
con.close()

print("  последние записи (сверху — свежие):")
print("    id   created_at            симв  status        observation_end")
for row in rows:
    print(f"    {row['id']:<4} {str(row['created_at'])[:19]:<19} "
          f"{row['n'] or 0:<5} {str(row['status'])[:12]:<13} "
          f"{str(row['observation_end_reason'])[:20]}")

print("  длины/задержки последних 5:")
for row in rows[:5]:
    print(f"    id={row['id']} символов={row['n']} длительность={row['duration_ms']} мс "
          f"e2e={row['e2e_latency_ms']} мс")
