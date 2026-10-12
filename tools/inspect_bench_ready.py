"""Что есть в базе для теста «чтение вслух» и что исключено из git."""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(r"C:\Projects\MyFLOW")
con = sqlite3.connect(ROOT / "data" / "myflow.sqlite")
print("  ТАБЛИЦЫ:")
for (name,) in con.execute("select name from sqlite_master where type='table'"):
    cols = [row[1] for row in con.execute(f"PRAGMA table_info({name})")]
    try:
        count = con.execute(f"select count(*) from {name}").fetchone()[0]
    except Exception:
        count = "?"
    print(f"    {name} ({count}): {', '.join(cols)}")
print("  ПОСЛЕДНЯЯ ЗАПИСЬ (обрезано):")
columns = [row[1] for row in con.execute("PRAGMA table_info(records)")]
for row in con.execute("select * from records order by id desc limit 1"):
    for name, value in zip(columns, row):
        text = str(value).replace("\n", " ")
        print(f"    {name} = {text[:60]}")
con.close()

print("\n  .gitignore целиком:")
ignore = ROOT / ".gitignore"
print("\n".join("    " + line for line in ignore.read_text(encoding="utf-8").splitlines()))
