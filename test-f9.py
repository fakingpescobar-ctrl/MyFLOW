"""Юнит Ф9: looks_like_edit_of + watch_pasted_edit (фейковый буфер, без GUI/серверов)."""
import sys, time
sys.path.insert(0, r"C:\Projects\MyFLOW")
sys.stdout.reconfigure(encoding="utf-8")
from myflow.learning import looks_like_edit_of, watch_pasted_edit
from myflow.db import Database

ok = True
def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"{'✅' if good else '❌'} {label}: {got!r} (хотелось {want!r})")

base = "открыть эндпоинт на порту 8081 и запустить тест проверяем пайплайн"
# 1) валидная правка: замена слова + вставка
check("правка (замена слова)", looks_like_edit_of(base, "открыть эндпоинт на порту 8081 и запустить тест проверяем пайплайн теперь"), True)
check("правка (замена слова)", looks_like_edit_of(base, "открой эндпоинт на порту 8081 и запусти тест проверяем пайплайн"), True)
check("идентичный текст — не правка", looks_like_edit_of(base, base), False)
check("пустая строка", looks_like_edit_of(base, ""), False)
check("совсем другой текст", looks_like_edit_of(base, "привет мир как дела сегодня погода хорошая день отличный"), False)
# 2) интеграция: фейковый клипборд, юзер «правит и копирует» через 0.2с
class FakeClip:
    def __init__(self, sequence):
        self._seq = list(sequence); self._i = 0
    def paste(self):
        v = self._seq[min(self._i, len(self._seq)-1)]
        if self._i < len(self._seq)-1: self._i += 1
        return v

db = Database()
hid = db.insert_history(__import__('myflow.db', fromlist=['DictationRecord']).DictationRecord(
    asr_text=base, polished_text=base, app_window="test", duration_ms=1000, e2e_latency_ms=100,
    detected_language="ru", style="plain"))
fake = FakeClip([base, base, "открыть эндпоинт на порту 8080 и запустить тест проверяем пайплайн"])
t0 = time.monotonic()
res = watch_pasted_edit(db, hid, base, fake, window_seconds=3.0, poll_interval=0.1)
print(f"watch: {res} за {time.monotonic()-t0:.1f}с")
check("watch поймал правку", isinstance(res, dict) and res["edit_distance"] > 0, True)
# редакция сохранена в истории
row = db.attach_edit and db.get_polished_text(hid)
e = db._conn.execute("SELECT edited_text FROM history WHERE id=?", (hid,)).fetchone()
check("edited_text записан", e is not None and e[0] != base, True)

# 3) окно истекло без правки
db2 = Database()
hid2 = db2.insert_history(__import__('myflow.db', fromlist=['DictationRecord']).DictationRecord(
    asr_text=base, polished_text=base, app_window="test", duration_ms=1000, e2e_latency_ms=100,
    detected_language="ru", style="plain"))
res2 = watch_pasted_edit(db2, hid2, base, FakeClip([base]*20), window_seconds=0.5, poll_interval=0.1)
check("окно истекло -> None", res2 is None, True)

print("\nИТОГ:", "OK" if ok else "FAIL")