"""Юнит Ф8: разрешение стиля (per-app маска > override F7 > дефолт) + цикл."""
import sys
sys.path.insert(0, r"C:\Projects\MyFLOW")
sys.stdout.reconfigure(encoding="utf-8")
from myflow.main import MyFlowApp
from myflow.config import SettingsStore

app = MyFlowApp()  # без start(): серверы/хоткеи не нужны
ok = True

def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"{'✅' if good else '❌'} {label}: {got!r} (хотелось {want!r})")

# 1) auto по окну: маска "terminal" на заголовке
app._style_override = None
check("терминал-окно -> terminal", app._resolve_style("Terminal — Ubuntu"), "terminal")
check("powershell-окно -> terminal", app._resolve_style("Windows PowerShell"), "terminal")
check("незнакомое окно -> дефолт", app._resolve_style("Visual Studio Code"), "plain")
# 2) override F7 перекрывает маску
app._style_override = "summary"  # ошибочный ключ должен падать
try:
    app._resolve_style("Terminal")
    check("неизвестный override -> падает", "не упал", "упал")
except KeyError:
    check("неизвестный override -> падает", "упал", "упал")
app._style_override = "formal"
check("override formal поверх терминала", app._resolve_style("Terminal"), "formal")
# 3) цикл F7: None -> plain, plain -> casual -> ... -> None
app._style_override = None
app._cycle_style(debounce=0)
check("F7 1: None -> plain", app._style_override, "plain")
app._cycle_style(debounce=0)
check("F7 2: plain -> casual", app._style_override, "casual")
for _ in range(3):
    app._cycle_style(debounce=0)
check("F7 5: formal -> summarize", app._style_override, "summarize")
app._cycle_style(debounce=0)
check("F7 6: summarize -> авто(None)", app._style_override, None)
# 4) сохранение настроек с новыми полями
s = SettingsStore()
assert "style_by_app" in s.get().__dict__, "style_by_app нет в Settings"
assert s.get().hotkey.style_cycle_key == "f7", "style_cycle_key не f7"
check("settings: style_by_app + f7", True, True)

print("\nИТОГ:", "OK" if ok else "FAIL")