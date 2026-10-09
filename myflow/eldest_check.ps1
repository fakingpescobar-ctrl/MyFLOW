# Живые собратья myflow-main: PID + время создания (миллисекунды!) + PPID + путь exe.
# 4-е поле (exe-путь) нужно, чтобы ОТСЕЯТЬ venv-шим (Scripts\python.exe): это
# обёртка, которая ждёт ребёнка — реальный интерпретатор в base Python
# (…\Python313\python.exe). Без этого фильтра «пара» (шим + настоящий python)
# с той же командной строкой вечно выглядит как два main.
$f = "Name='python.exe'"
Get-CimInstance Win32_Process -Filter $f |
  Where-Object { $_.CommandLine -like '*myflow*' } |
  ForEach-Object { "$($_.ProcessId);$($_.CreationDate.ToString('yyyyMMddHHmmssfff'));$($_.ParentProcessId);$($_.ExecutablePath)" }