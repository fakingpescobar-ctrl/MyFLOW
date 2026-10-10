"""Job Object с KILL_ON_JOB_CLOSE: серверы умирают вместе с MyFLOW.

Зачем: если окно MyFLOW закрыть, а тем более убить процесс (краш, диспетчер
задач, `Stop-Process -Force`), обычный `terminate()` не выполнится и
llama-server останется висеть в памяти и VRAM. Для игр это критично: 6 ГБ VRAM
заняты впустую.

Windows Job Object с флагом `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` решает это на
уровне ОС: как только закрывается наш процесс (любым способом), ядро само
убивает все процессы, привязанные к job'у. Привязываем ТОЛЬКО те серверы,
которые запустили сами, — чужие экземпляры (запущенные скриптами пользователя)
не трогаем.
"""
from __future__ import annotations

import ctypes
import sys

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class ProcessJob:
    """Job Object, который убивает своих детей при закрытии нашего процесса."""

    def __init__(self, name: str | None = None) -> None:
        self.handle = None
        self.error: str | None = None
        self.assigned = 0
        if sys.platform != "win32":
            self.error = "не Windows"
            return
        try:
            k = ctypes.windll.kernel32
            self.handle = k.CreateJobObjectW(None, name)
            if not self.handle:
                # ВАЖНО: при отказе CreateJobObject возвращает NULL (0), а не None.
                # Без сброса в None job ошибочно считался бы активным.
                self.handle = None
                self.error = "CreateJobObject не вернул описатель"
                return
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = k.SetInformationJobObject(
                self.handle, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info))
            if not ok:
                self.error = "SetInformationJobObject не применился"
                k.CloseHandle(self.handle)
                self.handle = None
        except Exception as exc:  # noqa: BLE001 — не Windows/нет прав: работаем без job
            self.error = repr(exc)
            self.handle = None

    @property
    def active(self) -> bool:
        # bool(), а не `is not None`: неудачный CreateJobObject даёт 0.
        return bool(self.handle)

    def assign(self, proc) -> bool:
        """Привязать процесс к job'у. False — не получилось (не критично)."""
        if self.handle is None or proc is None:
            return False
        try:
            handle = int(getattr(proc, "_handle", 0) or 0)
            if not handle:
                return False
            ok = ctypes.windll.kernel32.AssignProcessToJobObject(
                self.handle, ctypes.c_void_p(handle))
            if ok:
                self.assigned += 1
                return True
            self.error = f"AssignProcessToJobObject: код {ctypes.get_last_error()}"
            return False
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
            return False
