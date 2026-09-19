import codecs
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable

from .persistence import PersistenceError, RunStore

PREVIEW_LIMIT = 16000


class ToolError(Exception):
    def __init__(self, code: str, message: str, details=None):
        super().__init__(message)
        self.code = code
        self.details = details

    def result(self) -> dict:
        error = {"code": self.code, "message": str(self)}
        if self.details is not None:
            error["details"] = self.details
        return {"ok": False, "error": error}


@dataclass
class ExecutionContext:
    store: RunStore
    cancel: threading.Event = field(default_factory=threading.Event)
    callback: Callable[[dict], None] | None = None
    invocation_id: str = "direct"
    tool_call_id: str = "direct"
    tool: str = ""
    event_lock: threading.RLock = field(default_factory=threading.RLock)

    def check_cancelled(self):
        if self.cancel.is_set():
            raise ToolError("cancelled", "Execution cancelled")

    def emit(self, kind: str, *, persist: bool = True, **data):
        with self.event_lock:
            event = {"invocation_id": self.invocation_id, "tool_call_id": self.tool_call_id,
                     "tool": self.tool, **data}
            if persist:
                try:
                    event = self.store.record(kind, **event)
                except PersistenceError:
                    self.cancel.set()
                    raise
            else:
                event.update(kind=kind, session_id=self.store.session_id,
                             timestamp=datetime.now(timezone.utc).isoformat())
            if self.callback:
                try:
                    self.callback(event)
                except Exception as exc:
                    self.store.record("observer_error", message=str(exc))


class ProcessTree:
    """Own command descendants; Windows jobs also survive an early parent exit."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.job = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            kernel.CreateJobObjectW.restype = wintypes.HANDLE
            kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            kernel.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel.TerminateJobObject.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            kernel.SetInformationJobObject.restype = wintypes.BOOL

            class BasicLimits(ctypes.Structure):
                _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                            ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                            ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                            ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                            ("scheduling", wintypes.DWORD)]

            class ExtendedLimits(ctypes.Structure):
                _fields_ = [("basic", BasicLimits), ("io_counters", ctypes.c_uint64 * 6),
                            ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                            ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

            self.kernel = kernel
            self.job = kernel.CreateJobObjectW(None, None)
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            configured = self.job and kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            if not configured or not kernel.AssignProcessToJobObject(self.job, int(process._handle)):
                error = ctypes.get_last_error()
                if self.job:
                    kernel.CloseHandle(self.job)
                    self.job = None
                process.kill()
                process.wait()
                raise OSError(error, "Could not contain command in a Windows job")

    def close(self):
        if os.name == "nt":
            if self.job:
                self.kernel.TerminateJobObject(self.job, 1)
                self.kernel.CloseHandle(self.job)
                self.job = None
        else:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


def run_process(argv: list[str], cwd: Path, timeout: float, context: ExecutionContext) -> dict:
    context.check_cancelled()
    refs = {}
    paths = {}
    for stream in ("stdout", "stderr"):
        refs[stream], paths[stream] = context.store.new_artifact(stream)
        try:
            paths[stream].touch(exist_ok=False)
        except OSError as exc:
            raise PersistenceError(f"Could not create output artifact: {exc}") from exc
    context.emit("artifacts", artifacts=refs)
    previews = {"stdout": "", "stderr": ""}
    counts = {"stdout": 0, "stderr": 0}
    errors = []
    drain_stop = threading.Event()

    def drain(pipe, stream):
        try:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            if os.name != "nt":
                os.set_blocking(pipe.fileno(), False)
            with paths[stream].open("w", encoding="utf-8", newline="") as output:
                while True:
                    try:
                        chunk = os.read(pipe.fileno(), 4096)
                    except BlockingIOError:
                        if drain_stop.is_set():
                            chunk = b""
                        else:
                            drain_stop.wait(0.02)
                            continue
                    text = decoder.decode(chunk, final=not chunk)
                    if text:
                        output.write(text)
                        output.flush()
                        counts[stream] += len(text)
                        previews[stream] += text[:max(0, PREVIEW_LIMIT - len(previews[stream]))]
                        context.emit("output", persist=False, stream=stream, text=text)
                    if not chunk:
                        break
                os.fsync(output.fileno())
        except Exception as exc:
            errors.append(exc)
        finally:
            pipe.close()

    context.check_cancelled()
    try:
        command = [sys.executable, "-u", str(Path(__file__).with_name("_command.py")), *argv] if os.name == "nt" else argv
        process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE if os.name == "nt" else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except OSError as exc:
        raise ToolError("command_start_failed", str(exc), {"artifacts": refs}) from exc
    readers = []
    tree = None
    reason = None
    started = time.monotonic()
    try:
        tree = ProcessTree(process)
        for stream in ("stdout", "stderr"):
            reader = threading.Thread(target=drain, args=(getattr(process, stream), stream))
            reader.start()
            readers.append(reader)
        if process.stdin:
            process.stdin.write(b"1")
            process.stdin.close()
        while process.poll() is None:
            if context.cancel.is_set():
                reason = "cancelled"
                break
            if time.monotonic() - started >= timeout:
                reason = "timeout"
                break
            if errors:
                break
            context.cancel.wait(0.02)
    finally:
        if tree:
            tree.close()
        elif process.poll() is None:
            process.kill()
            process.wait()
        drain_stop.set()
        for reader in readers:
            reader.join()
        if process.stdin:
            process.stdin.close()
        for pipe in (process.stdout, process.stderr):
            pipe.close()
    if errors:
        raise PersistenceError(f"Could not stream command output: {errors[0]}") from errors[0]
    result = {"exit_code": process.returncode, **previews, "artifacts": refs,
              "characters": counts,
              "truncated": {s: counts[s] > PREVIEW_LIMIT for s in counts}}
    if reason:
        raise ToolError(reason, "Command timed out" if reason == "timeout" else "Command cancelled", result)
    if process.returncode != 0:
        raise ToolError("command_failed", f"Command exited with code {process.returncode}", result)
    return result
