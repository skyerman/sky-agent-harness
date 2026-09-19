import os
from pathlib import Path
import stat
import tempfile


class WorkspaceBusy(RuntimeError):
    pass


class WorkspaceLease:
    def __init__(self, root: Path):
        self.path = root / ".sky-agent" / "workspace.lock"
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise WorkspaceBusy("Another agent owns this workspace") from exc
        return self

    def __exit__(self, *args):
        if self.file:
            try:
                self.file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            finally:
                self.file.close()
                self.file = None


def replace_file(target: Path, data: bytes, recheck, *, creating: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".sky-agent-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        recheck()
        if creating:
            # Linking publishes a complete file without overwriting a concurrent creation.
            os.link(temporary, target)
        else:
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
            recheck()
            os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
