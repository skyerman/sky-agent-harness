import json
import os
import sys

from .execution import ToolError
from .permissions import ApprovalRequest


class TerminalApproval:
    def __call__(self, request: ApprovalRequest) -> bool:
        if not sys.stdin.isatty():
            return False
        if len(request.call.arguments_json) > 16000:
            print("Arguments exceed terminal approval display limit; call denied.", file=sys.stderr)
            return False
        print(f"\nPermission required: {request.call.tool_name}", file=sys.stderr)
        print(f"Workspace: {request.call.workspace or '(unspecified)'}", file=sys.stderr)
        print(f"Reason: {json.dumps(request.reason, ensure_ascii=True)}", file=sys.stderr)
        print(json.dumps(request.call.arguments, ensure_ascii=True, indent=2), file=sys.stderr)
        print("Allow this exact call? [y/N] ", end="", file=sys.stderr, flush=True)
        try:
            return self._readline(request.cancel).strip().lower() in {"y", "yes"}
        except EOFError:
            return False

    @staticmethod
    def _readline(cancel):
        # Poll the terminal so a cancelled worker cannot hold the executor open.
        if os.name == "nt":
            import msvcrt
            answer = ""
            while not cancel.is_set():
                if not msvcrt.kbhit():
                    cancel.wait(0.05)
                    continue
                char = msvcrt.getwch()
                if char in {"\r", "\n"}:
                    print(file=sys.stderr)
                    return answer
                if char == "\x03":
                    cancel.set()
                    break
                if char == "\x1a":
                    return ""
                if char in {"\x00", "\xe0"}:
                    msvcrt.getwch()
                elif char == "\b":
                    if answer:
                        answer = answer[:-1]
                        print("\b \b", end="", file=sys.stderr, flush=True)
                elif char.isprintable() and len(answer) < 20:
                    answer += char
                    print(char, end="", file=sys.stderr, flush=True)
        else:
            import select
            while not cancel.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    return sys.stdin.readline()
        raise ToolError("cancelled", "Approval cancelled")
