import argparse
import json
import math
import os
from pathlib import Path
import sys

from .agent import Agent, StepLimitExceeded
from .approval import TerminalApproval
from .classifier import DeepSeekActionClassifier
from .model import OpenAIChatModel
from .execution import ToolError
from .persistence import inspect_run
from .permissions import PERMISSION_MODES, PermissionPolicy
from .tools import workspace_tools


def configuration(args) -> tuple[str | None, str | None, str | None]:
    deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = args.model or os.getenv("DEEPSEEK_MODEL") or (
        "deepseek-flash" if deepseek else os.getenv("OPENAI_MODEL"))
    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL") or (
        "https://api.deepseek.com" if deepseek else os.getenv("OPENAI_BASE_URL"))
    return key, model, base_url


def progress(event):
    kind = event["kind"]
    if kind == "session_started":
        print(f"Session: {event['directory']}", file=sys.stderr, flush=True)
    elif kind == "started":
        print(f"[{event['tool_call_id']}] {event['tool']} started", file=sys.stderr, flush=True)
    elif kind == "output":
        print(f"[{event['tool_call_id']}:{event['stream']}] {event['text']}", end="", file=sys.stderr, flush=True)
    elif kind == "finished":
        print(f"\n[{event['tool_call_id']}] {event['status']} ({event['duration']:.2f}s)", file=sys.stderr, flush=True)
    elif kind == "permission_decision":
        print(f"[{event['tool_call_id']}] permission: {event['decision']} ({event['source']})", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal coding agent in a local workspace.")
    parser.add_argument("task", nargs="?")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--command-timeout", type=float, default=30)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--permission-mode", choices=PERMISSION_MODES, default="allow")
    parser.add_argument("--deny-tool", action="append", default=[], metavar="NAME")
    parser.add_argument("--ask-tool", action="append", default=[], metavar="NAME")
    parser.add_argument("--allow-tool", action="append", default=[], metavar="NAME")
    parser.add_argument("--classifier-model", help="Defaults to the agent model")
    parser.add_argument("--classifier-timeout", type=float, default=15)
    parser.add_argument("--auto-denial-limit", type=int, default=3)
    parser.add_argument("--quiet", action="store_true", help="Hide tool progress")
    parser.add_argument("--inspect", type=Path, metavar="SESSION_DIRECTORY")
    args = parser.parse_args(argv)
    if args.inspect:
        try:
            print(json.dumps(inspect_run(args.inspect), ensure_ascii=False, indent=2))
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0
    if not args.task or not args.task.strip():
        parser.error("A nonempty task is required")
    if args.max_steps < 1 or not 1 <= args.max_parallel <= 32 or not 0 < args.command_timeout <= 3600:
        parser.error("Require max-steps > 0, max-parallel in [1,32], and command-timeout in (0,3600]")
    if args.auto_denial_limit < 1 or not math.isfinite(args.classifier_timeout) or not 0 < args.classifier_timeout <= 120:
        parser.error("Require auto-denial-limit > 0 and classifier-timeout in (0,120]")
    api_key, model_name, base_url = configuration(args)
    if not model_name:
        parser.error("Set OPENAI_MODEL or pass --model")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY or OPENAI_API_KEY")
    agent = None
    try:
        tools = workspace_tools(args.workspace, command_timeout=args.command_timeout)
        unknown = (set(args.deny_tool) | set(args.ask_tool) | set(args.allow_tool)) - {tool.name for tool in tools}
        if unknown:
            parser.error(f"Unknown tool in permission rules: {', '.join(sorted(unknown))}")
        model = OpenAIChatModel(model_name, api_key=api_key, base_url=base_url)
        classifier = DeepSeekActionClassifier(model.client, args.classifier_model or model_name,
                                               timeout=args.classifier_timeout) if args.permission_mode == "auto" else None
        policy = PermissionPolicy(mode=args.permission_mode, read_only=args.read_only,
                                  denied=set(args.deny_tool), ask=set(args.ask_tool), allowed=set(args.allow_tool),
                                  classifier=classifier, rejection_threshold=args.auto_denial_limit,
                                  human_approve=TerminalApproval())
        agent = Agent(model, tools, max_steps=args.max_steps, workspace=args.workspace,
                      max_parallel=args.max_parallel, policy=policy,
                      on_event=None if args.quiet else progress)
        result = agent.run(args.task)
    except StepLimitExceeded as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except ToolError as exc:
        print(str(exc), file=sys.stderr)
        return 130 if exc.code == "cancelled" else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if agent and agent.last_session_directory:
            print(f"Session saved: {agent.last_session_directory}", file=sys.stderr)
    print(result.text)
    return 0
