"""Read-only git tool: status, diff, log, show and merge-base, run without a
permission prompt.

Git can run commands named in a repository's own configuration, even for
reading: an fsmonitor, a clean filter or filter process, a hook, an
external diff, a textconv, a pager or a signature check. A repository's
`.git/config` comes with it (a copied project, an unpacked archive), so this
tool runs git with those switched off. The user's own global and system
configuration stays in effect, except the external diffs, textconvs and
pager, which never run here. Arguments are fields, never flags: nothing the
model passes can turn an option back on.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin.bash import command_env, run_command
from cortex.tooling.builtin.common import resolve_repo_path
from cortex.tooling.types import ToolExecutionState, ToolResult

_TIMEOUT_SECONDS = 60.0
# Settings that make git run something, switched off for every call.
_SAFE_CONFIG = (
    "core.fsmonitor=false",
    "core.hooksPath=/dev/null",
    "log.showSignature=false",
    "core.pager=cat",
)
# Never let a diff run a program or recurse into a submodule's repository.
_SAFE_DIFF = ("--no-ext-diff", "--no-textconv", "--ignore-submodules=all", "--no-color")


class GitArgs(BaseModel):
    command: Literal["status", "diff", "log", "show", "merge-base"]
    revision: Optional[str] = Field(
        default=None,
        description=(
            "diff: a commit or range (HEAD, main...HEAD); log: a commit or range "
            "(default HEAD); show: a commit, or commit:path for a file at it (default HEAD); "
            "merge-base: the branch to compare HEAD with (required)."
        ),
    )
    paths: List[str] = Field(default_factory=list, description="Limit to these paths.")
    staged: bool = Field(default=False, description="diff: the staged changes.")
    stat: bool = Field(default=False, description="diff, log, show: a summary per file.")
    patch: bool = Field(
        default=False,
        description="diff, log, show: the changes line by line. diff and show give them "
        "unless stat is set (with stat, both); log gives them only with patch.",
    )
    max_count: int = Field(default=20, ge=1, le=200, description="log: how many commits.")


class GitTool(BaseTool[GitArgs]):
    name = "git"
    description = (
        "Read the repository's git state without asking: status, diff (working tree, "
        "staged, or between commits), log, show (a commit, or a file at one), and "
        "merge-base. Use bash for anything that changes the repository."
    )
    permission = "read"
    args_model = GitArgs

    def run(self, parsed_args: GitArgs, context: ToolContext) -> ToolResult:
        argv = _argv(parsed_args, context.repo_root)
        root = context.repo_root
        settings = [*_SAFE_CONFIG, *_repo_filter_overrides(root)]
        safe = [arg for setting in settings for arg in ("-c", setting)]
        output, exit_code, ended_by = run_command(
            ["git", "--no-pager", *safe, *argv],
            cwd=root,
            timeout=_TIMEOUT_SECONDS,
            interrupt=context.interrupt,
            env=_git_env(),
        )
        text = output.text().rstrip()
        if ended_by:
            ending = "timed out" if ended_by == "timeout" else "was interrupted"
            raise ToolExecutionError(f"git {parsed_args.command} {ending}")
        if exit_code != 0:
            raise ToolExecutionError(text or f"git {parsed_args.command} exited {exit_code}")
        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output=text,
            metadata={"command": parsed_args.command, "truncated": output.omitted > 0},
        )


def _argv(args: GitArgs, root: Path) -> List[str]:
    """The git command line for `args`, after checking that each field fits
    its command."""
    command = args.command
    if args.staged and command != "diff":
        raise ToolExecutionError("staged applies to diff only")
    for name in ("stat", "patch"):
        if getattr(args, name) and command in ("status", "merge-base"):
            raise ToolExecutionError(f"{name} does not apply to {command}")
    # The changes as asked for: a summary, the lines, or both.
    changes = [*(["--stat"] if args.stat else []), *(["--patch"] if args.patch else [])]
    revision = _revision(args.revision)
    if command == "status":
        if revision:
            raise ToolExecutionError("status takes no revision")
        argv = ["status", "--short", "--branch", "--untracked-files=all", "--ignore-submodules=all"]
    elif command == "merge-base":
        if not revision:
            raise ToolExecutionError("merge-base needs a revision: the branch to compare HEAD with")
        if args.paths:
            raise ToolExecutionError("merge-base takes no paths")
        return ["merge-base", "HEAD", revision]
    elif command == "diff":
        if args.staged and revision:
            raise ToolExecutionError("staged compares the index with HEAD: give no revision")
        argv = ["diff", *_SAFE_DIFF, *(["--cached"] if args.staged else []), *changes]
        argv += [revision] if revision else []
    elif command == "log":
        argv = ["log", f"--max-count={args.max_count}", "--date=short",
                "--format=%h %ad %an: %s", *_SAFE_DIFF]
        argv += [*changes, *([revision] if revision else [])]
    else:  # show
        argv = ["show", *_SAFE_DIFF, *changes, revision or "HEAD"]
    return [*argv, "--", *(_pathspec(path, root) for path in args.paths)]


def _revision(raw: Optional[str]) -> Optional[str]:
    """A revision as given, refused if git could read it as an option."""
    if raw is None or not raw.strip():
        return None
    revision = raw.strip()
    if revision.startswith("-") or any(ch.isspace() or ch == "\0" for ch in revision):
        raise ToolExecutionError(f"not a revision: {raw!r}")
    return revision


def _pathspec(raw: str, root: Path) -> str:
    """A path inside the repository, relative to it, matched literally."""
    try:
        resolved = resolve_repo_path(root=root, relative_path=raw)
    except ValueError as exc:
        raise ToolExecutionError(str(exc)) from exc
    relative = resolved.relative_to(root).as_posix()
    return f":(literal){relative}"


def _repo_filter_overrides(root: Path) -> List[str]:
    """Settings that switch off each filter driver the repository's own
    configuration defines; filters from the user's global configuration
    (git-lfs, for one) keep working."""
    drivers: set[str] = set()
    for scope in ("--local", "--worktree"):
        listed = subprocess.run(
            ["git", "config", scope, "--includes", "--name-only", "--get-regexp",
             r"^filter\..*\.(clean|smudge|process)$"],
            cwd=root, env=_git_env(), capture_output=True, text=True, timeout=_TIMEOUT_SECONDS,
        )
        for name in listed.stdout.splitlines():
            drivers.add(name.split(".", 1)[1].rsplit(".", 1)[0])
    return [
        f"filter.{driver}.{key}="
        for driver in sorted(drivers)
        for key in ("clean", "smudge", "process")
    ]


def _git_env() -> dict[str, str]:
    """The tool's environment: no GIT_* variable of the user's (a GIT_DIR or
    GIT_EXTERNAL_DIFF would change what runs), no prompt, and no lock taken
    just to read."""
    env = {name: value for name, value in command_env().items() if not name.startswith("GIT_")}
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
    return env
