"""Tool runner and specifications for Cortex."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cortex.tools.errors import ToolError, ValidationError
from cortex.tools.fs_ops import RepoFS
from cortex.tools.search import RepoSearch


ConfirmCallback = Callable[[str], bool]


class ToolRunner:
    """Execute tool calls with safety checks."""

    def __init__(self, root: Path, confirm_callback: Optional[ConfirmCallback] = None) -> None:
        self.fs = RepoFS(root)
        self.search = RepoSearch(self.fs)
        self.confirm_callback = confirm_callback

    def set_confirm_callback(self, callback: ConfirmCallback) -> None:
        self.confirm_callback = callback

    def tool_spec(self) -> Dict[str, Any]:
        return {
            "list_dir": {"args": {"path": "string", "recursive": "bool", "max_depth": "int"}},
            "read_file": {"args": {"path": "string", "start_line": "int", "end_line": "int", "max_bytes": "int"}},
            "search": {"args": {"query": "string", "path": "string", "use_regex": "bool", "max_results": "int"}},
            "write_file": {"args": {"path": "string", "content": "string", "expected_sha256": "string"}},
            "create_file": {"args": {"path": "string", "content": "string", "overwrite": "bool"}},
            "delete_file": {"args": {"path": "string"}},
            "replace_in_file": {"args": {"path": "string", "old": "string", "new": "string", "expected_replacements": "int"}},
            "insert_after": {"args": {"path": "string", "anchor": "string", "content": "string", "expected_matches": "int"}},
            "insert_before": {"args": {"path": "string", "anchor": "string", "content": "string", "expected_matches": "int"}},
        }

    def tool_instructions(self) -> str:
        spec = json.dumps(self.tool_spec(), ensure_ascii=True, indent=2)
        repo_root = str(self.fs.root)
        return (
            "[CORTEX_TOOL_INSTRUCTIONS v2]\n"
            "You have access to file tools. If a tool is required, respond ONLY with a <tool_calls> JSON block.\n"
            "Do not include any other text when calling tools.\n"
            f"Repo root: {repo_root}\n"
            "All paths must be relative to the repo root (use '.' for root). Do not use absolute paths or ~.\n"
            "If you are unsure about paths, call list_dir with path '.' first.\n"
            "Format:\n"
            "<tool_calls>{\"calls\":[{\"id\":\"call_1\",\"name\":\"tool_name\",\"arguments\":{...}}]}</tool_calls>\n"
            "Available tools:\n"
            f"{spec}"
        )

    def run_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for call in calls:
            call_id = call.get("id", "unknown")
            name = call.get("name")
            args = call.get("arguments") or {}
            try:
                if name == "list_dir":
                    result = self.fs.list_dir(**args)
                elif name == "read_file":
                    result = self.fs.read_text(**args)
                elif name == "search":
                    result = self.search.search(**args)
                elif name == "write_file":
                    result = self._write_file(**args)
                elif name == "create_file":
                    result = self._create_file(**args)
                elif name == "delete_file":
                    result = self._delete_file(**args)
                elif name == "replace_in_file":
                    result = self._replace_in_file(**args)
                elif name == "insert_after":
                    result = self._insert_relative(after=True, **args)
                elif name == "insert_before":
                    result = self._insert_relative(after=False, **args)
                else:
                    raise ValidationError(f"unknown tool: {name}")
                results.append({"id": call_id, "name": name, "ok": True, "result": result, "error": None})
            except Exception as e:
                results.append({"id": call_id, "name": name, "ok": False, "result": None, "error": str(e)})
        return results

    def _write_file(self, path: str, content: str, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
        before = self.fs.read_full_text(path)
        self._confirm_change(path, before, content, "write")
        return self.fs.write_text(path, content, expected_sha256=expected_sha256)

    def _create_file(self, path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
        before = ""
        self._confirm_change(path, before, content, "create")
        return self.fs.create_text(path, content, overwrite=overwrite)

    def _delete_file(self, path: str) -> Dict[str, Any]:
        before = self.fs.read_full_text(path)
        self._confirm_change(path, before, "", "delete")
        return self.fs.delete_file(path)

    def _replace_in_file(self, path: str, old: str, new: str, expected_replacements: int = 1) -> Dict[str, Any]:
        if not old:
            raise ValidationError("old must be a non-empty string")
        content = self.fs.read_full_text(path)
        count = content.count(old)
        if count != expected_replacements:
            raise ToolError(f"expected {expected_replacements} replacements, found {count}")
        updated = content.replace(old, new)
        self._confirm_change(path, content, updated, "replace")
        return self.fs.write_text(path, updated)

    def _insert_relative(self, path: str, anchor: str, content: str, expected_matches: int = 1, after: bool = True) -> Dict[str, Any]:
        if not anchor:
            raise ValidationError("anchor must be a non-empty string")
        original = self.fs.read_full_text(path)
        count = original.count(anchor)
        if count != expected_matches:
            raise ToolError(f"expected {expected_matches} matches, found {count}")
        insert_text = anchor + content if after else content + anchor
        updated = original.replace(anchor, insert_text, count if expected_matches > 1 else 1)
        self._confirm_change(path, original, updated, "insert")
        return self.fs.write_text(path, updated)

    def _confirm_change(self, path: str, before: str, after: str, action: str) -> None:
        if self.confirm_callback is None:
            raise ToolError("confirmation required but no callback configured")
        if before == after:
            raise ToolError("no changes to apply")
        diff = "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"{path} (before)",
                tofile=f"{path} (after)",
                lineterm="",
            )
        )
        prompt = f"Apply {action} to {path}?\n{diff}\n"
        if not self.confirm_callback(prompt):
            raise ToolError("change declined by user")
