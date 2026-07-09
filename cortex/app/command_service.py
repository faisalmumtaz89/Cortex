"""Slash command service for worker mode."""

from __future__ import annotations

import json
import shlex
import time
import urllib.request
from typing import Callable, Dict, cast

from cortex.app.command_output import format_auth_status, format_gpu_status, format_status_summary
from cortex.cloud.types import CloudProvider
from cortex.lumen_runtime import LumenModel

_CLOUD_LOGIN_PROVIDERS = {"openai", "anthropic", "azure"}


class CommandService:
    """Execute slash commands without terminal IO."""

    def __init__(
        self,
        *,
        model_service,
        clear_session: Callable[[str], Dict[str, object]],
        save_session: Callable[[str], Dict[str, object]],
        lumen_runtime,
        update_service=None,
    ) -> None:
        self.model_service = model_service
        self.clear_session = clear_session
        self.save_session = save_session
        self.lumen_runtime = lumen_runtime
        self.update_service = update_service

    @staticmethod
    def _parse_command_args(args: str) -> tuple[bool, list[str] | str]:
        try:
            return True, shlex.split(args)
        except ValueError as exc:
            return False, str(exc)

    @staticmethod
    def _summarize_error_message(message: object, *, max_chars: int = 220) -> str:
        text = str(message or "").strip()
        if not text:
            return "unknown error"
        first_line = text.splitlines()[0].strip()
        summary = first_line or text

        # Collapse duplicated prefixes such as:
        # "Failed to load X: Failed to load X: details"
        while True:
            parts = summary.split(":", 2)
            if len(parts) < 3:
                break
            first = parts[0].strip()
            second = parts[1].strip()
            remainder = parts[2].strip()
            if not first or first != second:
                break
            summary = f"{first}: {remainder}"

        if len(summary) > max_chars:
            return f"{summary[: max_chars - 3].rstrip()}..."
        return summary

    # ---- /download (lumen pull) ------------------------------------------

    def preflight_download(self, args: str) -> Dict[str, object]:
        """Validate /download args against the Lumen catalog (no network)."""
        ok, parsed = self._parse_command_args(args)
        if not ok:
            return {"ok": False, "message": f"Invalid download arguments: {parsed}"}
        parsed_args = parsed if isinstance(parsed, list) else []

        if len(parsed_args) != 1:
            return {"ok": False, "message": "Usage: /download <model[:quant]>"}

        resolved = self.model_service.resolve_local_selector(parsed_args[0])
        if not bool(resolved.get("ok")):
            return {"ok": False, "message": str(resolved.get("message", "Invalid model."))}

        model = cast(LumenModel, resolved["model"])
        return {
            "ok": True,
            "selector": model.selector,
            # Kept for callers that key progress on repo_id.
            "repo_id": model.selector,
            "cached": model.cached,
        }

    def _handle_download(
        self,
        args: str,
        *,
        progress_callback: Callable[[Dict[str, object]], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        activation_guard: Callable[[], bool] | None = None,
    ) -> Dict[str, object]:
        preflight = self.preflight_download(args)
        if not bool(preflight.get("ok")):
            return {
                "ok": False,
                "message": str(preflight.get("message", "Invalid /download command")),
            }

        selector = str(preflight.get("selector", ""))
        if bool(preflight.get("cached")):
            # One mental model: getting a model makes it usable — an
            # already-downloaded selector simply loads.
            load = cast(Dict[str, object], self.model_service.select_local_model(selector))
            if bool(load.get("ok")):
                return {
                    "ok": True,
                    "message": f"Already downloaded: local · {selector} — now active.",
                    "download": {"repo_id": selector, "preexisting": True},
                }
            return {
                "ok": False,
                "message": (
                    f"Already downloaded: local · {selector}, but loading failed: "
                    f"{self._summarize_error_message(load.get('message'))}"
                ),
                "download": {"repo_id": selector, "preexisting": True},
            }

        def on_line(line: str) -> None:
            if progress_callback is not None:
                # Bytes are measured by the caller (cache-side poller); pull's
                # own progress bar is TTY-only and never reaches this pipe.
                progress_callback(
                    {
                        "kind": "download",
                        "repo_id": selector,
                        "phase": line,
                    }
                )

        ok, message = self.lumen_runtime.pull(
            selector, on_line=on_line, cancel_requested=cancel_requested
        )
        if not ok:
            summarized = self._summarize_error_message(message)
            if "cancel" in summarized.lower():
                # A user-initiated cancel is not a failure — say what happened.
                return {
                    "ok": False,
                    "message": "Download cancelled.",
                    "download": {"repo_id": selector, "cancelled": True},
                }
            return {
                "ok": False,
                "message": f"Download failed: {summarized}",
                "download": {"repo_id": selector},
            }

        # Downloaded models auto-load: the download IS the selection.
        if activation_guard is not None and not activation_guard():
            return {
                "ok": True,
                "message": (
                    f"Downloaded local · {selector} (kept inactive — you switched models)."
                ),
                "download": {"repo_id": selector},
            }
        if progress_callback is not None:
            # Distinct kind: the byte phase is over — the UI switches from the
            # download indicator to the GPU-load indicator.
            progress_callback(
                {
                    "kind": "model-load",
                    "repo_id": selector,
                    "phase": "loading",
                }
            )
        load = cast(Dict[str, object], self.model_service.select_local_model(selector))
        if not bool(load.get("ok")):
            return {
                "ok": False,
                "message": (
                    f"Downloaded local · {selector}, but loading failed: "
                    f"{self._summarize_error_message(load.get('message'))}"
                ),
                "download": {"repo_id": selector},
            }
        return {
            "ok": True,
            "message": f"local · {selector} ready — now active.",
            "download": {"repo_id": selector},
        }

    # ---- /benchmark (via the managed Lumen server) -------------------------

    def _handle_benchmark(self, args: str) -> Dict[str, object]:
        ok, parsed = self._parse_command_args(args)
        if not ok:
            return {"ok": False, "message": f"Invalid benchmark arguments: {parsed}"}
        parsed_args = parsed if isinstance(parsed, list) else []

        num_tokens = 100
        prompt = "Once upon a time"
        idx = 0
        while idx < len(parsed_args):
            part = parsed_args[idx]
            if idx == 0 and part.isdigit():
                num_tokens = int(part)
                idx += 1
                continue
            if part in {"-n", "--tokens"}:
                if idx + 1 >= len(parsed_args):
                    return {"ok": False, "message": "Usage: /benchmark [tokens] [--prompt <text>]"}
                value = parsed_args[idx + 1]
                if not value.isdigit():
                    return {"ok": False, "message": "Benchmark tokens must be a positive integer."}
                num_tokens = int(value)
                idx += 2
                continue
            if part == "--prompt":
                if idx + 1 >= len(parsed_args):
                    return {"ok": False, "message": "Usage: /benchmark [tokens] [--prompt <text>]"}
                prompt = parsed_args[idx + 1]
                idx += 2
                continue
            return {"ok": False, "message": "Usage: /benchmark [tokens] [--prompt <text>]"}

        if num_tokens < 1 or num_tokens > 8192:
            return {"ok": False, "message": "Benchmark tokens must be between 1 and 8192."}
        if not prompt.strip():
            return {"ok": False, "message": "Benchmark prompt cannot be empty."}

        active_target = getattr(self.model_service, "active_target", None)
        if getattr(active_target, "backend", "local") == "cloud":
            return {
                "ok": False,
                "message": "Benchmark is currently available for local models only.",
            }
        selector = getattr(active_target, "local_model", None)
        if not selector:
            return {"ok": False, "message": "No local model loaded. Pick one with /model."}

        server_ok, server_message = self.lumen_runtime.ensure_server(selector)
        if not server_ok:
            return {"ok": False, "message": server_message}
        base_url = self.lumen_runtime.base_url()
        if not base_url:
            return {"ok": False, "message": "Lumen server is not running."}

        payload = json.dumps(
            {
                "model": str(selector).split(":", 1)[0],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": num_tokens,
                "temperature": 0.7,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {"ok": False, "message": f"Benchmark request failed: {exc}"}
        elapsed = max(1e-6, time.time() - started)

        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        tokens_per_second = completion_tokens / elapsed if completion_tokens else 0.0

        benchmark = {
            "tokens_generated": completion_tokens,
            "time_elapsed": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_second, 1),
            "requested_tokens": num_tokens,
            "prompt": prompt,
            "model": str(selector),
        }
        return {
            "ok": True,
            "message": (
                "Benchmark complete: "
                f"{tokens_per_second:.1f} tok/s "
                f"({completion_tokens} tokens in {elapsed:.1f}s)"
            ),
            "benchmark": benchmark,
        }

    # ---- /model listing ------------------------------------------------------

    @staticmethod
    def _origin_label(models: Dict[str, object]) -> str:
        raw = models.get("active_target", {}) if isinstance(models, dict) else {}
        active: Dict[str, object] = raw if isinstance(raw, dict) else {}
        label = str(active.get("label", "No model loaded") or "No model loaded")
        backend = str(active.get("backend", "") or "").strip()
        if backend and label != "No model loaded":
            return f"{backend} · {label}"
        return label

    def _handle_model_list(self) -> Dict[str, object]:
        models = self.model_service.list_models()
        active = models.get("active_target", {}) if isinstance(models, dict) else {}
        active_label = str(active.get("label", "No model loaded"))
        backend = str(active.get("backend", "") or "").strip()
        if backend and active_label != "No model loaded":
            active_label = f"{backend} · {active_label}"

        lines: list[str] = [f"Active model: {active_label}", ""]

        local_models = self._sorted_local_models(models if isinstance(models, dict) else {})
        lines.append("Local models (Lumen):")
        if local_models:
            for item in local_models:
                name = str(item.get("name", "unknown"))
                tags: list[str] = []
                if bool(item.get("active")):
                    tags.append("active")
                if bool(item.get("loaded")):
                    tags.append("loaded")
                if not bool(item.get("cached")):
                    tags.append("not downloaded")
                suffix = f" ({', '.join(tags)})" if tags else ""
                lines.append(f"- {name}{suffix}")
        else:
            lines.append(
                "- none (install Lumen: curl -fsSL https://servelumen.com/install.sh | bash)"
            )

        lines.append("")
        lines.append("Cloud models:")
        cloud_models = self._sorted_cloud_models(models if isinstance(models, dict) else {})
        if cloud_models:
            for item in cloud_models:
                selector = self._cloud_selector(item)
                cloud_tags: list[str] = []
                if bool(item.get("active")):
                    cloud_tags.append("active")
                if bool(item.get("authenticated")):
                    cloud_tags.append("ready")
                else:
                    cloud_tags.append("login required")
                lines.append(f"- {selector} ({', '.join(cloud_tags)})")
        else:
            lines.append("- none")

        lines.append("")
        lines.append(
            "Open the picker with /model, or select directly with "
            "/model <name> or /model provider:model."
        )
        return {"ok": True, "message": "\n".join(lines), "models": models}

    @staticmethod
    def _sorted_local_models(models_payload: Dict[str, object]) -> list[Dict[str, object]]:
        raw_local_models = models_payload.get("local", [])
        if not isinstance(raw_local_models, list):
            return []
        local_models = [item for item in raw_local_models if isinstance(item, dict)]
        return sorted(local_models, key=lambda item: str(item.get("name", "")).lower())

    @staticmethod
    def _cloud_selector(item: Dict[str, object]) -> str:
        selector = str(item.get("selector", "")).strip()
        if selector:
            return selector
        provider = str(item.get("provider", "")).strip()
        model_id = str(item.get("model_id", "")).strip()
        if provider and model_id:
            return f"{provider}:{model_id}"
        return "unknown"

    @classmethod
    def _sorted_cloud_models(cls, models_payload: Dict[str, object]) -> list[Dict[str, object]]:
        raw_cloud_models = models_payload.get("cloud", [])
        if not isinstance(raw_cloud_models, list):
            return []
        cloud_models = [item for item in raw_cloud_models if isinstance(item, dict)]
        return sorted(cloud_models, key=lambda item: cls._cloud_selector(item).lower())

    # ---- /setup ----------------------------------------------------------------

    def _handle_setup(self) -> Dict[str, object]:
        models = self.model_service.list_models()
        active = models.get("active_target", {}) if isinstance(models, dict) else {}
        active_label = (
            str(active.get("label", "No model loaded") or "No model loaded").strip()
            or "No model loaded"
        )
        if active_label != "No model loaded":
            return {
                "ok": True,
                "message": f"Setup complete. Active model: {self._origin_label(models)}",
                "models": models,
            }

        local_models = models.get("local", []) if isinstance(models, dict) else []
        first_cached = ""
        if isinstance(local_models, list):
            for item in local_models:
                if isinstance(item, dict) and bool(item.get("cached")):
                    name = str(item.get("name", "")).strip()
                    if name:
                        first_cached = name
                        break

        if first_cached:
            load_result = cast(
                Dict[str, object], self.model_service.select_local_model(first_cached)
            )
            if "message" not in load_result or not str(load_result.get("message", "")).strip():
                if bool(load_result.get("ok")):
                    load_result["message"] = f"Setup complete. Active model: local · {first_cached}"
                else:
                    load_result["message"] = f"Failed to load local model: {first_cached}"
            return load_result

        return {
            "ok": False,
            "message": (
                "No local models downloaded. Pick one with /model — "
                "selecting it downloads and loads it automatically."
            ),
            "models": models,
        }

    # ---- /update ----------------------------------------------------------------

    def _handle_update(
        self,
        args: str,
        *,
        progress_callback: Callable[[Dict[str, object]], None] | None = None,
    ) -> Dict[str, object]:
        if self.update_service is None:
            return {"ok": False, "message": "Updates are not available in this runtime."}
        action = args.strip().lower()
        if action in {"", "status"}:
            return cast(Dict[str, object], self.update_service.status_report())
        if action == "lumen":
            return cast(
                Dict[str, object],
                self.update_service.update_lumen(progress_callback=progress_callback),
            )
        if action == "cortex":
            return cast(
                Dict[str, object],
                self.update_service.update_cortex(progress_callback=progress_callback),
            )
        return {"ok": False, "message": "Usage: /update [lumen|cortex]"}

    # ---- dispatch ---------------------------------------------------------------

    def execute(
        self,
        *,
        session_id: str,
        command: str,
        progress_callback: Callable[[Dict[str, object]], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        activation_guard: Callable[[], bool] | None = None,
    ) -> Dict[str, object]:
        raw = command.strip()
        if not raw:
            return {"ok": False, "message": "Command cannot be empty."}
        if not raw.startswith("/"):
            return {"ok": False, "message": f"Not a slash command: {raw}"}

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"/quit", "/exit"}:
            return {"ok": True, "exit": True}
        if cmd == "/help":
            return {
                "ok": True,
                "message": (
                    "Commands: /help /status /gpu /model [name | provider:model] /clear /save "
                    "/login /download /update [lumen|cortex] /benchmark /setup /quit"
                ),
            }
        if cmd == "/status":
            status = self.model_service.status_summary()
            return {"ok": True, "status": status, "message": format_status_summary(status)}
        if cmd == "/gpu":
            status = self.model_service.gpu_status()
            return {"ok": True, "status": status, "message": format_gpu_status(status)}
        if cmd == "/clear":
            return self.clear_session(session_id)
        if cmd == "/save":
            result = self.save_session(session_id)
            if "message" not in result:
                path = str(result.get("path", "")).strip()
                if path:
                    result["message"] = f"Saved conversation: {path}."
            return result
        if cmd == "/download":
            return self._handle_download(
                args,
                progress_callback=progress_callback,
                cancel_requested=cancel_requested,
                activation_guard=activation_guard,
            )
        if cmd == "/update":
            return self._handle_update(args, progress_callback=progress_callback)
        if cmd == "/benchmark":
            return self._handle_benchmark(args)
        if cmd == "/setup":
            return self._handle_setup()
        if cmd == "/model":
            # The TUI opens its interactive picker for bare /model; this text
            # list is the headless/worker fallback (also `/model list`).
            if not args or args.lower() in {"list", "ls"}:
                return self._handle_model_list()
            selector = args.strip()
            if ":" in selector:
                provider_name = selector.split(":", 1)[0].strip().lower()
                if provider_name in _CLOUD_LOGIN_PROVIDERS:
                    model_id = selector.split(":", 1)[1].strip()
                    if not model_id:
                        return {
                            "ok": False,
                            "message": (
                                "Usage: /model <provider:model> (example: /model openai:gpt-5.1)"
                            ),
                        }
                    return cast(
                        Dict[str, object],
                        self.model_service.select_cloud_model(
                            provider=provider_name, model_id=model_id
                        ),
                    )
                # Anything else with a colon is a Lumen selector (name:quant).
            return cast(Dict[str, object], self.model_service.select_local_model(selector))
        if cmd == "/login":
            if not args:
                return {
                    "ok": False,
                    "message": "Usage: /login openai|anthropic|azure [api_key]",
                }
            login_parts = args.split(maxsplit=1)
            provider_name = login_parts[0].strip().lower()

            if provider_name == "lumen":
                return {
                    "ok": False,
                    "message": "lumen is the managed local engine, not a login target.",
                }
            if provider_name not in _CLOUD_LOGIN_PROVIDERS:
                return {
                    "ok": False,
                    "message": (
                        "Unsupported provider. Use /login openai, /login anthropic, "
                        "or /login azure."
                    ),
                }
            provider = CloudProvider.from_value(provider_name)

            if len(login_parts) == 1:
                auth = self.model_service.auth_status(provider)
                return {
                    "ok": True,
                    "auth": auth,
                    "message": format_auth_status(provider=provider.value, auth=auth),
                }
            result = self.model_service.auth_save_key(provider, login_parts[1])
            if not str(result.get("message", "")).strip():
                result["message"] = f"Saved {provider.value} API key."
            return cast(Dict[str, object], result)

        return {"ok": False, "message": f"Unknown command: {cmd}"}
