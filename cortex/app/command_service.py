"""Slash command service for worker mode."""

from __future__ import annotations

import io
import shlex
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, cast

from cortex.app.command_output import format_auth_status, format_gpu_status, format_status_summary
from cortex.cloud.types import CloudProvider


class CommandService:
    """Execute slash commands without terminal IO."""

    def __init__(
        self,
        *,
        model_service,
        clear_session: Callable[[str], Dict[str, object]],
        save_session: Callable[[str], Dict[str, object]],
        model_downloader,
        template_registry,
        inference_engine=None,
    ) -> None:
        self.model_service = model_service
        self.clear_session = clear_session
        self.save_session = save_session
        self.model_downloader = model_downloader
        self.template_registry = template_registry
        self.inference_engine = inference_engine

    @staticmethod
    def _parse_command_args(args: str) -> tuple[bool, list[str] | str]:
        try:
            return True, shlex.split(args)
        except ValueError as exc:
            return False, str(exc)

    @staticmethod
    def _template_config_payload(config: Any) -> Dict[str, object]:
        return {
            "detected_type": getattr(config, "detected_type", "unknown"),
            "user_preference": getattr(config, "user_preference", "auto"),
            "custom_filters": list(getattr(config, "custom_filters", []) or []),
            "show_reasoning": bool(getattr(config, "show_reasoning", False)),
            "confidence": float(getattr(config, "confidence", 0.0) or 0.0),
            "last_updated": str(getattr(config, "last_updated", "") or ""),
        }

    def _handle_download(self, args: str) -> Dict[str, object]:
        raw = args.strip()
        if not raw:
            return {
                "ok": False,
                "message": "Usage: /download <repo_id> [filename] [--load]",
            }

        parts = raw.split()
        should_load = False
        filtered_parts = []
        for part in parts:
            if part == "--load":
                should_load = True
            else:
                filtered_parts.append(part)

        if not filtered_parts:
            return {
                "ok": False,
                "message": "Usage: /download <repo_id> [filename] [--load]",
            }
        if len(filtered_parts) > 2:
            return {
                "ok": False,
                "message": "Too many arguments. Usage: /download <repo_id> [filename] [--load]",
            }

        repo_id = filtered_parts[0].strip()
        filename = filtered_parts[1].strip() if len(filtered_parts) > 1 else None

        if "/" not in repo_id:
            return {
                "ok": False,
                "message": "Invalid format. Expected: username/model-name",
            }

        # The downloader writes progress to stdout/stderr; keep worker RPC channel clean.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            success, message, path = self.model_downloader.download_model(repo_id, filename)

        result: Dict[str, object] = {
            "ok": bool(success),
            "message": str(message),
            "download": {
                "repo_id": repo_id,
                "filename": filename,
                "path": str(path) if path is not None else None,
            },
        }
        if not success:
            return result

        if should_load and path is not None:
            load_result = self.model_service.select_local_model(str(path))
            result["load"] = load_result
            if not load_result.get("ok"):
                result["ok"] = False
                result["message"] = f"{message} (downloaded) but failed to load: {load_result.get('message')}"
            else:
                result["message"] = f"{message} (loaded)"

        return result

    def _handle_template(self, args: str) -> Dict[str, object]:
        model_name = (self.model_service.model_manager.current_model or "").strip()
        if not model_name:
            return {"ok": False, "message": "No model loaded."}

        tokenizer = self.model_service.model_manager.tokenizers.get(model_name)
        parts = args.strip().split()
        subcommand = parts[0].lower() if parts else "auto"

        if subcommand == "status":
            config = self.template_registry.config_manager.get_model_config(model_name)
            if not config:
                return {"ok": False, "message": f"No template configuration for {model_name}"}
            return {
                "ok": True,
                "message": f"Template status for {model_name}",
                "model": model_name,
                "template": self._template_config_payload(config),
            }

        if subcommand == "reset":
            reset_ok = self.template_registry.reset_model_config(model_name)
            if reset_ok:
                return {"ok": True, "message": f"Template configuration reset for {model_name}"}
            return {"ok": False, "message": f"No template configuration found for {model_name}"}

        if subcommand == "list":
            return {
                "ok": True,
                "message": "Available templates listed.",
                "templates": self.template_registry.list_templates(),
            }

        if subcommand not in {"auto", "configure"}:
            return {
                "ok": False,
                "message": "Usage: /template [status|reset|list|auto]",
            }

        profile = self.template_registry.setup_model(
            model_name,
            tokenizer=tokenizer,
            interactive=False,
            force_setup=True,
        )
        config = self.template_registry.config_manager.get_model_config(model_name)

        payload: Dict[str, object] = {
            "ok": True,
            "message": f"Template configured for {model_name}: {profile.config.name}",
            "model": model_name,
            "template_name": profile.config.name,
        }
        if config is not None:
            payload["template"] = self._template_config_payload(config)
        return payload

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
        backend = getattr(active_target, "backend", "local")
        if backend == "cloud":
            return {
                "ok": False,
                "message": "Benchmark is currently available for local models only.",
            }

        model_name = (self.model_service.model_manager.current_model or "").strip()
        if not model_name:
            return {"ok": False, "message": "No model loaded."}

        benchmark_fn = getattr(self.inference_engine, "benchmark", None)
        if not callable(benchmark_fn):
            return {"ok": False, "message": "Benchmark engine is not available in worker mode."}

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            metrics = benchmark_fn(prompt=prompt, num_tokens=num_tokens)
        if metrics is None:
            return {"ok": False, "message": "Benchmark failed to produce metrics."}

        payload = {
            "tokens_generated": int(getattr(metrics, "tokens_generated", 0) or 0),
            "time_elapsed": float(getattr(metrics, "time_elapsed", 0.0) or 0.0),
            "tokens_per_second": float(getattr(metrics, "tokens_per_second", 0.0) or 0.0),
            "first_token_latency": float(getattr(metrics, "first_token_latency", 0.0) or 0.0),
            "gpu_utilization": float(getattr(metrics, "gpu_utilization", 0.0) or 0.0),
            "memory_used_gb": float(getattr(metrics, "memory_used_gb", 0.0) or 0.0),
            "requested_tokens": num_tokens,
            "prompt": prompt,
            "model": model_name,
        }
        return {
            "ok": True,
            "message": (
                "Benchmark complete: "
                f"{payload['tokens_per_second']:.1f} tok/s, "
                f"first token {payload['first_token_latency']:.3f}s, "
                f"memory {payload['memory_used_gb']:.1f}GB"
            ),
            "benchmark": payload,
        }

    def _handle_finetune(self, args: str) -> Dict[str, object]:
        ok, parsed = self._parse_command_args(args)
        if not ok:
            return {"ok": False, "message": f"Invalid finetune arguments: {parsed}"}
        parsed_args = parsed if isinstance(parsed, list) else []

        if not parsed_args or parsed_args[0] in {"help", "--help", "-h"}:
            return {
                "ok": True,
                "message": (
                    "Usage: /finetune status\n"
                    "Interactive fine-tune flow is not yet ported to OpenTUI worker mode."
                ),
            }

        subcommand = parsed_args[0].lower()
        if subcommand == "status":
            from cortex.fine_tuning.mlx_lora_trainer import MLXLoRATrainer

            discover_fn = getattr(self.model_service.model_manager, "discover_available_models", None)
            local_models = discover_fn() if callable(discover_fn) else []
            available = bool(MLXLoRATrainer.is_available())
            current_model = (self.model_service.model_manager.current_model or "").strip()

            payload = {
                "mlx_available": available,
                "current_model": current_model or None,
                "available_local_models": len(local_models),
                "interactive_worker_support": False,
            }
            if available:
                return {
                    "ok": True,
                    "message": "Fine-tuning prerequisites look good. Interactive worker flow is pending migration.",
                    "finetune": payload,
                }
            return {
                "ok": False,
                "message": "Fine-tuning is unavailable because the MLX stack is not available in this environment.",
                "finetune": payload,
            }

        return {
            "ok": False,
            "message": (
                "Interactive fine-tune flow is not yet ported to OpenTUI worker mode. "
                "Run `/finetune status` for readiness checks."
            ),
        }

    def _handle_model_list(self) -> Dict[str, object]:
        models = self.model_service.list_models()
        active = models.get("active_target", {}) if isinstance(models, dict) else {}
        active_label = str(active.get("label", "No model loaded"))

        lines: list[str] = [f"Active model: {active_label}", ""]

        local_models = models.get("local", []) if isinstance(models, dict) else []
        lines.append("Local models:")
        if local_models:
            for item in local_models:
                name = str(item.get("name", "unknown"))
                tags: list[str] = []
                if bool(item.get("active")):
                    tags.append("active")
                if bool(item.get("loaded")):
                    tags.append("loaded")
                suffix = f" ({', '.join(tags)})" if tags else ""
                lines.append(f"- {name}{suffix}")
        else:
            lines.append("- none")

        lines.append("")
        lines.append("Cloud models:")
        cloud_models = models.get("cloud", []) if isinstance(models, dict) else []
        if cloud_models:
            for item in cloud_models:
                selector = str(item.get("selector", "unknown"))
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
        lines.append("Use /model <local-name-or-path> or /model <provider:model> to switch.")
        return {"ok": True, "message": "\n".join(lines), "models": models}

    def execute(self, *, session_id: str, command: str) -> Dict[str, object]:
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
                    "Commands: /help /status /gpu /model [/selector] /models /clear /save /login "
                    "/download /template /finetune /benchmark /quit"
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
                    result["message"] = f"Saved conversation: {path}"
            return result
        if cmd in {"/download", "/template", "/finetune", "/benchmark"}:
            if cmd == "/download":
                return self._handle_download(args)
            if cmd == "/template":
                return self._handle_template(args)
            if cmd == "/benchmark":
                return self._handle_benchmark(args)
            if cmd == "/finetune":
                return self._handle_finetune(args)
        if cmd == "/models":
            return self._handle_model_list()
        if cmd == "/model":
            if not args or args.lower() in {"list", "ls"}:
                return self._handle_model_list()
            if ":" in args:
                provider_raw, model_id = args.split(":", 1)
                provider = CloudProvider.from_value(provider_raw)
                return cast(
                    Dict[str, object],
                    self.model_service.select_cloud_model(provider=provider.value, model_id=model_id.strip()),
                )
            return cast(Dict[str, object], self.model_service.select_local_model(args))
        if cmd == "/login":
            if not args:
                return {"ok": False, "message": "Usage: /login openai|anthropic <api_key>"}
            login_parts = args.split(maxsplit=1)
            if len(login_parts) == 1:
                provider = CloudProvider.from_value(login_parts[0])
                auth = self.model_service.auth_status(provider)
                return {
                    "ok": True,
                    "auth": auth,
                    "message": format_auth_status(provider=provider.value, auth=auth),
                }
            provider = CloudProvider.from_value(login_parts[0])
            result = self.model_service.auth_save_key(provider, login_parts[1])
            if "message" not in result:
                result["message"] = f"Saved {provider.value} API key."
            return cast(Dict[str, object], result)

        return {"ok": False, "message": f"Unknown command: {cmd}"}
