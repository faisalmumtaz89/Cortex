"""Slash command service for worker mode."""

from __future__ import annotations

import io
import shlex
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
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

    def preflight_download(self, args: str) -> Dict[str, object]:
        """Validate and normalize /download args without performing network IO."""
        ok, parsed = self._parse_command_args(args)
        if not ok:
            return {"ok": False, "message": f"Invalid download arguments: {parsed}"}
        parsed_args = parsed if isinstance(parsed, list) else []

        if not parsed_args:
            return {
                "ok": False,
                "message": "Usage: /download <repo_id> [filename] [--load]",
            }

        should_load = False
        filtered_parts: list[str] = []
        for part in parsed_args:
            if part == "--load":
                should_load = True
            elif part.startswith("--"):
                return {
                    "ok": False,
                    "message": f"Unknown option: {part}. Usage: /download <repo_id> [filename] [--load]",
                }
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

        return {
            "ok": True,
            "repo_id": repo_id,
            "filename": filename,
            "should_load": should_load,
        }

    @staticmethod
    def _huggingface_status() -> Dict[str, object]:
        try:
            from huggingface_hub import HfApi
        except Exception:
            return {
                "ok": False,
                "message": "huggingface-hub not installed. Install with: pip install huggingface-hub",
            }

        try:
            user_info = HfApi().whoami()
        except Exception:
            return {
                "ok": False,
                "message": (
                    "HuggingFace is not authenticated. Run `huggingface-cli login` in your shell "
                    "or set HF_TOKEN, then retry /download."
                ),
            }

        username = str((user_info or {}).get("name", "Unknown")).strip() or "Unknown"
        return {
            "ok": True,
            "auth": {
                "provider": "huggingface",
                "authenticated": True,
                "username": username,
            },
            "message": (
                "Authentication status\n"
                "- Provider: huggingface\n"
                "- Authenticated: True\n"
                f"- Username: {username}"
            ),
        }

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
            return f"{summary[:max_chars - 3].rstrip()}..."
        return summary

    def _handle_download(
        self,
        args: str,
        *,
        progress_callback: Callable[[Dict[str, object]], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Dict[str, object]:
        preflight = self.preflight_download(args)
        if not bool(preflight.get("ok")):
            return {
                "ok": False,
                "message": str(preflight.get("message", "Invalid /download command")),
            }

        repo_id = str(preflight.get("repo_id", "")).strip()
        filename_raw = preflight.get("filename")
        filename = str(filename_raw).strip() if isinstance(filename_raw, str) else None
        should_load = bool(preflight.get("should_load"))

        if not repo_id:
            return {"ok": False, "message": "Usage: /download <repo_id> [filename] [--load]"}

        download_kind = "file" if filename else "repo"
        existing_message_template = (
            "File already exists: {path}"
            if download_kind == "file"
            else "Model already exists: {path}"
        )

        inspected_target: Dict[str, object] | None = None
        inspect_target = getattr(self.model_downloader, "inspect_download_target", None)
        if callable(inspect_target):
            try:
                raw_inspected = inspect_target(repo_id, filename)
                if isinstance(raw_inspected, dict):
                    inspected_target = raw_inspected
            except Exception:
                inspected_target = None

        target_path: Path | None = None
        target_exists = False
        target_resumable = False
        if inspected_target is not None:
            candidate = inspected_target.get("path")
            if isinstance(candidate, Path):
                target_path = candidate
            elif isinstance(candidate, str) and candidate.strip():
                target_path = Path(candidate).expanduser().resolve()
            target_exists = bool(inspected_target.get("exists"))
            target_resumable = bool(inspected_target.get("resumable"))

        if target_exists and target_path is not None and not target_resumable:
            existing_message = existing_message_template.format(path=target_path)
            existing_result: Dict[str, object] = {
                "ok": False,
                "message": existing_message,
                "download": {
                    "repo_id": repo_id,
                    "filename": filename,
                    "path": str(target_path),
                    "preexisting": True,
                },
            }
            if should_load:
                load_result = self.model_service.select_local_model(str(target_path))
                existing_result["load"] = load_result
                if bool(load_result.get("ok")):
                    existing_result["ok"] = True
                    existing_result["message"] = f"{existing_message} (loaded)"
                else:
                    load_error = self._summarize_error_message(load_result.get("message"))
                    existing_result["message"] = (
                        f"{existing_message} but failed to load: {load_error}"
                    )
            return existing_result

        # The downloader writes progress to stdout/stderr; keep worker RPC channel clean.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                success, message, path = self.model_downloader.download_model(
                    repo_id,
                    filename,
                    progress_callback=progress_callback,
                    cancel_requested=cancel_requested,
                )
            except TypeError:
                try:
                    # Backward compatibility for downloader stubs that do not support cancel_requested.
                    success, message, path = self.model_downloader.download_model(
                        repo_id,
                        filename,
                        progress_callback=progress_callback,
                    )
                except TypeError:
                    # Backward compatibility for downloader stubs that do not support callbacks.
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
                load_error = self._summarize_error_message(load_result.get("message"))
                result["ok"] = False
                result["message"] = f"{message} (downloaded) but failed to load: {load_error}"
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

            discover_fn = getattr(
                self.model_service.model_manager, "discover_available_models", None
            )
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

        local_models = self._sorted_local_models(models if isinstance(models, dict) else {})
        lines.append("Local models:")
        next_index = 1
        if local_models:
            for item in local_models:
                name = str(item.get("name", "unknown"))
                tags: list[str] = []
                if bool(item.get("active")):
                    tags.append("active")
                if bool(item.get("loaded")):
                    tags.append("loaded")
                suffix = f" ({', '.join(tags)})" if tags else ""
                lines.append(f"- [{next_index}] {name}{suffix}")
                next_index += 1
        else:
            lines.append("- none")

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
                lines.append(f"- [{next_index}] {selector} ({', '.join(cloud_tags)})")
                next_index += 1
        else:
            lines.append("- none")

        lines.append("")
        lines.append(
            "Use /model <number> to select from this list, or /model <local-name|provider:model>."
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

    @classmethod
    def _build_model_index_entries(cls, models_payload: Dict[str, object]) -> list[Dict[str, str]]:
        entries: list[Dict[str, str]] = []
        for item in cls._sorted_local_models(models_payload):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            entries.append(
                {
                    "backend": "local",
                    "model_name_or_path": name,
                }
            )

        for item in cls._sorted_cloud_models(models_payload):
            selector = cls._cloud_selector(item)
            provider = str(item.get("provider", "")).strip()
            model_id = str(item.get("model_id", "")).strip()
            if not provider or not model_id:
                if ":" in selector:
                    derived_provider, derived_model_id = selector.split(":", 1)
                    provider = provider or derived_provider.strip()
                    model_id = model_id or derived_model_id.strip()
            if not provider or not model_id:
                continue
            entries.append(
                {
                    "backend": "cloud",
                    "provider": provider,
                    "model_id": model_id,
                }
            )
        return entries

    @staticmethod
    def _parse_numeric_model_selector(selector: str) -> int | None:
        normalized = selector.strip()
        if not normalized:
            return None
        if normalized.startswith("#"):
            normalized = normalized[1:].strip()
        if not normalized or not normalized.isdigit():
            return None
        return int(normalized)

    def _resolve_model_index_selector(
        self, *, selector: str, models_payload: Dict[str, object]
    ) -> Dict[str, object]:
        parsed_index = self._parse_numeric_model_selector(selector)
        if parsed_index is None:
            return {"ok": False, "message": f"Invalid model index: {selector}"}

        entries = self._build_model_index_entries(models_payload)
        if parsed_index < 1 or parsed_index > len(entries):
            return {
                "ok": False,
                "message": f"Model index out of range: {selector}. Run /model to list available models.",
            }

        entry = entries[parsed_index - 1]
        backend = entry.get("backend", "")
        if backend == "cloud":
            provider = str(entry.get("provider", "")).strip()
            model_id = str(entry.get("model_id", "")).strip()
            if not provider or not model_id:
                return {
                    "ok": False,
                    "message": f"Model index out of range: {selector}. Run /model to list available models.",
                }
            return {
                "ok": True,
                "backend": "cloud",
                "provider": provider,
                "model_id": model_id,
            }

        local_model = str(entry.get("model_name_or_path", "")).strip()
        if not local_model:
            return {
                "ok": False,
                "message": f"Model index out of range: {selector}. Run /model to list available models.",
            }
        return {
            "ok": True,
            "backend": "local",
            "model_name_or_path": local_model,
        }

    @staticmethod
    def _extract_local_model_names(models_payload: Dict[str, object]) -> list[str]:
        names: list[str] = []
        for item in CommandService._sorted_local_models(models_payload):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
        return names

    def _resolve_local_model_selector(
        self,
        selector: str,
        *,
        models_payload: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        normalized = selector.strip()
        if not normalized:
            return {
                "ok": False,
                "message": "Usage: /model <number|local-name|local-path|provider:model>",
            }

        looks_like_path = normalized.startswith(("/", "./", "../", "~"))
        if looks_like_path:
            return {"ok": True, "model_name_or_path": normalized}

        models = (
            models_payload if isinstance(models_payload, dict) else self.model_service.list_models()
        )
        local_names = self._extract_local_model_names(models if isinstance(models, dict) else {})
        if not local_names:
            return {"ok": False, "message": "No local models found. Run /download first."}

        lowered = normalized.lower()
        exact = [name for name in local_names if name.lower() == lowered]
        if len(exact) == 1:
            return {"ok": True, "model_name_or_path": exact[0]}

        prefix_matches = [name for name in local_names if name.lower().startswith(lowered)]
        if len(prefix_matches) == 1:
            return {"ok": True, "model_name_or_path": prefix_matches[0]}

        contains_matches = [name for name in local_names if lowered in name.lower()]
        if len(contains_matches) == 1:
            return {"ok": True, "model_name_or_path": contains_matches[0]}

        if prefix_matches or contains_matches:
            matches = prefix_matches if prefix_matches else contains_matches
            preview = matches[:8]
            lines = [
                f"Ambiguous local model selector '{normalized}'. Matches:",
            ]
            for name in preview:
                lines.append(f"- {name}")
            lines.append("Use /model <number> from `/model` list for exact selection.")
            return {"ok": False, "message": "\n".join(lines)}

        return {
            "ok": False,
            "message": f"No local model matches '{normalized}'. Run /model to list available models.",
        }

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
                "message": f"Setup complete. Active model: {active_label}",
                "models": models,
            }

        local_models = models.get("local", []) if isinstance(models, dict) else []
        first_local = ""
        if isinstance(local_models, list):
            for item in local_models:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name:
                    first_local = name
                    break

        if first_local:
            load_result = cast(
                Dict[str, object], self.model_service.select_local_model(first_local)
            )
            if "message" not in load_result or not str(load_result.get("message", "")).strip():
                if bool(load_result.get("ok")):
                    load_result["message"] = f"Loaded local model: {first_local}"
                else:
                    load_result["message"] = f"Failed to load local model: {first_local}"
            return load_result

        return {
            "ok": False,
            "message": "No local model installed. Run: /download mlx-community/Nanbeige4.1-3B-bf16 --load",
            "models": models,
        }

    def execute(
        self,
        *,
        session_id: str,
        command: str,
        progress_callback: Callable[[Dict[str, object]], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
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
                    "Commands: /help /status /gpu /model [/selector] /models /clear /save /login "
                    "/download /template /finetune /benchmark /setup /quit"
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
                return self._handle_download(
                    args,
                    progress_callback=progress_callback,
                    cancel_requested=cancel_requested,
                )
            if cmd == "/template":
                return self._handle_template(args)
            if cmd == "/benchmark":
                return self._handle_benchmark(args)
            if cmd == "/finetune":
                return self._handle_finetune(args)
        if cmd == "/setup":
            return self._handle_setup()
        if cmd == "/models":
            return self._handle_model_list()
        if cmd == "/model":
            if not args or args.lower() in {"list", "ls"}:
                return self._handle_model_list()
            selector = args.strip()
            if ":" in selector:
                provider_raw, model_id = selector.split(":", 1)
                provider_name = provider_raw.strip()
                selector_model_id = model_id.strip()
                if not provider_name or not selector_model_id:
                    return {
                        "ok": False,
                        "message": "Usage: /model <provider:model> (example: /model openai:gpt-5.1)",
                    }
                try:
                    provider = CloudProvider.from_value(provider_name)
                except ValueError:
                    return {
                        "ok": False,
                        "message": "Unsupported cloud provider. Use openai or anthropic.",
                    }
                return cast(
                    Dict[str, object],
                    self.model_service.select_cloud_model(
                        provider=provider.value, model_id=selector_model_id
                    ),
                )

            models = self.model_service.list_models()
            if self._parse_numeric_model_selector(selector) is not None:
                indexed_selector = self._resolve_model_index_selector(
                    selector=selector, models_payload=models
                )
                if not bool(indexed_selector.get("ok")):
                    return {
                        "ok": False,
                        "message": str(
                            indexed_selector.get("message", "Failed to resolve model selector.")
                        ),
                    }
                if str(indexed_selector.get("backend", "")) == "cloud":
                    provider_name = str(indexed_selector.get("provider", "")).strip()
                    model_id = str(indexed_selector.get("model_id", "")).strip()
                    if not provider_name or not model_id:
                        return {
                            "ok": False,
                            "message": f"Model index out of range: {selector}. Run /model to list available models.",
                        }
                    return cast(
                        Dict[str, object],
                        self.model_service.select_cloud_model(
                            provider=provider_name, model_id=model_id
                        ),
                    )
                resolved = str(indexed_selector.get("model_name_or_path", "")).strip()
                return cast(Dict[str, object], self.model_service.select_local_model(resolved))

            local_selector = self._resolve_local_model_selector(selector, models_payload=models)
            if not bool(local_selector.get("ok")):
                return {
                    "ok": False,
                    "message": str(
                        local_selector.get("message", "Failed to resolve local model selector.")
                    ),
                }
            resolved = str(local_selector.get("model_name_or_path", "")).strip()
            return cast(Dict[str, object], self.model_service.select_local_model(resolved))
        if cmd == "/login":
            if not args:
                return {
                    "ok": False,
                    "message": "Usage: /login openai|anthropic <api_key> OR /login huggingface",
                }
            login_parts = args.split(maxsplit=1)
            provider_name = login_parts[0].strip().lower()

            if provider_name in {"huggingface", "hf"}:
                if len(login_parts) > 1:
                    return {
                        "ok": False,
                        "message": (
                            "Do not paste HuggingFace tokens into chat. "
                            "Run `huggingface-cli login` in your shell, then retry /download."
                        ),
                    }
                return self._huggingface_status()

            try:
                provider = CloudProvider.from_value(provider_name)
            except ValueError:
                return {
                    "ok": False,
                    "message": "Unsupported provider. Use /login openai, /login anthropic, or /login huggingface.",
                }

            if len(login_parts) == 1:
                auth = self.model_service.auth_status(provider)
                return {
                    "ok": True,
                    "auth": auth,
                    "message": format_auth_status(provider=provider.value, auth=auth),
                }
            result = self.model_service.auth_save_key(provider, login_parts[1])
            if "message" not in result:
                result["message"] = f"Saved {provider.value} API key."
            return cast(Dict[str, object], result)

        return {"ok": False, "message": f"Unknown command: {cmd}"}
