"""Slash command service for worker mode."""

from __future__ import annotations

from typing import Callable, Dict, cast

from cortex.app.command_output import format_auth_status, format_status_summary
from cortex.cloud.types import CloudProvider

_CLOUD_LOGIN_PROVIDERS = {"openai", "anthropic", "azure", "openai-compatible"}
_MODEL_USAGE = "Usage: /model <provider>:<model> (example: /model openai:gpt-5.5)"


class CommandService:
    """Execute slash commands without terminal IO."""

    def __init__(
        self,
        *,
        model_service,
        clear_session: Callable[[str], Dict[str, object]],
        save_session: Callable[[str], Dict[str, object]],
        update_service=None,
    ) -> None:
        self.model_service = model_service
        self.clear_session = clear_session
        self.save_session = save_session
        self.update_service = update_service

    # ---- /model listing ------------------------------------------------------

    def _handle_model_list(self) -> Dict[str, object]:
        models = self.model_service.list_models()
        active = models.get("active_target", {}) if isinstance(models, dict) else {}
        active_label = str(active.get("label", "No model loaded"))

        lines: list[str] = [f"Active model: {active_label}", "", "Models:"]
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
        lines.append("Open the picker with /model, or select directly with /model provider:model.")
        return {"ok": True, "message": "\n".join(lines), "models": models}

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
        if action == "cortex":
            return cast(
                Dict[str, object],
                self.update_service.update_cortex(progress_callback=progress_callback),
            )
        return {"ok": False, "message": "Usage: /update [cortex]"}

    # ---- dispatch ---------------------------------------------------------------

    def execute(
        self,
        *,
        session_id: str,
        command: str,
        progress_callback: Callable[[Dict[str, object]], None] | None = None,
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
                    "Commands: /help /status /model [provider:model] /clear /save /login "
                    "/update [cortex] /quit"
                ),
            }
        if cmd == "/status":
            status = self.model_service.status_summary()
            return {"ok": True, "status": status, "message": format_status_summary(status)}
        if cmd == "/clear":
            return self.clear_session(session_id)
        if cmd == "/save":
            result = self.save_session(session_id)
            if "message" not in result:
                path = str(result.get("path", "")).strip()
                if path:
                    result["message"] = f"Saved conversation: {path}."
            return result
        if cmd == "/update":
            return self._handle_update(args, progress_callback=progress_callback)
        if cmd == "/model":
            # The TUI opens its interactive picker for bare /model; this text
            # list is the headless/worker fallback (also `/model list`).
            if not args or args.lower() in {"list", "ls"}:
                return self._handle_model_list()
            provider_name, _, model_id = args.partition(":")
            provider_name = provider_name.strip().lower()
            model_id = model_id.strip()
            if not model_id:
                return {"ok": False, "message": _MODEL_USAGE}
            if provider_name not in _CLOUD_LOGIN_PROVIDERS:
                return {
                    "ok": False,
                    "message": (
                        f"Unsupported provider '{provider_name}'. Use one of: "
                        f"{', '.join(sorted(_CLOUD_LOGIN_PROVIDERS))}."
                    ),
                }
            return cast(
                Dict[str, object],
                self.model_service.select_cloud_model(provider=provider_name, model_id=model_id),
            )
        if cmd == "/login":
            if not args:
                return {
                    "ok": False,
                    "message": "Usage: /login openai|anthropic|azure|openai-compatible [api_key]",
                }
            login_parts = args.split(maxsplit=1)
            provider_name = login_parts[0].strip().lower()

            if provider_name not in _CLOUD_LOGIN_PROVIDERS:
                return {
                    "ok": False,
                    "message": (
                        "Unsupported provider. Use /login openai, /login anthropic, "
                        "/login azure, or /login openai-compatible."
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
