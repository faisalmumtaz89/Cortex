"""Model and auth service for worker RPC methods."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cortex.cloud import CloudCredentialStore, CloudModelCatalog, CloudRouter
from cortex.cloud.credentials import ENV_KEY_MAP
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider


class ModelService:
    """Backend model/auth operations used by the worker RPC server."""

    def __init__(
        self,
        *,
        config,
        cloud_router: CloudRouter,
        credential_store: CloudCredentialStore,
        cloud_catalog: CloudModelCatalog,
    ) -> None:
        self.config = config
        self.cloud_router = cloud_router
        self.credential_store = credential_store
        self.cloud_catalog = cloud_catalog

        self.active_target = ActiveModelTarget()
        # Last verified turn provenance: what actually answered the most
        # recent turn (set by the session service after verification).
        self.last_turn_provenance: dict | None = None

    def record_turn_provenance(self, *, label: str, verified: bool, record: dict | None) -> None:
        self.last_turn_provenance = {
            "label": label,
            "verified": verified,
            "record": dict(record or {}),
        }

    def get_active_model_label(self) -> str:
        return self.active_target.label

    # ---- listing ---------------------------------------------------------

    def list_models(self) -> Dict[str, Any]:
        auth: Dict[CloudProvider, Tuple[bool, Optional[str]]] = {
            provider: self.cloud_router.get_auth_status(provider)
            for provider in CloudProvider
        }
        refs = self.cloud_catalog.list_models()
        active_ref = self.active_target.cloud_model
        if active_ref is not None and active_ref not in refs:
            refs.append(active_ref)
        cloud = []
        for ref in refs:
            is_auth, source = auth[ref.provider]
            cloud.append(
                {
                    "provider": ref.provider.value,
                    "model_id": ref.model_id,
                    "selector": ref.selector,
                    "authenticated": bool(is_auth),
                    "auth_source": source,
                    "active": ref == active_ref,
                }
            )

        return {
            "active_target": {
                "provider": (
                    self.active_target.cloud_model.provider.value
                    if self.active_target.cloud_model
                    else None
                ),
                "model_id": (
                    self.active_target.cloud_model.model_id
                    if self.active_target.cloud_model
                    else None
                ),
                "label": self.get_active_model_label(),
            },
            "cloud": cloud,
            "providers": [
                {"provider": provider.value, "authenticated": bool(is_auth), "auth_source": source}
                for provider, (is_auth, source) in auth.items()
            ],
        }

    # ---- status -----------------------------------------------------------

    def status_summary(self) -> Dict[str, Any]:
        last_turn = ""
        if self.last_turn_provenance:
            served = self.last_turn_provenance
            suffix = " (verified)" if served.get("verified") else " (unverified)"
            last_turn = f"{served.get('label')}{suffix}"
        return {
            "active_model": self.get_active_model_label(),
            "last_turn_served_by": last_turn,
        }

    # ---- selection ---------------------------------------------------------

    def select_cloud_model(self, provider: str, model_id: str) -> Dict[str, Any]:
        provider_enum = CloudProvider.from_value(provider)
        normalized_model_id = model_id.strip()
        if not normalized_model_id:
            return {"ok": False, "message": "Cloud model ID cannot be empty."}

        is_auth, _source = self.cloud_router.get_auth_status(provider_enum)
        if not is_auth:
            return {
                "ok": False,
                "message": (
                    f"{provider_enum.value} is not authenticated. "
                    f"Run /login {provider_enum.value} <api_key> or set "
                    f"{ENV_KEY_MAP[provider_enum]}."
                ),
            }

        if provider_enum == CloudProvider.AZURE:
            endpoint = self.cloud_router._azure_endpoint()
            if not endpoint:
                return {
                    "ok": False,
                    "message": (
                        "Azure OpenAI endpoint not configured. Set AZURE_OPENAI_ENDPOINT or "
                        "cloud_azure_endpoint in ~/.cortex/config.yaml, then re-select the model."
                    ),
                }
            # Persist so later sessions work without the env var.
            self.config.set_state_value("azure_endpoint", endpoint)

        if provider_enum == CloudProvider.OPENAI_COMPATIBLE:
            base_url = self.cloud_router.openai_compatible_base_url()
            if not base_url:
                return {
                    "ok": False,
                    "message": (
                        "OpenAI-compatible base URL not configured. Set "
                        "OPENAI_COMPATIBLE_BASE_URL or cloud_openai_compatible_base_url in "
                        "~/.cortex/config.yaml, then re-select the model."
                    ),
                }
            self.config.set_state_value("openai_compatible_base_url", base_url)

        ref = CloudModelRef(provider=provider_enum, model_id=normalized_model_id)
        self.active_target = ActiveModelTarget.cloud(ref)

        self.config.set_state_value("last_used_cloud_provider", provider_enum.value)
        self.config.set_state_value("last_used_cloud_model", ref.model_id)
        return {
            "ok": True,
            "message": f"{ref.selector} — now active.",
            "active_model": ref.selector,
        }

    def auth_status(self, provider: CloudProvider) -> Dict[str, Any]:
        summary = self.credential_store.get_auth_summary(provider)
        return {"provider": provider.value, **summary}

    def auth_save_key(self, provider: CloudProvider, api_key: str) -> Dict[str, Any]:
        ok, message = self.credential_store.save_api_key(provider, api_key)
        return {"ok": ok, "message": message}

    def auth_delete_key(self, provider: CloudProvider) -> Dict[str, Any]:
        ok, message = self.credential_store.delete_api_key(provider)
        return {"ok": ok, "message": message}
