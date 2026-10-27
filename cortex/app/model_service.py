"""Model and auth service for worker RPC methods."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cortex.cloud import CloudCredentialStore, CloudModelCatalog, CloudRouter
from cortex.cloud.credentials import ENV_KEY_MAP
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider

# Providers whose /login also takes the URL requests go to.
LOGIN_URL_ARGUMENT = {
    CloudProvider.AZURE: "<endpoint>",
    CloudProvider.OPENAI_COMPATIBLE: "<base_url>",
}


# How many models picked by name the /model picker keeps listing.
_REMEMBERED_MODELS = 5


URL_ENV_VARIABLE = {
    CloudProvider.AZURE: "AZURE_OPENAI_ENDPOINT",
    CloudProvider.OPENAI_COMPATIBLE: "OPENAI_COMPATIBLE_BASE_URL",
}


def _login_url_hint(provider: CloudProvider) -> str:
    argument = LOGIN_URL_ARGUMENT.get(provider)
    return f" {argument}" if argument else ""


def _setup_variables(provider: CloudProvider) -> str:
    """The environment variables that set a provider up, for messages."""
    names = [ENV_KEY_MAP[provider], *filter(None, [URL_ENV_VARIABLE.get(provider)])]
    return " and ".join(names)


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

    def _selected_selectors(self) -> list[str]:
        saved = self.config.get_state_value("selected_models", [])
        return [str(item) for item in saved] if isinstance(saved, list) else []

    def _selected_elsewhere(self) -> list[CloudModelRef]:
        """Models selected by name that the catalog does not list, newest first."""
        refs = []
        for selector in self._selected_selectors():
            provider, _, model_id = selector.partition(":")
            if not model_id:
                continue
            try:
                refs.append(CloudModelRef(provider=CloudProvider.from_value(provider),
                                          model_id=model_id))
            except ValueError:
                continue
        return refs

    # ---- listing ---------------------------------------------------------

    def list_models(self) -> Dict[str, Any]:
        auth: Dict[CloudProvider, Tuple[bool, Optional[str]]] = {
            provider: self.cloud_router.get_auth_status(provider)
            for provider in CloudProvider
        }
        refs = self.cloud_catalog.list_models()
        active_ref = self.active_target.cloud_model
        for ref in [*self._selected_elsewhere(), active_ref]:
            if ref is not None and ref not in refs:
                refs.append(ref)
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
                {
                    "provider": provider.value,
                    "authenticated": bool(is_auth),
                    "auth_source": source,
                    "url_saved": bool(self.endpoint_url(provider)),
                }
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
            # A saved URL is used again: only the key is missing.
            url_saved = bool(self.endpoint_url(provider_enum))
            url_hint = "" if url_saved else _login_url_hint(provider_enum)
            variables = ENV_KEY_MAP[provider_enum] if url_saved else _setup_variables(provider_enum)
            return {
                "ok": False,
                "message": (
                    f"{provider_enum.value} is not authenticated. "
                    f"Add a key with /login {provider_enum.value} <api_key>{url_hint}, "
                    f"or set {variables}."
                ),
                # The same, for a caller that words it its own way (headless).
                "missing_key": {
                    "provider": provider_enum.value,
                    "url": url_hint.strip(),
                    "variables": variables,
                },
            }

        if provider_enum == CloudProvider.AZURE:
            endpoint = self.cloud_router._azure_endpoint()
            if not endpoint:
                return {
                    "ok": False,
                    "message": (
                        "Azure OpenAI also needs your resource endpoint: "
                        "/login azure <api_key> <endpoint>, or set AZURE_OPENAI_ENDPOINT."
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
                        "An OpenAI-compatible endpoint also needs its base URL: "
                        "/login openai-compatible <api_key> <base_url>, "
                        "or set OPENAI_COMPATIBLE_BASE_URL."
                    ),
                }
            self.config.set_state_value("openai_compatible_base_url", base_url)

        ref = CloudModelRef(provider=provider_enum, model_id=normalized_model_id)
        self.active_target = ActiveModelTarget.cloud(ref)

        self.config.set_state_value("last_used_cloud_provider", provider_enum.value)
        self.config.set_state_value("last_used_cloud_model", ref.model_id)
        if ref not in self.cloud_catalog.list_models():
            # Picked by name (an OpenAI-compatible model, an Azure deployment):
            # the /model picker lists it from now on.
            recent = [ref.selector, *(s for s in self._selected_selectors() if s != ref.selector)]
            self.config.set_state_value("selected_models", recent[:_REMEMBERED_MODELS])
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

    def endpoint_url(self, provider: CloudProvider) -> str:
        """The Azure endpoint or OpenAI-compatible base URL in effect ("" if none)."""
        if provider == CloudProvider.AZURE:
            return self.cloud_router._azure_endpoint()
        if provider == CloudProvider.OPENAI_COMPATIBLE:
            return self.cloud_router.openai_compatible_base_url()
        return ""

    def save_endpoint_url(self, provider: CloudProvider, url: str) -> Dict[str, Any]:
        """Save the Azure endpoint or OpenAI-compatible base URL given to /login.
        The environment variable and config.yaml still take precedence."""
        url = url.strip().rstrip("/")
        if not url.startswith(("https://", "http://")):
            return {
                "ok": False,
                "message": f"The URL must start with https:// or http:// (got {url}).",
            }
        is_azure = provider == CloudProvider.AZURE
        state_key = "azure_endpoint" if is_azure else "openai_compatible_base_url"
        self.config.set_state_value(state_key, url)
        in_effect = self.endpoint_url(provider)
        if in_effect != url:
            variable = URL_ENV_VARIABLE[provider]
            return {
                "ok": True,
                "message": f"Saved {url}, but {in_effect} is in use: {variable} or "
                "~/.cortex/config.yaml sets it, and takes precedence.",
            }
        return {"ok": True, "message": f"Using {url}."}

    def auth_delete_key(self, provider: CloudProvider) -> Dict[str, Any]:
        ok, message = self.credential_store.delete_api_key(provider)
        return {"ok": ok, "message": message}
