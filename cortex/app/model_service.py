"""Model and auth service for worker RPC methods.

Local models are served by the managed Lumen engine (cortex/lumen_runtime.py):
the catalog is Lumen's registry, selection boots/switches the lumen-server
process, and generation reaches it through the OpenAI-compatible endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from cortex.cloud import CloudCredentialStore, CloudModelCatalog, CloudRouter
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider
from cortex.lumen_runtime import LumenModel, LumenRuntime, parse_selector


class ModelService:
    """Backend model/auth operations used by the worker RPC server."""

    def __init__(
        self,
        *,
        config,
        lumen_runtime: LumenRuntime,
        gpu_validator,
        cloud_router: CloudRouter,
        credential_store: CloudCredentialStore,
        cloud_catalog: CloudModelCatalog,
    ) -> None:
        self.config = config
        self.lumen_runtime = lumen_runtime
        self.gpu_validator = gpu_validator
        self.cloud_router = cloud_router
        self.credential_store = credential_store
        self.cloud_catalog = cloud_catalog

        self.active_target = ActiveModelTarget.local(model_name=None)
        # Last verified turn provenance: what actually answered the most
        # recent turn (set by the session service after verification).
        self.last_turn_provenance: dict | None = None

    def record_turn_provenance(
        self, *, backend: str, label: str, verified: bool, record: dict | None
    ) -> None:
        self.last_turn_provenance = {
            "backend": backend,
            "label": label,
            "verified": verified,
            "record": dict(record or {}),
        }

    def get_active_model_label(self) -> str:
        if self.active_target.backend == "cloud" and self.active_target.cloud_model:
            return self.active_target.cloud_model.selector
        if self.active_target.local_model:
            return str(self.active_target.local_model)
        return "No model loaded"

    # ---- local (Lumen) catalog ------------------------------------------

    def _lumen_catalog(self) -> List[LumenModel]:
        """Lumen's model list merged on (name, quant); cached entries win."""
        merged: Dict[str, LumenModel] = {}
        for model in self.lumen_runtime.list_models():
            key = f"{model.name}:{model.quant}"
            existing = merged.get(key)
            if existing is None:
                merged[key] = model
                continue
            if model.cached and not existing.cached:
                # Keep the richer display name from the availability listing.
                merged[key] = LumenModel(
                    name=model.name,
                    quant=model.quant,
                    cached=True,
                    display_name=existing.display_name or model.display_name,
                    size=model.size,
                )
            elif existing.cached and not model.cached and not existing.display_name:
                merged[key] = LumenModel(
                    name=existing.name,
                    quant=existing.quant,
                    cached=True,
                    display_name=model.display_name,
                    size=existing.size,
                )
        return sorted(merged.values(), key=lambda m: (m.name, m.quant))

    def resolve_local_selector(self, selector: str) -> Dict[str, Any]:
        """Resolve user input to a canonical lumen selector.

        Bare names prefer a CACHED quant (Q8_0 first), falling back to Lumen's
        default Q8_0 when nothing is cached.
        """
        raw = (selector or "").strip()
        if not raw:
            return {"ok": False, "message": "Usage: /model <name[:quant] | provider:model>"}

        catalog = self._lumen_catalog()
        if not catalog:
            ok, message = self.lumen_runtime.available()
            if not ok:
                return {"ok": False, "message": message}
            return {"ok": False, "message": "No Lumen models available."}

        name, quant = parse_selector(raw)
        by_name = [m for m in catalog if m.name == name]
        if not by_name:
            valid = ", ".join(sorted({m.name for m in catalog}))
            return {
                "ok": False,
                "message": f"Unknown local model '{name}'. Lumen supports: {valid}.",
            }

        if ":" in raw:
            match = next((m for m in by_name if m.quant == quant), None)
            if match is None:
                quants = ", ".join(sorted({m.quant.lower() for m in by_name}))
                return {
                    "ok": False,
                    "message": f"Unknown quant '{quant.lower()}' for {name}. Available: {quants}.",
                }
            return {"ok": True, "model": match}

        cached = [m for m in by_name if m.cached]
        pool = cached or by_name
        preferred = sorted(pool, key=lambda m: (m.quant != "Q8_0", m.quant))
        return {"ok": True, "model": preferred[0]}

    # ---- listing ---------------------------------------------------------

    def list_models(self) -> Dict[str, Any]:
        serving = self.lumen_runtime.serving_selector()
        starting = self.lumen_runtime.starting_selector()
        local = []
        for model in self._lumen_catalog():
            selector = model.selector
            local.append(
                {
                    "name": selector,
                    "display_name": model.display_name,
                    "cached": model.cached,
                    "size": model.size,
                    "loaded": serving == selector,
                    "loading": starting == selector,
                    "active": (
                        self.active_target.backend == "local"
                        and self.active_target.local_model == selector
                    ),
                }
            )

        cloud = []
        for ref in self.cloud_catalog.list_models():
            is_auth, source = self.cloud_router.get_auth_status(ref.provider)
            cloud.append(
                {
                    "provider": ref.provider.value,
                    "model_id": ref.model_id,
                    "selector": ref.selector,
                    "authenticated": bool(is_auth),
                    "auth_source": source,
                    "active": (
                        self.active_target.backend == "cloud"
                        and self.active_target.cloud_model is not None
                        and self.active_target.cloud_model.selector == ref.selector
                    ),
                }
            )

        return {
            "active_target": {
                "backend": self.active_target.backend,
                "local_model": self.active_target.local_model,
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
            "local": local,
            "cloud": cloud,
        }

    # ---- status -----------------------------------------------------------

    def _gpu_info(self):
        gpu_info = getattr(self.gpu_validator, "gpu_info", None)
        if gpu_info is None:
            try:
                _ok, gpu_info, _errors = self.gpu_validator.validate()
            except Exception:
                gpu_info = None
        return gpu_info

    def status_summary(self) -> Dict[str, Any]:
        gpu_info = self._gpu_info()
        lumen = self.lumen_runtime.status()
        last_turn = ""
        if self.last_turn_provenance:
            served = self.last_turn_provenance
            suffix = " (verified)" if served.get("verified") else " (unverified)"
            last_turn = f"{served.get('backend')} · {served.get('label')}{suffix}"
        summary: Dict[str, Any] = {
            "active_model": self.get_active_model_label(),
            "backend": self.active_target.backend,
            "last_turn_served_by": last_turn,
            "chip_name": getattr(gpu_info, "chip_name", "unknown"),
            "gpu_cores": getattr(gpu_info, "gpu_cores", 0),
            "total_memory_gb": (
                round(float(getattr(gpu_info, "total_memory", 0)) / (1024**3), 2)
                if gpu_info
                else 0.0
            ),
            "lumen_server": (
                "stopped"
                if not lumen.get("running")
                else (
                    f"serving {lumen.get('selector')} on port {lumen.get('port')}"
                    if lumen.get("ready")
                    else f"starting {lumen.get('selector')}…"
                )
            ),
        }
        return summary

    def gpu_status(self) -> Dict[str, Any]:
        gpu_info = self._gpu_info()

        status: Dict[str, Any] = {}
        if gpu_info is not None:
            status["chip_name"] = gpu_info.chip_name
            status["gpu_cores"] = gpu_info.gpu_cores
            status["metal"] = "yes" if gpu_info.has_metal else "no"

        lumen = self.lumen_runtime.status()
        status["lumen_server"] = (
            "stopped"
            if not lumen.get("running")
            else ("running" if lumen.get("ready") else "starting")
        )
        if lumen.get("running"):
            status["lumen_model"] = lumen.get("selector")
            status["lumen_port"] = lumen.get("port")
            uptime = lumen.get("uptime_seconds")
            if isinstance(uptime, (int, float)):
                status["lumen_uptime_s"] = round(float(uptime), 1)
        return status

    # ---- selection ---------------------------------------------------------

    def select_local_model(self, selector: str) -> Dict[str, Any]:
        resolved = self.resolve_local_selector(selector)
        if not bool(resolved.get("ok")):
            return {"ok": False, "message": str(resolved.get("message", "Invalid selector."))}

        model = cast(LumenModel, resolved["model"])
        canonical = model.selector
        if not model.cached:
            # The worker intercepts this case in the TUI flow and starts the
            # download automatically; this result is the synchronous fallback
            # (direct API/headless callers) and carries a structured marker.
            return {
                "ok": False,
                "not_cached": True,
                "selector": canonical,
                "message": (
                    f"local · {canonical} is not downloaded yet. Select it with /model "
                    "in Cortex to download and load it automatically."
                ),
            }

        already_active = (
            self.active_target.backend == "local"
            and self.active_target.local_model == canonical
            and self.lumen_runtime.serving_selector() == canonical
        )
        ok, message = self.lumen_runtime.ensure_server(canonical)
        if not ok:
            return {"ok": False, "message": message}
        if already_active:
            return {
                "ok": True,
                "message": f"local · {canonical} is already active.",
                "active_model": self.get_active_model_label(),
            }
        return self.activate_local(canonical)

    def activate_local(self, canonical: str) -> Dict[str, Any]:
        """Mark an already-serving local model as the active target."""
        self.active_target = ActiveModelTarget.local(model_name=canonical)
        self.config.set_state_value("last_used_backend", "local")
        self.config.update_last_used_model(canonical)
        return {
            "ok": True,
            "message": f"local · {canonical} ready — now active.",
            "active_model": self.get_active_model_label(),
        }

    def select_cloud_model(self, provider: str, model_id: str) -> Dict[str, Any]:
        provider_enum = CloudProvider.from_value(provider)
        if provider_enum == CloudProvider.LUMEN:
            return {
                "ok": False,
                "message": "lumen is the managed local engine — pick a local model with /model.",
            }
        normalized_model_id = model_id.strip()
        if not normalized_model_id:
            return {"ok": False, "message": "Cloud model ID cannot be empty."}

        is_auth, _source = self.cloud_router.get_auth_status(provider_enum)
        if not is_auth:
            return {
                "ok": False,
                "message": (
                    f"{provider_enum.value} is not authenticated. "
                    f"Run /login {provider_enum.value} <api_key>."
                ),
            }

        if provider_enum == CloudProvider.AZURE:
            endpoint = self.cloud_router._azure_endpoint()
            if not endpoint:
                return {
                    "ok": False,
                    "message": (
                        "Azure OpenAI endpoint not configured. Set AZURE_OPENAI_ENDPOINT "
                        "or cloud_azure_endpoint in config.yaml, then re-select the model."
                    ),
                }
            # Persist so later sessions work without the env var.
            self.config.set_state_value("azure_endpoint", endpoint)

        ref = CloudModelRef(provider=provider_enum, model_id=normalized_model_id)
        self.active_target = ActiveModelTarget.cloud(ref)

        self.config.set_state_value("last_used_backend", "cloud")
        self.config.set_state_value("last_used_cloud_provider", provider_enum.value)
        self.config.set_state_value("last_used_cloud_model", ref.model_id)
        return {
            "ok": True,
            "message": f"cloud · {ref.selector} — now active.",
            "active_model": ref.selector,
        }

    def delete_local_model(self, model_name: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "message": (
                "Local model files are managed by Lumen. "
                "Remove cached models from ~/.cache/lumen with the lumen CLI."
            ),
        }

    def restore_local_target(self, selector: str) -> Optional[str]:
        """Lazily restore a persisted local selection without booting the server.

        Returns the canonical selector when it is still cached, else None.
        """
        resolved = self.resolve_local_selector(selector)
        if not bool(resolved.get("ok")):
            return None
        model = cast(LumenModel, resolved["model"])
        if not model.cached:
            return None
        self.active_target = ActiveModelTarget.local(model_name=model.selector)
        return model.selector

    def auth_status(self, provider: CloudProvider) -> Dict[str, Any]:
        summary = self.credential_store.get_auth_summary(provider)
        return {"provider": provider.value, **summary}

    def auth_save_key(self, provider: CloudProvider, api_key: str) -> Dict[str, Any]:
        ok, message = self.credential_store.save_api_key(provider, api_key)
        return {"ok": ok, "message": message}

    def auth_delete_key(self, provider: CloudProvider) -> Dict[str, Any]:
        ok, message = self.credential_store.delete_api_key(provider)
        return {"ok": ok, "message": message}
