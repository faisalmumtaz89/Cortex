"""Model and auth service for worker RPC methods."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, cast

from cortex.cloud import CloudCredentialStore, CloudModelCatalog, CloudRouter
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider


class ModelService:
    """Backend model/auth operations used by the worker RPC server."""

    def __init__(
        self,
        *,
        config,
        model_manager,
        gpu_validator,
        cloud_router: CloudRouter,
        credential_store: CloudCredentialStore,
        cloud_catalog: CloudModelCatalog,
    ) -> None:
        self.config = config
        self.model_manager = model_manager
        self.gpu_validator = gpu_validator
        self.cloud_router = cloud_router
        self.credential_store = credential_store
        self.cloud_catalog = cloud_catalog

        self.active_target = ActiveModelTarget.local(model_name=self.model_manager.current_model)

    def get_active_model_label(self) -> str:
        if self.active_target.backend == "cloud" and self.active_target.cloud_model:
            return self.active_target.cloud_model.selector
        if self.model_manager.current_model:
            return str(self.model_manager.current_model)
        return "No model loaded"

    def list_models(self) -> Dict[str, Any]:
        local_available = self.model_manager.discover_available_models()
        loaded = {item["name"] for item in self.model_manager.list_models()}

        local = []
        for item in local_available:
            entry = dict(item)
            entry["loaded"] = item.get("name") in loaded
            entry["active"] = item.get("name") == self.model_manager.current_model and self.active_target.backend == "local"
            local.append(entry)

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
                "provider": self.active_target.cloud_model.provider.value if self.active_target.cloud_model else None,
                "model_id": self.active_target.cloud_model.model_id if self.active_target.cloud_model else None,
                "label": self.get_active_model_label(),
            },
            "local": local,
            "cloud": cloud,
        }

    def status_summary(self) -> Dict[str, Any]:
        gpu_info = getattr(self.gpu_validator, "gpu_info", None)
        if gpu_info is None:
            try:
                _ok, gpu_info, _errors = self.gpu_validator.validate()
            except Exception:
                gpu_info = None
        return {
            "active_model": self.get_active_model_label(),
            "backend": self.active_target.backend,
            "chip_name": getattr(gpu_info, "chip_name", "unknown"),
            "gpu_cores": getattr(gpu_info, "gpu_cores", 0),
            "total_memory_gb": round(float(getattr(gpu_info, "total_memory", 0)) / (1024**3), 2)
            if gpu_info
            else 0.0,
        }

    def gpu_status(self) -> Dict[str, Any]:
        status = cast(Dict[str, Any], self.model_manager.get_memory_status())
        gpu_info = getattr(self.gpu_validator, "gpu_info", None)
        if gpu_info is not None:
            status["chip_name"] = gpu_info.chip_name
            status["gpu_cores"] = gpu_info.gpu_cores
            status["metal"] = gpu_info.has_metal
            status["mps"] = gpu_info.has_mps
            status["mlx"] = gpu_info.has_mlx
        return status

    def _find_local_model_path(self, model_name_or_path: str) -> Optional[str]:
        candidate = Path(model_name_or_path).expanduser()
        if candidate.exists():
            return str(candidate.resolve())

        for model in self.model_manager.discover_available_models():
            if model.get("name") == model_name_or_path:
                model_path = str(model.get("path", "")).strip()
                if model_path:
                    return model_path
        return None

    def select_local_model(self, model_name_or_path: str) -> Dict[str, Any]:
        model_path = self._find_local_model_path(model_name_or_path)
        if model_path is None:
            return {"ok": False, "message": f"Local model not found: {model_name_or_path}"}

        success, message = self.model_manager.load_model(model_path=model_path)
        if not success:
            return {"ok": False, "message": message}

        model_name = self.model_manager.current_model
        self.active_target = ActiveModelTarget.local(model_name=model_name)
        self.config.set_state_value("last_used_backend", "local")
        if model_name:
            self.config.update_last_used_model(model_name)
        return {"ok": True, "message": message, "active_model": self.get_active_model_label()}

    def select_cloud_model(self, provider: str, model_id: str) -> Dict[str, Any]:
        provider_enum = CloudProvider.from_value(provider)
        ref = CloudModelRef(provider=provider_enum, model_id=model_id.strip())
        self.active_target = ActiveModelTarget.cloud(ref)

        self.config.set_state_value("last_used_backend", "cloud")
        self.config.set_state_value("last_used_cloud_provider", provider_enum.value)
        self.config.set_state_value("last_used_cloud_model", ref.model_id)
        return {"ok": True, "message": f"Active cloud model set to {ref.selector}", "active_model": ref.selector}

    def delete_local_model(self, model_name: str) -> Dict[str, Any]:
        available = self.model_manager.discover_available_models()
        target = next((item for item in available if item.get("name") == model_name), None)
        if target is None:
            return {"ok": False, "message": f"Model not found: {model_name}"}

        if self.model_manager.current_model == model_name:
            unload_ok, unload_msg = self.model_manager.unload_model(model_name)
            if not unload_ok:
                return {"ok": False, "message": unload_msg}

        path = Path(str(target.get("path", ""))).expanduser()
        if not path.exists():
            return {"ok": False, "message": f"Model path no longer exists: {path}"}

        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
        return {"ok": True, "message": f"Deleted local model: {model_name}"}

    def auth_status(self, provider: CloudProvider) -> Dict[str, Any]:
        summary = self.credential_store.get_auth_summary(provider)
        return {"provider": provider.value, **summary}

    def auth_save_key(self, provider: CloudProvider, api_key: str) -> Dict[str, Any]:
        ok, message = self.credential_store.save_api_key(provider, api_key)
        return {"ok": ok, "message": message}

    def auth_delete_key(self, provider: CloudProvider) -> Dict[str, Any]:
        ok, message = self.credential_store.delete_api_key(provider)
        return {"ok": ok, "message": message}
