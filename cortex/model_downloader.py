"""Simple model downloader for Cortex."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

hf_constants: Any = None
hf_tqdm: Any = None
try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub import constants as _hf_constants
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    from huggingface_hub.utils import tqdm as _hf_tqdm

    hf_constants = _hf_constants
    hf_tqdm = _hf_tqdm
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class DownloadCancelledError(RuntimeError):
    """Raised when a user cancels an in-flight download."""


class ModelDownloader:
    """Simple model downloader from HuggingFace."""

    def __init__(self, model_path: Path):
        """Initialize downloader with model directory."""
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_repo_dir_name(repo_id: str) -> str:
        normalized = repo_id.strip().replace("\\", "/").strip("/")
        if not normalized:
            return "model"
        safe = normalized.replace("/", "--")
        safe = safe.replace("..", "__")
        safe = safe.replace(":", "-")
        return safe

    def _repo_target_paths(self, repo_id: str) -> tuple[Path, Path]:
        preferred = self.model_path / self._safe_repo_dir_name(repo_id)
        legacy = self.model_path / repo_id.split("/")[-1].strip()
        return preferred, legacy

    def _resolve_repo_download_path(self, repo_id: str) -> Path:
        preferred, legacy = self._repo_target_paths(repo_id)
        if preferred.exists():
            return preferred
        if legacy.exists():
            return legacy
        return preferred

    def _resolve_file_download_path(self, repo_id: str, filename: str) -> tuple[Path, Path]:
        preferred_root = self.model_path / self._safe_repo_dir_name(repo_id)
        preferred = (preferred_root / filename).expanduser().resolve()
        if preferred_root != preferred and preferred_root not in preferred.parents:
            raise ValueError(f"Invalid filename path: {filename}")
        legacy = (self.model_path / filename).expanduser().resolve()
        return preferred, legacy

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _env_positive_int(name: str, *, default: int, minimum: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            parsed = int(raw.strip())
        except (TypeError, ValueError):
            return default
        return max(parsed, minimum)

    @contextmanager
    def _progress_download_runtime_tuning(self, *, enabled: bool):
        if not enabled or hf_constants is None:
            yield
            return

        disable_xet = self._env_flag("CORTEX_HF_DISABLE_XET", True)
        chunk_size = self._env_positive_int(
            "CORTEX_HF_DOWNLOAD_CHUNK_BYTES",
            default=1 * 1024 * 1024,
            minimum=64 * 1024,
        )

        original_chunk_size = getattr(hf_constants, "DOWNLOAD_CHUNK_SIZE", None)
        original_disable_xet = os.getenv("HF_HUB_DISABLE_XET")
        original_disable_xet_constant = getattr(hf_constants, "HF_HUB_DISABLE_XET", None)

        try:
            if isinstance(original_chunk_size, int) and original_chunk_size != chunk_size:
                hf_constants.DOWNLOAD_CHUNK_SIZE = chunk_size
            if disable_xet:
                os.environ["HF_HUB_DISABLE_XET"] = "1"
                if isinstance(original_disable_xet_constant, bool):
                    hf_constants.HF_HUB_DISABLE_XET = True
            yield
        finally:
            if isinstance(original_chunk_size, int):
                hf_constants.DOWNLOAD_CHUNK_SIZE = original_chunk_size
            if isinstance(original_disable_xet_constant, bool):
                hf_constants.HF_HUB_DISABLE_XET = original_disable_xet_constant
            if original_disable_xet is None:
                os.environ.pop("HF_HUB_DISABLE_XET", None)
            else:
                os.environ["HF_HUB_DISABLE_XET"] = original_disable_xet

    def inspect_download_target(
        self, repo_id: str, filename: Optional[str] = None
    ) -> dict[str, object]:
        """Return deterministic local target metadata before network calls."""
        if filename:
            preferred_path, legacy_path = self._resolve_file_download_path(repo_id, filename)
            chosen_path = (
                preferred_path
                if preferred_path.exists() or not legacy_path.exists()
                else legacy_path
            )
            return {
                "kind": "file",
                "path": chosen_path,
                "preferred_path": preferred_path,
                "legacy_path": legacy_path,
                "exists": chosen_path.exists(),
                "resumable": False,
            }

        preferred_path, legacy_path = self._repo_target_paths(repo_id)
        local_path = self._resolve_repo_download_path(repo_id)
        exists = local_path.exists() and local_path.is_dir() and any(local_path.iterdir())
        resumable = exists and self._is_incomplete_snapshot(local_path)
        return {
            "kind": "repo",
            "path": local_path,
            "preferred_path": preferred_path,
            "legacy_path": legacy_path,
            "exists": exists,
            "resumable": resumable,
        }

    def check_auth_status(self) -> Tuple[bool, Optional[str]]:
        """Check if user is authenticated with HuggingFace.

        Returns:
            Tuple of (is_authenticated, username)
        """
        if not HF_HUB_AVAILABLE:
            return False, None

        try:
            api = HfApi()
            user_info = api.whoami()
            if user_info:
                return True, user_info.get("name", "Unknown")
        except Exception:
            pass

        return False, None

    @staticmethod
    def _has_model_weights(model_dir: Path) -> bool:
        """Return True if directory contains recognizable weight files."""
        weight_patterns = (
            "*.safetensors",
            "*.gguf",
            "*.ggml",
            "pytorch_model*.bin",
            "weights.npz",
        )
        for pattern in weight_patterns:
            if any(model_dir.glob(pattern)):
                return True
        return False

    def _is_incomplete_snapshot(self, model_dir: Path) -> bool:
        """Detect interrupted snapshot downloads that should be resumed."""
        if not model_dir.exists() or not model_dir.is_dir():
            return False

        hf_download_cache = model_dir / ".cache" / "huggingface" / "download"
        if hf_download_cache.exists():
            if any(hf_download_cache.glob("*.incomplete")):
                return True
            if any(hf_download_cache.glob("*.lock")):
                return True

        # Some interrupted downloads leave metadata without any model weights.
        has_metadata = (model_dir / "config.json").exists() or (
            model_dir / "model.safetensors.index.json"
        ).exists()
        if has_metadata and not self._has_model_weights(model_dir):
            return True

        return False

    @staticmethod
    def _build_tqdm_class(
        *,
        repo_id: str,
        progress_callback: Optional[Callable[[dict[str, object]], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ):
        if hf_tqdm is None:
            return None
        if progress_callback is None and cancel_requested is None:
            return None

        class _CortexDownloadTqdm(hf_tqdm):  # type: ignore[misc, valid-type]
            @staticmethod
            def _to_non_negative_float(value: object) -> float:
                if isinstance(value, (int, float)):
                    numeric = float(value)
                    if numeric > 0:
                        return numeric
                return 0.0

            @staticmethod
            def _to_positive_float_or_none(value: object) -> float | None:
                if isinstance(value, (int, float)):
                    numeric = float(value)
                    if numeric > 0:
                        return numeric
                return None

            def __init__(self, *args, **kwargs):
                initial = kwargs.get("initial", 0.0)
                total = kwargs.get("total", None)
                desc = kwargs.get("desc", "")
                unit = kwargs.get("unit", "")

                self._cortex_units_downloaded = self._to_non_negative_float(initial)
                self._cortex_units_total = self._to_positive_float_or_none(total)
                self._cortex_desc = str(desc or "").strip()
                self._cortex_unit = str(unit or "").strip()
                super().__init__(*args, **kwargs)
                self._sync_cached_state()

            def _sync_cached_state(self) -> None:
                current_downloaded = self._to_non_negative_float(getattr(self, "n", 0.0))
                if current_downloaded > self._cortex_units_downloaded:
                    self._cortex_units_downloaded = current_downloaded

                current_total = self._to_positive_float_or_none(getattr(self, "total", None))
                if current_total is not None:
                    if self._cortex_units_total is None or current_total > self._cortex_units_total:
                        self._cortex_units_total = current_total

                current_desc = str(getattr(self, "desc", "") or "").strip()
                if current_desc:
                    self._cortex_desc = current_desc

                current_unit = str(getattr(self, "unit", "") or "").strip()
                if current_unit:
                    self._cortex_unit = current_unit

            def _emit_progress(self, *, done: bool = False, check_cancel: bool = True) -> None:
                if check_cancel and callable(cancel_requested) and cancel_requested():
                    raise DownloadCancelledError("Download cancelled by user.")

                if progress_callback is None:
                    return

                try:
                    self._sync_cached_state()
                    downloaded = max(self._cortex_units_downloaded, 0.0)
                    total = self._cortex_units_total
                    if total is not None:
                        percent = (downloaded / total) * 100.0
                        bytes_total: int | None = int(total)
                    else:
                        percent = None
                        bytes_total = None
                    description = self._cortex_desc
                    unit = self._cortex_unit
                    payload: dict[str, object] = {
                        "repo_id": repo_id,
                        "description": description,
                        "unit": unit,
                        "units_downloaded": downloaded,
                        "units_total": total,
                        "bytes_downloaded": int(max(downloaded, 0.0)),
                        "bytes_total": bytes_total,
                        "percent": percent,
                        "done": done,
                    }
                    progress_callback(payload)
                except DownloadCancelledError:
                    raise
                except Exception:
                    # Progress updates are best-effort only.
                    return

            def update(self, n=1):  # type: ignore[override]
                if callable(cancel_requested) and cancel_requested():
                    raise DownloadCancelledError("Download cancelled by user.")
                delta = self._to_non_negative_float(n)
                if delta > 0:
                    self._cortex_units_downloaded += delta
                result = super().update(n)
                self._emit_progress()
                return result

            def refresh(self, *args, **kwargs):  # type: ignore[override]
                if callable(cancel_requested) and cancel_requested():
                    raise DownloadCancelledError("Download cancelled by user.")
                result = super().refresh(*args, **kwargs)
                self._emit_progress()
                return result

            def close(self):  # type: ignore[override]
                self._emit_progress(done=True, check_cancel=False)
                return super().close()

        return _CortexDownloadTqdm

    def download_model(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[dict[str, object]], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Download a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            filename: Optional specific file to download (for GGUF models)

        Returns:
            Tuple of (success, message, local_path)
        """
        if not HF_HUB_AVAILABLE:
            return (
                False,
                "huggingface-hub not installed. Install with: pip install huggingface-hub",
                None,
            )

        try:
            with self._progress_download_runtime_tuning(enabled=progress_callback is not None):
                if filename:
                    preferred_path, legacy_path = self._resolve_file_download_path(
                        repo_id, filename
                    )
                    local_path = (
                        preferred_path
                        if preferred_path.exists() or not legacy_path.exists()
                        else legacy_path
                    )
                    if local_path.exists():
                        return False, f"File already exists: {local_path}", local_path

                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=local_path.parent,
                        tqdm_class=self._build_tqdm_class(
                            repo_id=repo_id,
                            progress_callback=progress_callback,
                            cancel_requested=cancel_requested,
                        ),
                    )
                    downloaded = Path(downloaded_path).expanduser().resolve()
                    return True, f"Downloaded to {downloaded}", downloaded

                local_path = self._resolve_repo_download_path(repo_id)
                local_path.mkdir(parents=True, exist_ok=True)
                resumed = False
                if any(local_path.iterdir()):
                    if self._is_incomplete_snapshot(local_path):
                        resumed = True
                    else:
                        return False, f"Model already exists: {local_path}", local_path

                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_path,
                    tqdm_class=self._build_tqdm_class(
                        repo_id=repo_id,
                        progress_callback=progress_callback,
                        cancel_requested=cancel_requested,
                    ),
                )
                downloaded = Path(downloaded_path).expanduser().resolve()
                if resumed:
                    return True, f"Resumed and downloaded to {downloaded}", downloaded
                return True, f"Downloaded to {downloaded}", downloaded

        except DownloadCancelledError:
            return False, "Download cancelled.", None
        except ValueError as exc:
            return False, f"Download failed: {exc}", None
        except GatedRepoError:
            is_auth, username = self.check_auth_status()
            if is_auth:
                return (
                    False,
                    (
                        f"Model {repo_id} is gated. You're logged in as {username} but may need to accept "
                        f"the model license agreement at https://huggingface.co/{repo_id}"
                    ),
                    None,
                )
            return (
                False,
                (
                    f"Model {repo_id} requires authentication. Run `huggingface-cli login` in your shell "
                    "or set HF_TOKEN, then retry /download."
                ),
                None,
            )
        except RepositoryNotFoundError:
            return False, f"Repository {repo_id} not found on HuggingFace", None
        except Exception as exc:
            return False, f"Download failed: {exc}", None

    def list_downloaded_models(self) -> list[dict[str, object]]:
        """List all downloaded models."""
        models: list[dict[str, object]] = []

        if not self.model_path.exists():
            return models

        for item in self.model_path.iterdir():
            if item.is_file() and item.suffix in [".gguf", ".ggml", ".bin"]:
                size_gb = item.stat().st_size / (1024**3)
                models.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "size_gb": round(size_gb, 2),
                    }
                )
            elif item.is_dir() and any(item.iterdir()):
                total_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                size_gb = total_size / (1024**3)
                models.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "size_gb": round(size_gb, 2),
                    }
                )

        return models
