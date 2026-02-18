"""Simple model downloader for Cortex."""

from pathlib import Path
from typing import Optional, Tuple

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class ModelDownloader:
    """Simple model downloader from HuggingFace."""

    def __init__(self, model_path: Path):
        """Initialize downloader with model directory."""
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_path.mkdir(parents=True, exist_ok=True)

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
                return True, user_info.get('name', 'Unknown')
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
        has_metadata = (model_dir / "config.json").exists() or (model_dir / "model.safetensors.index.json").exists()
        if has_metadata and not self._has_model_weights(model_dir):
            return True

        return False

    def download_model(self, repo_id: str, filename: Optional[str] = None) -> Tuple[bool, str, Optional[Path]]:
        """
        Download a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            filename: Optional specific file to download (for GGUF models)

        Returns:
            Tuple of (success, message, local_path)
        """
        if not HF_HUB_AVAILABLE:
            return False, "huggingface-hub not installed. Install with: pip install huggingface-hub", None

        try:
            if filename:
                # Download single file
                print(f"Downloading {filename} from {repo_id}...")
                local_path = self.model_path / filename

                if local_path.exists():
                    return False, f"File already exists: {local_path}", local_path

                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.model_path
                    # Downloads always resume when possible by default
                )

                return True, f"Downloaded to {local_path}", Path(downloaded_path)

            else:
                # Download entire repository
                model_name = repo_id.split('/')[-1]
                local_path = self.model_path / model_name

                print(f"Downloading repository {repo_id}...")

                resumed = False
                if local_path.exists() and any(local_path.iterdir()):
                    if self._is_incomplete_snapshot(local_path):
                        resumed = True
                    else:
                        return False, f"Model already exists: {local_path}", local_path

                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_path
                    # Downloads always resume when possible by default
                )

                if resumed:
                    return True, f"Resumed and downloaded to {local_path}", Path(downloaded_path)
                return True, f"Downloaded to {local_path}", Path(downloaded_path)

        except GatedRepoError:
            # Check if user is logged in
            is_auth, username = self.check_auth_status()
            if is_auth:
                return False, f"Model {repo_id} is gated. You're logged in as {username} but may need to accept the model's license agreement at https://huggingface.co/{repo_id}", None
            else:
                return False, f"Model {repo_id} requires authentication. Please use /login command to authenticate with HuggingFace", None
        except RepositoryNotFoundError:
            return False, f"Repository {repo_id} not found on HuggingFace", None
        except Exception as e:
            return False, f"Download failed: {str(e)}", None

    def list_downloaded_models(self) -> list[dict[str, object]]:
        """List all downloaded models."""
        models: list[dict[str, object]] = []

        if not self.model_path.exists():
            return models

        for item in self.model_path.iterdir():
            if item.is_file() and item.suffix in ['.gguf', '.ggml', '.bin']:
                size_gb = item.stat().st_size / (1024**3)
                models.append({
                    'name': item.name,
                    'path': str(item),
                    'size_gb': round(size_gb, 2)
                })
            elif item.is_dir() and any(item.iterdir()):
                total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                size_gb = total_size / (1024**3)
                models.append({
                    'name': item.name,
                    'path': str(item),
                    'size_gb': round(size_gb, 2)
                })

        return models
