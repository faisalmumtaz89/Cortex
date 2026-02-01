"""Simple model downloader for Cortex."""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import requests

try:
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
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
        except:
            pass
        
        return False, None
    
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
                
                if local_path.exists() and any(local_path.iterdir()):
                    return False, f"Model already exists: {local_path}", local_path
                
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_path
                    # Downloads always resume when possible by default
                )
                
                return True, f"Downloaded to {local_path}", local_path
                
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
    
    def list_downloaded_models(self) -> list:
        """List all downloaded models."""
        models = []
        
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