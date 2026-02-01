"""Compatibility helpers for MLX / mlx_lm API changes."""

from __future__ import annotations

import contextlib
from typing import Optional, List, Any


def _get_device_info(mx) -> dict:
    try:
        return mx.device_info()
    except Exception:
        return {}

def patch_mlx_device_info() -> None:
    """Redirect deprecated mx.metal.device_info to mx.device_info when possible."""
    try:
        import mlx.core as mx
    except Exception:
        return

    if hasattr(mx, "device_info") and hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
        try:
            mx.metal.device_info = mx.device_info  # type: ignore[attr-defined]
        except Exception:
            pass


def patch_mlx_lm_device_info() -> None:
    """Patch mlx_lm call sites to use mx.device_info() instead of mx.metal.device_info()."""
    try:
        import mlx.core as mx
        from mlx.utils import tree_reduce
    except Exception:
        return

    if not hasattr(mx, "device_info"):
        return

    patch_mlx_device_info()

    try:
        import mlx_lm.generate as mlx_generate
    except Exception:
        mlx_generate = None

    try:
        import mlx_lm.server as mlx_server
    except Exception:
        mlx_server = None

    if mlx_generate is not None and getattr(mlx_generate, "__cortex_patched__", False) is False:
        @contextlib.contextmanager
        def wired_limit(model: Any, streams: Optional[List[Any]] = None):
            if not mx.metal.is_available():
                try:
                    yield
                finally:
                    pass
                return

            model_bytes = tree_reduce(
                lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
            )
            info = _get_device_info(mx)
            max_rec_size = info.get("max_recommended_working_set_size")

            if max_rec_size and model_bytes > 0.9 * max_rec_size:
                model_mb = model_bytes // 2**20
                max_rec_mb = max_rec_size // 2**20
                print(
                    f"[WARNING] Generating with a model that requires {model_mb} MB "
                    f"which is close to the maximum recommended size of {max_rec_mb} "
                    "MB. This can be slow. See the documentation for possible work-arounds: "
                    "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
                )

            old_limit = None
            if max_rec_size:
                old_limit = mx.set_wired_limit(max_rec_size)

            try:
                yield
            finally:
                if streams is not None:
                    for s in streams:
                        mx.synchronize(s)
                else:
                    mx.synchronize()
                if old_limit is not None:
                    mx.set_wired_limit(old_limit)

        mlx_generate.wired_limit = wired_limit
        mlx_generate.__cortex_patched__ = True

    if mlx_server is not None and getattr(mlx_server, "__cortex_patched__", False) is False:
        def get_system_fingerprint():
            gpu_arch = ""
            if mx.metal.is_available():
                info = _get_device_info(mx)
                gpu_arch = info.get("architecture", "") if isinstance(info, dict) else ""
            return f"{mlx_server.__version__}-{mx.__version__}-{mlx_server.platform.platform()}-{gpu_arch}"

        mlx_server.get_system_fingerprint = get_system_fingerprint
        mlx_server.__cortex_patched__ = True
