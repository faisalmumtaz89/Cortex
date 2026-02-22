import os
from pathlib import Path

import pytest

import cortex.model_downloader as model_downloader_mod
from cortex.model_downloader import HF_HUB_AVAILABLE, ModelDownloader


def test_inspect_download_target_uses_namespaced_repo_path(tmp_path: Path) -> None:
    downloader = ModelDownloader(tmp_path)
    target = downloader.inspect_download_target("hf-internal-testing/tiny-random-bert")

    assert target["kind"] == "repo"
    assert target["path"] == tmp_path / "hf-internal-testing--tiny-random-bert"
    assert target["exists"] is False
    assert target["resumable"] is False


def test_inspect_download_target_prefers_legacy_repo_path_when_present(tmp_path: Path) -> None:
    legacy_path = tmp_path / "tiny-random-bert"
    legacy_path.mkdir(parents=True)
    (legacy_path / "weights.npz").write_bytes(b"ok")

    downloader = ModelDownloader(tmp_path)
    target = downloader.inspect_download_target("hf-internal-testing/tiny-random-bert")

    assert target["path"] == legacy_path
    assert target["exists"] is True
    assert target["resumable"] is False


def test_inspect_file_target_uses_namespaced_repo_folder(tmp_path: Path) -> None:
    downloader = ModelDownloader(tmp_path)
    target = downloader.inspect_download_target(
        "someone/repo",
        "quantized/model.Q4_K_M.gguf",
    )

    assert target["kind"] == "file"
    assert target["path"] == tmp_path / "someone--repo" / "quantized" / "model.Q4_K_M.gguf"
    assert target["exists"] is False


def test_inspect_file_target_rejects_path_traversal(tmp_path: Path) -> None:
    downloader = ModelDownloader(tmp_path)

    with pytest.raises(ValueError):
        downloader.inspect_download_target("someone/repo", "../escape.gguf")


@pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub unavailable")
def test_tqdm_progress_tracks_disabled_bytes_bar(tmp_path: Path) -> None:
    downloader = ModelDownloader(tmp_path)
    payloads: list[dict[str, object]] = []

    tqdm_class = downloader._build_tqdm_class(
        repo_id="someone/repo",
        progress_callback=payloads.append,
        cancel_requested=None,
    )

    assert tqdm_class is not None

    progress_bar = tqdm_class(
        total=1_024,
        initial=0,
        desc="Downloading (incomplete total...)",
        unit="B",
        disable=True,
    )
    progress_bar.update(128)
    progress_bar.update(256)
    progress_bar.total += 512
    progress_bar.refresh()
    progress_bar.close()

    assert payloads, "Expected progress callbacks from disabled tqdm bar."
    assert any(float(payload.get("units_downloaded", 0.0) or 0.0) >= 384.0 for payload in payloads)
    assert any(str(payload.get("description", "")) == "Downloading (incomplete total...)" for payload in payloads)
    assert any(str(payload.get("unit", "")) == "B" for payload in payloads)
    assert any(float(payload.get("units_total", 0.0) or 0.0) >= 1_536.0 for payload in payloads)
    assert any(bool(payload.get("done")) for payload in payloads)


@pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub unavailable")
def test_progress_runtime_tuning_sets_and_restores_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = ModelDownloader(tmp_path)

    monkeypatch.setenv("CORTEX_HF_DOWNLOAD_CHUNK_BYTES", "262144")
    monkeypatch.setenv("CORTEX_HF_DISABLE_XET", "1")
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    assert model_downloader_mod.hf_constants is not None
    original_chunk_size = model_downloader_mod.hf_constants.DOWNLOAD_CHUNK_SIZE
    original_disable_xet_constant = model_downloader_mod.hf_constants.HF_HUB_DISABLE_XET

    with downloader._progress_download_runtime_tuning(enabled=True):
        assert model_downloader_mod.hf_constants.DOWNLOAD_CHUNK_SIZE == 262_144
        assert os.getenv("HF_HUB_DISABLE_XET") == "1"
        assert model_downloader_mod.hf_constants.HF_HUB_DISABLE_XET is True

    assert model_downloader_mod.hf_constants.DOWNLOAD_CHUNK_SIZE == original_chunk_size
    assert model_downloader_mod.hf_constants.HF_HUB_DISABLE_XET == original_disable_xet_constant
    assert os.getenv("HF_HUB_DISABLE_XET") is None
