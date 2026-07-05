from pathlib import Path

from cortex.cloud.catalog import CloudModelCatalog
from cortex.cloud.types import CloudProvider


def test_default_catalog_lists_validated_latest_models(tmp_path: Path):
    # Every id below was validated with a real API call (2026-07-05).
    catalog = CloudModelCatalog(override_path=tmp_path / "missing.json")

    openai_models = catalog.list_model_ids(CloudProvider.OPENAI)
    anthropic_models = catalog.list_model_ids(CloudProvider.ANTHROPIC)

    assert openai_models == ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
    assert anthropic_models == [
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-haiku-4-5",
    ]


def test_override_catalog_merges_without_duplicates(tmp_path: Path):
    override_path = tmp_path / "cloud_models.json"
    override_path.write_text(
        """
{
  "openai": ["gpt-5.5", "gpt-custom-experimental"],
  "anthropic": ["claude-sonnet-5", "claude-labs-preview"]
}
""".strip(),
        encoding="utf-8",
    )

    catalog = CloudModelCatalog(override_path=override_path)
    openai_models = catalog.list_model_ids(CloudProvider.OPENAI)
    anthropic_models = catalog.list_model_ids(CloudProvider.ANTHROPIC)

    assert openai_models.count("gpt-5.5") == 1
    assert "gpt-custom-experimental" in openai_models
    assert anthropic_models.count("claude-sonnet-5") == 1
    assert "claude-labs-preview" in anthropic_models


def test_parse_selector_supports_provider_model_format(tmp_path: Path):
    catalog = CloudModelCatalog(override_path=tmp_path / "missing.json")

    ref = catalog.parse_selector("openai:gpt-5.1")
    assert ref is not None
    assert ref.provider == CloudProvider.OPENAI
    assert ref.model_id == "gpt-5.1"

    invalid = catalog.parse_selector("not-a-selector")
    assert invalid is None
