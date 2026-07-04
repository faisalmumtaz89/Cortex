"""Per-turn model provenance verification.

Every model client attaches a provenance record to its FinishEvent —
{client_kind, reported_model, response_id, endpoint} — taken from the actual
response payload. After each turn the orchestrator verifies that record
against the *requested* target and FAILS the turn on any mismatch, so the UI
can never display one model while another answered.

Guarantees, per backend:
- local (Lumen): the answering client must be the Lumen client, the endpoint
  must be exactly the managed lumen-server's base URL for THIS process, that
  server must still be alive and ready, and the model name reported inside the
  response stream must equal the selector's model (dot/dash separators
  normalized — lumen accepts both spellings of e.g. qwen3.5-9b).
- openai/anthropic: the answering client kind must match the provider and the
  reported model must equal the requested id, allowing only the provider's
  own date/version suffixing (prefix match in either direction).
- azure: deployments are user-named aliases of the underlying model, so name
  equality is not defined; identity is bound by client kind + the configured
  Azure endpoint instead. A missing reported model still fails.
- scripted (CORTEX_SCRIPTED_MODEL): passes verification but is labeled
  "(scripted)" and the worker banners the override at startup — it can never
  masquerade silently as a real model.

Missing provenance is always a failure: no proof, no turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from cortex.cloud.types import CloudProvider

_SEPARATORS = re.compile(r"[.\-_\s]+")

_EXPECTED_CLIENT_KIND: Dict[CloudProvider, str] = {
    CloudProvider.LUMEN: "lumen",
    CloudProvider.OPENAI: "openai",
    CloudProvider.AZURE: "openai",  # Azure is served through the OpenAI client
    CloudProvider.ANTHROPIC: "anthropic",
}


def normalize_model_name(name: str) -> str:
    """Case/separator-insensitive model-name form: qwen3.5-9b == qwen3-5-9b."""
    return _SEPARATORS.sub("-", str(name or "").strip().lower()).strip("-")


@dataclass(frozen=True)
class ProvenanceVerdict:
    ok: bool
    reason: str = ""
    scripted: bool = False


def _models_match(provider: CloudProvider, requested: str, reported: str) -> bool:
    req = normalize_model_name(requested)
    rep = normalize_model_name(reported)
    if not rep:
        return False
    if provider == CloudProvider.LUMEN:
        return rep == req
    if provider == CloudProvider.AZURE:
        # Deployment name != model name by design; transport identity (client
        # kind + endpoint) is the Azure guarantee. Reported just has to exist.
        return True
    # Providers suffix released models with dates/versions (gpt-5.1 →
    # gpt-5.1-2026-01-15, claude-sonnet-4-5 → claude-sonnet-4-5-20250929).
    return rep == req or rep.startswith(req) or req.startswith(rep)


def _endpoints_match(expected: Optional[str], reported: Optional[str]) -> bool:
    if not expected:
        return True  # no expectation registered (cloud default endpoints)
    if not reported:
        return False
    return str(expected).rstrip("/") == str(reported).rstrip("/")


def verify_turn_provenance(
    *,
    provider: CloudProvider,
    requested_model: str,
    provenance: Optional[Dict[str, Any]],
    expected_endpoint: Optional[str] = None,
    lumen_ready: Optional[bool] = None,
) -> ProvenanceVerdict:
    """Verify a finished turn's response-side identity against the request."""
    if not provenance:
        return ProvenanceVerdict(
            ok=False,
            reason="the response carried no provenance record (no proof of which model answered)",
        )

    client_kind = str(provenance.get("client_kind", "") or "")
    reported_model = str(provenance.get("reported_model", "") or "")
    reported_endpoint = provenance.get("endpoint")

    if client_kind == "scripted":
        return ProvenanceVerdict(ok=True, scripted=True)

    expected_kind = _EXPECTED_CLIENT_KIND.get(provider)
    if client_kind != expected_kind:
        return ProvenanceVerdict(
            ok=False,
            reason=(
                f"answered by a {client_kind or 'unknown'} client instead of "
                f"the {expected_kind} client"
            ),
        )

    if provider == CloudProvider.LUMEN:
        if not _endpoints_match(expected_endpoint, reported_endpoint):
            return ProvenanceVerdict(
                ok=False,
                reason=(
                    f"answered from endpoint {reported_endpoint!r} instead of the "
                    f"managed lumen-server at {expected_endpoint!r}"
                ),
            )
        if lumen_ready is not True:
            return ProvenanceVerdict(
                ok=False,
                reason="the managed lumen-server is no longer alive/ready",
            )

    if not _models_match(provider, requested_model, reported_model):
        return ProvenanceVerdict(
            ok=False,
            reason=(
                f"response reports model {reported_model or 'unknown'!r}, "
                f"requested {requested_model!r}"
            ),
        )

    return ProvenanceVerdict(ok=True)
