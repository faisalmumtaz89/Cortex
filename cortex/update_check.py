"""Release discovery for Cortex and the managed Lumen engine.

Both projects publish GitHub releases, and GitHub answers
``/{repo}/releases/latest`` with a redirect to ``/releases/tag/<tag>`` — a
rate-limit-free, unauthenticated way to learn the latest version without the
API. This module probes that redirect (redirects DISABLED — the Location
header is the answer), parses ``--version``-style output, and caches results
for a day so the startup check costs at most one round-trip per component per
day.

Every network-facing function fails SILENTLY (returns None): update discovery
is a courtesy, never a startup hazard.

Test seams (all injectable, no monkeypatching required):
  - ``opener``: any object with ``open(url, timeout=...)`` (urllib-compatible).
  - ``clock``: epoch-seconds callable for the cache TTL.
  - ``CORTEX_UPDATE_PROBE_BASE`` (env, TEST-ONLY): overrides the
    ``https://github.com`` base so suites can point the probe at a local stub
    HTTP server instead of the real network.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

LUMEN_REPO = "faisalmumtaz89/Lumen"
CORTEX_REPO = "faisalmumtaz89/Cortex"

DEFAULT_PROBE_BASE = "https://github.com"
PROBE_BASE_ENV = "CORTEX_UPDATE_PROBE_BASE"  # test-only override, see module docstring

CACHE_TTL_SECONDS = 24 * 60 * 60  # daily check

_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


class UrlOpener(Protocol):
    """The slice of urllib's OpenerDirector that the probe needs."""

    def open(self, url: str, /, *, timeout: float | None = None) -> Any: ...


# ---- version parsing ----------------------------------------------------


def parse_version_token(text: object) -> Optional[Tuple[int, ...]]:
    """Parse a version from CLI output like ``lumen v0.3.0`` or bare ``0.3.0``.

    The version is the LAST whitespace token (tools prepend their name), an
    optional leading ``v`` is stripped, and the remainder must be dot-separated
    ASCII-decimal integers. Prereleases (``0.4.0-rc1``) and anything
    unparseable → None — update logic must never act on a version it cannot
    order. Strictly ASCII: str.isdigit() alone also accepts Unicode digit
    characters (superscripts crash int(); Arabic-Indic numerals convert but
    are not release-tag material), so segments are validated as ASCII 0-9.
    """
    if not isinstance(text, str):
        return None
    tokens = text.split()
    if not tokens:
        return None
    candidate = tokens[-1]
    if candidate.startswith(("v", "V")):
        candidate = candidate[1:]
    if not candidate:
        return None
    parts = candidate.split(".")
    numbers: list[int] = []
    for part in parts:
        # ASCII-decimal only: isdigit() alone passes "²" (int() raises) and
        # "٣" (converts, but is not a comparable release segment).
        if not part or not part.isascii() or not part.isdigit():
            return None  # prerelease suffix, build metadata, or garbage
        numbers.append(int(part))
    return tuple(numbers)


def normalize_version(text: object) -> Optional[str]:
    """Canonical dotted form of a parseable version token, else None."""
    parsed = parse_version_token(text)
    if parsed is None:
        return None
    return ".".join(str(part) for part in parsed)


def is_newer(candidate: object, installed: object) -> bool:
    """True only when both versions parse AND candidate > installed."""
    candidate_parsed = parse_version_token(candidate)
    installed_parsed = parse_version_token(installed)
    if candidate_parsed is None or installed_parsed is None:
        return False
    return candidate_parsed > installed_parsed


# ---- latest-release probe ------------------------------------------------


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects: urllib then raises HTTPError with the 3xx
    status and headers, which is exactly the answer we want to inspect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def probe_base_url() -> str:
    override = os.environ.get(PROBE_BASE_ENV, "").strip()
    return (override or DEFAULT_PROBE_BASE).rstrip("/")


def _tag_from_location(location: str) -> Optional[str]:
    """Extract ``<tag>`` from a ``…/releases/tag/<tag>`` Location header.

    Accepts absolute and relative URLs; anything without the marker path (or
    with an empty/nested tag segment) is garbage → None.
    """
    try:
        path = urllib.parse.urlsplit(location.strip()).path
    except ValueError:
        return None
    marker = "/releases/tag/"
    index = path.find(marker)
    if index < 0:
        return None
    remainder = path[index + len(marker):]
    if not remainder:
        return None
    tag = urllib.parse.unquote(remainder.split("/", 1)[0]).strip()
    return tag or None


@dataclass(frozen=True)
class ReleaseProbe:
    """One probe answer.

    ``definitive`` separates AUTHORITATIVE answers from transient trouble:
      - a parsed redirect tag → definitive, tag set
      - HTTP 404 → definitive, tag None (the repo truly has no releases)
      - network errors, unexpected statuses, unparseable Locations → NOT
        definitive (tag None) — callers must not overwrite cached knowledge
        with these.
    """

    tag: Optional[str]
    definitive: bool


_TRANSIENT = ReleaseProbe(tag=None, definitive=False)


def probe_latest_release(
    repo: str,
    *,
    opener: UrlOpener | None = None,
    timeout: float = 3.0,
) -> ReleaseProbe:
    """Probe ``{base}/{repo}/releases/latest`` with redirects disabled and
    classify the answer (see ReleaseProbe). Never raises."""
    try:
        url = f"{probe_base_url()}/{repo}/releases/latest"
        active_opener: UrlOpener = opener or urllib.request.build_opener(_NoRedirectHandler)
        location: Optional[str] = None
        try:
            response = active_opener.open(url, timeout=timeout)
            try:
                status = getattr(response, "status", None)
                if status is None:
                    status = getattr(response, "code", None)
                if status in _REDIRECT_CODES:
                    location = response.headers.get("Location")
                else:
                    # GitHub always redirects this path; anything else is an
                    # interposed answer (proxy/captive portal) — transient.
                    return _TRANSIENT
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except urllib.error.HTTPError as error:
            # The no-redirect handler surfaces 3xx as HTTPError — the redirect
            # IS the success path here.
            if error.code in _REDIRECT_CODES:
                location = error.headers.get("Location")
            elif error.code == 404:
                return ReleaseProbe(tag=None, definitive=True)  # no releases
            else:
                return _TRANSIENT
        if not location:
            return _TRANSIENT
        tag = _tag_from_location(str(location))
        if tag is None:
            return _TRANSIENT  # answered, but unusably — never act on it
        return ReleaseProbe(tag=tag, definitive=True)
    except Exception:
        logger.debug("probe_latest_release(%s) failed", repo, exc_info=True)
        return _TRANSIENT


def latest_release_tag(
    repo: str,
    *,
    opener: UrlOpener | None = None,
    timeout: float = 3.0,
) -> Optional[str]:
    """Latest release tag of ``repo`` (e.g. ``v0.4.0``), or None on ANY failure
    (no releases, network trouble, malformed answers). Never raises."""
    return probe_latest_release(repo, opener=opener, timeout=timeout).tag


# ---- daily cache -----------------------------------------------------------


@dataclass(frozen=True)
class CachedComponent:
    """One component's last DEFINITIVE probe answer and when it was obtained."""

    tag: Optional[str]  # None = the repo definitively had no releases
    checked_at: float


class UpdateCheckCache:
    """Per-component JSON cache at ``~/.cortex/update-check.json``.

    Schema: ``{"components": {"<name>": {"latest": <tag|null>, "checked_at": <epoch>}}}``.

    Each component carries its OWN ``checked_at`` — the load-bearing rule is
    that only a DEFINITIVE probe answer (a release tag, or GitHub's
    authoritative 404 = no releases) may refresh a component. A transient
    failure never overwrites a cached tag and never extends its freshness:
    the stale tag stays available as a fallback and the component is re-probed
    on the next check. The earlier flat schema ({checked_at, lumen_latest,
    cortex_latest}) is read tolerantly and upgraded on the next write.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        clock=time.time,
        ttl_seconds: float = CACHE_TTL_SECONDS,
    ) -> None:
        self.path = path or (Path.home() / ".cortex" / "update-check.json")
        self.clock = clock
        self.ttl_seconds = ttl_seconds

    def load(self) -> Optional[dict]:
        """The raw JSON payload (any schema), or None when absent/corrupt."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _entry_from(raw: object) -> Optional[CachedComponent]:
        if not isinstance(raw, dict):
            return None
        checked_at = raw.get("checked_at")
        if not isinstance(checked_at, (int, float)):
            return None
        tag = raw.get("latest")
        return CachedComponent(
            tag=tag if isinstance(tag, str) else None, checked_at=float(checked_at)
        )

    def load_components(self) -> dict[str, CachedComponent]:
        payload = self.load()
        if payload is None:
            return {}
        components = payload.get("components")
        if isinstance(components, dict):
            entries: dict[str, CachedComponent] = {}
            for name, raw in components.items():
                entry = self._entry_from(raw)
                if isinstance(name, str) and entry is not None:
                    entries[name] = entry
            return entries
        # Legacy flat schema: one shared checked_at.
        checked_at = payload.get("checked_at")
        if not isinstance(checked_at, (int, float)):
            return {}
        entries = {}
        for name, key in (("lumen", "lumen_latest"), ("cortex", "cortex_latest")):
            tag = payload.get(key)
            entries[name] = CachedComponent(
                tag=tag if isinstance(tag, str) else None, checked_at=float(checked_at)
            )
        return entries

    def is_fresh(self, entry: CachedComponent) -> bool:
        age = float(self.clock()) - entry.checked_at
        return 0 <= age < self.ttl_seconds

    def store_components(self, entries: dict[str, CachedComponent]) -> None:
        payload = {
            "components": {
                name: {"latest": entry.tag, "checked_at": entry.checked_at}
                for name, entry in entries.items()
            }
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            logger.debug("failed to write update-check cache at %s", self.path, exc_info=True)

    def store(self, *, lumen_latest: Optional[str], cortex_latest: Optional[str]) -> None:
        """Convenience writer: both components definitively answered NOW."""
        now = float(self.clock())
        self.store_components(
            {
                "lumen": CachedComponent(tag=lumen_latest, checked_at=now),
                "cortex": CachedComponent(tag=cortex_latest, checked_at=now),
            }
        )


# ---- orchestration ------------------------------------------------------


@dataclass(frozen=True)
class UpdateCheckResult:
    """Latest known release tags (``v``-prefixed as published) per component.

    ``lumen_resolved``/``cortex_resolved`` carry the probe layer's
    definitive-vs-transient distinction (see ReleaseProbe): a None tag with
    ``resolved=True`` means the repo AUTHORITATIVELY has no releases (GitHub
    404, possibly remembered from the cache), while ``resolved=False`` means
    every probe failed transiently and nothing was cached — the latest is
    simply UNKNOWN. User-facing surfaces must never present the unknown case
    as the factual claim "no releases exist"."""

    lumen_latest: Optional[str]
    cortex_latest: Optional[str]
    from_cache: bool
    lumen_resolved: bool
    cortex_resolved: bool


def check_for_updates(
    *,
    cache: UpdateCheckCache | None = None,
    opener: UrlOpener | None = None,
    timeout: float = 3.0,
    force: bool = False,
    lumen_repo: str = LUMEN_REPO,
    cortex_repo: str = CORTEX_REPO,
) -> UpdateCheckResult:
    """Latest release tags for Lumen and Cortex — at most one probe per
    component per TTL window, with per-component merge semantics:

      - A component with a FRESH cache entry is not probed (unless ``force``,
        for the live /update status path).
      - A DEFINITIVE probe answer (tag, or 404 = no releases) replaces that
        component's entry and refreshes its ``checked_at``.
      - A TRANSIENT failure keeps the component's previous cached tag (even a
        stale one — it is still the best known answer) and does NOT extend its
        freshness, so the component is re-probed on the next check.
      - Nothing is written unless at least one definitive answer arrived, so
        a total outage never suppresses checking for a TTL window.
    """
    active_cache = cache or UpdateCheckCache()
    previous = active_cache.load_components()
    repos = {"lumen": lumen_repo, "cortex": cortex_repo}

    resolved: dict[str, CachedComponent] = {}
    probed_any = False
    definitive_any = False
    for name, repo in repos.items():
        prior = previous.get(name)
        if not force and prior is not None and active_cache.is_fresh(prior):
            resolved[name] = prior
            continue
        probed_any = True
        probe = probe_latest_release(repo, opener=opener, timeout=timeout)
        if probe.definitive:
            definitive_any = True
            resolved[name] = CachedComponent(tag=probe.tag, checked_at=float(active_cache.clock()))
        elif prior is not None:
            resolved[name] = prior  # stale fallback; freshness NOT extended

    if definitive_any:
        active_cache.store_components(resolved)

    def _tag(name: str) -> Optional[str]:
        entry = resolved.get(name)
        return entry.tag if entry is not None else None

    return UpdateCheckResult(
        lumen_latest=_tag("lumen"),
        cortex_latest=_tag("cortex"),
        from_cache=not probed_any,
        # An entry in ``resolved`` is an authoritative answer: a fresh or
        # stale-fallback tag, or a definitive "no releases" (tag None). A
        # component with NO entry failed transiently with nothing cached.
        lumen_resolved="lumen" in resolved,
        cortex_resolved="cortex" in resolved,
    )


def installed_cortex_version() -> Optional[str]:
    """The running Cortex version (installed distribution, else the source
    tree's ``__version__``) — the same source ``cortex --version`` reports."""
    try:
        from importlib.metadata import version

        return version("cortex-llm")
    except Exception:
        pass
    try:
        import cortex

        return getattr(cortex, "__version__", None)
    except Exception:
        return None
