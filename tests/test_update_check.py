"""Unit tests for cortex.update_check: version parsing, the redirect probe
(via injected openers — no sockets), the daily cache (injected clock), and
the two-repo orchestration. Everything here is hermetic by construction."""

from __future__ import annotations

import email.message
import io
import urllib.error
from pathlib import Path

import pytest

from cortex.update_check import (
    CachedComponent,
    UpdateCheckCache,
    check_for_updates,
    installed_cortex_version,
    is_newer,
    latest_release_tag,
    normalize_version,
    parse_version_token,
    probe_latest_release,
)

# ---- fake openers ---------------------------------------------------------


def _headers(location: str | None) -> email.message.Message:
    headers = email.message.Message()
    if location is not None:
        headers["Location"] = location
    return headers


class RedirectOpener:
    """urllib-with-redirects-disabled behavior: 3xx surfaces as HTTPError."""

    def __init__(self, location: str | None, *, code: int = 302) -> None:
        self.location = location
        self.code = code
        self.calls: list[str] = []

    def open(self, url: str, *, timeout: float | None = None):
        self.calls.append(url)
        raise urllib.error.HTTPError(url, self.code, "redirect", _headers(self.location), io.BytesIO(b""))


class PlainResponseOpener:
    """A 200 answer (no redirect) — not a usable release pointer."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def open(self, url: str, *, timeout: float | None = None):
        self.calls.append(url)

        class _Response:
            status = 200
            headers = _headers(None)

            def close(self) -> None:
                return None

        return _Response()


class ErrorOpener:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[str] = []

    def open(self, url: str, *, timeout: float | None = None):
        self.calls.append(url)
        raise self.error


class PerRepoOpener:
    """Routes by repo: Lumen redirects to a tag, Cortex has no releases (404)."""

    def __init__(self, *, lumen_tag: str | None = "v0.4.0", cortex_tag: str | None = None) -> None:
        self.lumen_tag = lumen_tag
        self.cortex_tag = cortex_tag
        self.calls: list[str] = []

    def open(self, url: str, *, timeout: float | None = None):
        self.calls.append(url)
        tag = self.lumen_tag if "/Lumen/" in url else self.cortex_tag
        if tag is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", _headers(None), io.BytesIO(b""))
        location = f"https://github.com/x/y/releases/tag/{tag}"
        raise urllib.error.HTTPError(url, 302, "Found", _headers(location), io.BytesIO(b""))


# ---- version parsing --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("lumen v0.3.0", (0, 3, 0)),  # installed-binary format
        ("0.3.0", (0, 3, 0)),  # bare format some builds print
        ("v1.2", (1, 2)),
        ("V2.0.1", (2, 0, 1)),
        ("cortex 1.0.18", (1, 0, 18)),
    ],
)
def test_parse_version_token_accepts_real_formats(text: str, expected: tuple) -> None:
    assert parse_version_token(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "v0.4.0-rc1",  # prerelease must never be offered as an update
        "0.4.0-beta.2",
        "lumen",  # no version at all
        "",
        "v",
        "1.2.x",
        "1..2",  # empty segment
        None,  # non-string input
    ],
)
def test_parse_version_token_rejects_prerelease_and_garbage(text) -> None:
    assert parse_version_token(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "v1.2²",  # superscript two: isdigit() True but int() RAISES
        "v١.٢.٣",  # Arabic-Indic ١.٢.٣: int() converts, but not a release tag
        "v①.0",  # circled digit one: isdigit-family trap
        "1.２.3",  # fullwidth ２
    ],
)
def test_parse_version_token_rejects_non_ascii_unicode_digits(text) -> None:
    """Adversarial Unicode tags must return None, never raise (str.isdigit is
    True for many non-ASCII digit characters; int() crashes on some of them)."""
    assert parse_version_token(text) is None
    assert is_newer(text, "0.1.0") is False


def test_normalize_version_strips_prefix_and_tool_name() -> None:
    assert normalize_version("lumen v0.4.0") == "0.4.0"
    assert normalize_version("v0.4.0") == "0.4.0"
    assert normalize_version("nonsense") is None


def test_is_newer_comparison_matrix() -> None:
    assert is_newer("v0.4.0", "0.3.0") is True
    assert is_newer("0.4.0", "v0.3.0") is True
    assert is_newer("v0.3.0", "0.3.0") is False  # equal
    assert is_newer("v0.2.0", "0.3.0") is False  # older
    assert is_newer("v0.3.10", "0.3.9") is True  # numeric, not lexicographic
    assert is_newer("v1.0", "0.9.9") is True  # differing segment counts
    assert is_newer(None, "0.3.0") is False
    assert is_newer("v0.4.0", None) is False
    assert is_newer("v0.4.0-rc1", "0.3.0") is False  # prerelease skipped


# ---- latest_release_tag ------------------------------------------------------


def test_latest_release_tag_parses_absolute_location() -> None:
    opener = RedirectOpener("https://github.com/faisalmumtaz89/Lumen/releases/tag/v0.4.0")
    assert latest_release_tag("faisalmumtaz89/Lumen", opener=opener) == "v0.4.0"
    assert opener.calls == ["https://github.com/faisalmumtaz89/Lumen/releases/latest"]


def test_latest_release_tag_parses_relative_location() -> None:
    opener = RedirectOpener("/faisalmumtaz89/Lumen/releases/tag/v0.4.0")
    assert latest_release_tag("faisalmumtaz89/Lumen", opener=opener) == "v0.4.0"


def test_latest_release_tag_unquotes_encoded_tags() -> None:
    opener = RedirectOpener("/r/x/releases/tag/v0.4.0%2Bmetal")
    assert latest_release_tag("r/x", opener=opener) == "v0.4.0+metal"


@pytest.mark.parametrize(
    "location",
    [
        "https://example.com/somewhere/else",  # no /releases/tag/ marker
        "/faisalmumtaz89/Lumen/releases/tag/",  # empty tag
        "not a url at all \x00",
        None,  # redirect without a Location header
    ],
)
def test_latest_release_tag_garbage_location_returns_none(location) -> None:
    opener = RedirectOpener(location)
    assert latest_release_tag("faisalmumtaz89/Lumen", opener=opener) is None


def test_latest_release_tag_no_redirect_returns_none() -> None:
    assert latest_release_tag("r/x", opener=PlainResponseOpener()) is None


def test_latest_release_tag_404_means_no_releases() -> None:
    opener = RedirectOpener(None, code=404)
    assert latest_release_tag("faisalmumtaz89/Cortex", opener=opener) is None


def test_latest_release_tag_is_silent_on_network_failure() -> None:
    assert latest_release_tag("r/x", opener=ErrorOpener(urllib.error.URLError("down"))) is None
    assert latest_release_tag("r/x", opener=ErrorOpener(RuntimeError("boom"))) is None


def test_latest_release_tag_honors_probe_base_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", "http://127.0.0.1:1/")
    opener = RedirectOpener("/r/x/releases/tag/v9.9.9")
    assert latest_release_tag("r/x", opener=opener) == "v9.9.9"
    assert opener.calls == ["http://127.0.0.1:1/r/x/releases/latest"]


# ---- UpdateCheckCache --------------------------------------------------------


class _Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_cache_round_trip_and_ttl(tmp_path: Path) -> None:
    clock = _Clock()
    cache = UpdateCheckCache(tmp_path / "update-check.json", clock=clock)
    assert cache.load_components() == {}  # nothing stored yet

    cache.store(lumen_latest="v0.4.0", cortex_latest=None)
    entries = cache.load_components()
    assert entries["lumen"].tag == "v0.4.0"
    assert entries["cortex"].tag is None  # definitive "no releases"
    assert cache.is_fresh(entries["lumen"]) is True

    clock.now += 24 * 60 * 60 - 1
    assert cache.is_fresh(cache.load_components()["lumen"]) is True  # inside TTL

    clock.now += 2
    stale = cache.load_components()["lumen"]
    assert cache.is_fresh(stale) is False  # expired
    assert stale.tag == "v0.4.0"  # ...but still loadable as a stale fallback


def test_cache_reads_legacy_flat_schema(tmp_path: Path) -> None:
    """The pre-per-component flat schema is read tolerantly (one shared
    checked_at) so an existing cache file survives the upgrade."""
    path = tmp_path / "update-check.json"
    path.write_text(
        '{"checked_at": 1000000, "lumen_latest": "v0.4.0", "cortex_latest": null}',
        encoding="utf-8",
    )
    cache = UpdateCheckCache(path, clock=_Clock(1000010.0))
    entries = cache.load_components()
    assert entries["lumen"] == CachedComponent(tag="v0.4.0", checked_at=1000000.0)
    assert entries["cortex"] == CachedComponent(tag=None, checked_at=1000000.0)
    assert cache.is_fresh(entries["lumen"]) is True


def test_cache_tolerates_corruption_and_clock_skew(tmp_path: Path) -> None:
    path = tmp_path / "update-check.json"
    path.write_text("{not json", encoding="utf-8")
    cache = UpdateCheckCache(path, clock=_Clock())
    assert cache.load() is None
    assert cache.load_components() == {}

    # A checked_at in the future (clock skew) must not count as fresh forever.
    clock = _Clock(now=100.0)
    cache = UpdateCheckCache(path, clock=clock)
    path.write_text('{"checked_at": 999999, "lumen_latest": "v1.0.0"}', encoding="utf-8")
    entries = cache.load_components()
    assert cache.is_fresh(entries["lumen"]) is False


# ---- check_for_updates -------------------------------------------------------


def test_check_for_updates_probes_both_repos_and_caches(tmp_path: Path) -> None:
    clock = _Clock()
    cache = UpdateCheckCache(tmp_path / "cache.json", clock=clock)
    opener = PerRepoOpener(lumen_tag="v0.4.0", cortex_tag=None)

    result = check_for_updates(cache=cache, opener=opener)
    assert result.lumen_latest == "v0.4.0"
    assert result.cortex_latest is None
    assert result.from_cache is False
    assert len(opener.calls) == 2

    # Second call inside the TTL: served from cache, zero probes.
    result = check_for_updates(cache=cache, opener=opener)
    assert result.from_cache is True
    assert result.lumen_latest == "v0.4.0"
    assert len(opener.calls) == 2

    # force=True probes regardless of freshness.
    result = check_for_updates(cache=cache, opener=opener, force=True)
    assert result.from_cache is False
    assert len(opener.calls) == 4

    # After the TTL expires, the check probes again.
    clock.now += 25 * 60 * 60
    check_for_updates(cache=cache, opener=opener)
    assert len(opener.calls) == 6


def test_check_for_updates_total_failure_is_not_cached(tmp_path: Path) -> None:
    """A transient network failure must not suppress the check for a whole
    TTL window: nothing is written, so the next call probes again."""
    cache = UpdateCheckCache(tmp_path / "cache.json", clock=_Clock())
    opener = ErrorOpener(urllib.error.URLError("offline"))

    result = check_for_updates(cache=cache, opener=opener)
    assert result.lumen_latest is None and result.cortex_latest is None
    assert cache.load() is None

    result = check_for_updates(cache=cache, opener=opener)
    assert result.from_cache is False  # probed again, not served a cached failure
    assert len(opener.calls) == 4


class ScriptedOpener:
    """Per-repo behavior keyed by URL substring: a tag string → 302 redirect,
    404 → authoritative no-releases, "error" → transient network failure."""

    def __init__(self, behaviors: dict[str, object]) -> None:
        self.behaviors = dict(behaviors)
        self.calls: list[str] = []

    def open(self, url: str, *, timeout: float | None = None):
        self.calls.append(url)
        for key, behavior in self.behaviors.items():
            if key in url:
                if behavior == "error":
                    raise urllib.error.URLError("offline")
                if behavior == 404:
                    raise urllib.error.HTTPError(
                        url, 404, "Not Found", _headers(None), io.BytesIO(b"")
                    )
                location = f"https://github.com/x/y/releases/tag/{behavior}"
                raise urllib.error.HTTPError(url, 302, "Found", _headers(location), io.BytesIO(b""))
        raise AssertionError(f"unexpected url {url}")

    def calls_for(self, key: str) -> list[str]:
        return [url for url in self.calls if key in url]


def test_probe_classification_definitive_vs_transient() -> None:
    redirect = RedirectOpener("/r/x/releases/tag/v0.4.0")
    assert probe_latest_release("r/x", opener=redirect) is not None
    assert probe_latest_release("r/x", opener=redirect).definitive is True
    assert probe_latest_release("r/x", opener=redirect).tag == "v0.4.0"

    no_releases = probe_latest_release("r/x", opener=RedirectOpener(None, code=404))
    assert no_releases.definitive is True and no_releases.tag is None

    for transient_opener in (
        ErrorOpener(urllib.error.URLError("down")),  # network failure
        RedirectOpener(None, code=500),  # server error
        RedirectOpener("https://example.com/elsewhere"),  # unparseable Location
        RedirectOpener(None, code=302),  # redirect without a Location
        PlainResponseOpener(),  # 200 where a redirect belongs (proxy)
    ):
        probe = probe_latest_release("r/x", opener=transient_opener)
        assert probe.definitive is False and probe.tag is None


def test_partial_transient_failure_keeps_cached_tag_and_reprobes(tmp_path: Path) -> None:
    """Finding-2 regression: a transient failure for ONE component must not
    clobber its previously-cached tag, and its TTL must not be extended — the
    other component's fresh answer is stored normally."""
    clock = _Clock(1_000_000.0)
    cache = UpdateCheckCache(tmp_path / "cache.json", clock=clock)
    cache.store(lumen_latest="v0.4.0", cortex_latest="v1.2.0")
    seeded_at = clock.now

    clock.now += 25 * 60 * 60  # both entries stale → both probed
    opener = ScriptedOpener({"/Lumen/": "error", "/Cortex/": "v1.3.0"})
    result = check_for_updates(cache=cache, opener=opener)

    assert result.lumen_latest == "v0.4.0"  # stale fallback survives
    assert result.cortex_latest == "v1.3.0"  # fresh answer wins
    assert result.from_cache is False

    entries = cache.load_components()
    assert entries["lumen"].tag == "v0.4.0"
    assert entries["lumen"].checked_at == seeded_at  # TTL NOT extended
    assert entries["cortex"].tag == "v1.3.0"
    assert entries["cortex"].checked_at == clock.now  # refreshed

    # Next non-forced check: only the failed component is re-probed.
    check_for_updates(cache=cache, opener=opener)
    assert len(opener.calls_for("/Lumen/")) == 2
    assert len(opener.calls_for("/Cortex/")) == 1


def test_forced_partial_failure_does_not_extend_failed_component_ttl(
    tmp_path: Path,
) -> None:
    clock = _Clock(1_000_000.0)
    cache = UpdateCheckCache(tmp_path / "cache.json", clock=clock)
    cache.store(lumen_latest="v0.4.0", cortex_latest="v1.2.0")
    seeded_at = clock.now

    clock.now += 24 * 60 * 60 - 10  # still fresh
    opener = ScriptedOpener({"/Lumen/": "error", "/Cortex/": "v1.2.0"})
    result = check_for_updates(cache=cache, opener=opener, force=True)
    assert result.lumen_latest == "v0.4.0"  # kept despite the failed force probe

    clock.now += 20  # lumen's ORIGINAL entry is now expired; cortex was refreshed
    check_for_updates(cache=cache, opener=opener)
    assert len(opener.calls_for("/Lumen/")) == 2  # re-probed (freshness anchored at t0)
    assert len(opener.calls_for("/Cortex/")) == 1  # fresh from the forced probe
    assert cache.load_components()["lumen"].checked_at == seeded_at


def test_definitive_404_overwrites_previously_cached_tag(tmp_path: Path) -> None:
    """An authoritative 404 (releases yanked) is a real answer: it replaces a
    cached tag instead of being treated as a failure to preserve."""
    clock = _Clock(1_000_000.0)
    cache = UpdateCheckCache(tmp_path / "cache.json", clock=clock)
    cache.store(lumen_latest="v0.4.0", cortex_latest=None)

    clock.now += 25 * 60 * 60
    opener = ScriptedOpener({"/Lumen/": 404, "/Cortex/": 404})
    result = check_for_updates(cache=cache, opener=opener)
    assert result.lumen_latest is None
    entries = cache.load_components()
    assert entries["lumen"].tag is None
    assert entries["lumen"].checked_at == clock.now


def test_transient_failure_with_no_prior_leaves_component_unknown(tmp_path: Path) -> None:
    clock = _Clock(1_000_000.0)
    cache = UpdateCheckCache(tmp_path / "cache.json", clock=clock)
    opener = ScriptedOpener({"/Lumen/": "v0.4.0", "/Cortex/": "error"})
    result = check_for_updates(cache=cache, opener=opener)
    assert result.lumen_latest == "v0.4.0"
    assert result.cortex_latest is None
    entries = cache.load_components()
    assert entries["lumen"].tag == "v0.4.0"
    assert "cortex" not in entries  # nothing known, nothing invented


def test_result_resolved_flags_distinguish_unknown_from_no_releases(tmp_path: Path) -> None:
    """Finding-3 plumbing: the result carries the probe layer's definitive-
    vs-transient distinction per component, so the /update surfaces can tell
    'the repo has no releases' (authoritative 404) apart from 'the latest is
    unknown' (transient failure, nothing cached)."""
    clock = _Clock(1_000_000.0)
    cache = UpdateCheckCache(tmp_path / "cache.json", clock=clock)

    # Authoritative 404 → resolved with tag None; transient error with a
    # cold cache → NOT resolved (both tags are None — the flags are the only
    # way to tell the two apart).
    opener = ScriptedOpener({"/Lumen/": 404, "/Cortex/": "error"})
    result = check_for_updates(cache=cache, opener=opener)
    assert result.lumen_latest is None and result.lumen_resolved is True
    assert result.cortex_latest is None and result.cortex_resolved is False

    # A stale-fallback tag is still a usable answer: resolved despite the
    # failed probe.
    seeded = UpdateCheckCache(tmp_path / "seeded.json", clock=clock)
    seeded.store(lumen_latest="v0.4.0", cortex_latest="v1.2.0")
    clock.now += 25 * 60 * 60  # both entries stale → both re-probed and fail
    failing = ScriptedOpener({"/Lumen/": "error", "/Cortex/": "error"})
    result = check_for_updates(cache=seeded, opener=failing)
    assert result.lumen_latest == "v0.4.0" and result.lumen_resolved is True
    assert result.cortex_latest == "v1.2.0" and result.cortex_resolved is True


def test_installed_cortex_version_reports_a_parseable_version() -> None:
    version = installed_cortex_version()
    assert version is not None
    assert parse_version_token(version) is not None
