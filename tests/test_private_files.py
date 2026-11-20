from __future__ import annotations

import os
import select
import stat
from pathlib import Path

import pytest

from cortex.config import keep_private


@pytest.fixture()
def cortex_home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = tmp_path / "home" / ".cortex"
    folder.mkdir(parents=True)
    return folder


def _mode(path: Path) -> str:
    return oct(stat.S_IMODE(os.lstat(path).st_mode))


@pytest.mark.parametrize("where", ["inside", "elsewhere"])
def test_a_missing_file_is_created_for_its_owner_only(cortex_home, tmp_path, where):
    path = (cortex_home if where == "inside" else tmp_path) / "cortex.log"
    before = os.umask(0)
    try:
        keep_private(path)
    finally:
        os.umask(before)
    assert _mode(path) == "0o600"


@pytest.mark.parametrize(
    ("mode", "tightened"),
    [(0o644, "0o600"), (0o640, "0o600"), (0o604, "0o600"), (0o666, "0o600"), (0o444, "0o400")],
)
def test_a_file_left_readable_inside_the_cortex_folder_is_tightened(cortex_home, mode, tightened):
    """Only what others may do is taken away: a file its owner made
    read-only stays read-only."""
    path = cortex_home / "conversations.db"
    path.write_text("kept", encoding="utf-8")
    path.chmod(mode)
    keep_private(path)
    assert (_mode(path), path.read_text(encoding="utf-8")) == (tightened, "kept")


def test_a_file_shared_on_purpose_elsewhere_keeps_its_mode(cortex_home, tmp_path):
    shared = tmp_path / "team.log"
    shared.write_text("", encoding="utf-8")
    shared.chmod(0o664)
    keep_private(shared)
    assert _mode(shared) == "0o664"


def test_a_link_and_a_file_with_another_name_are_left_as_they_are(cortex_home, tmp_path):
    target = cortex_home / "target.log"
    target.write_text("", encoding="utf-8")
    target.chmod(0o644)
    (cortex_home / "linked.log").symlink_to(target)
    shared = tmp_path / "team.log"
    shared.write_text("", encoding="utf-8")
    shared.chmod(0o664)
    os.link(shared, cortex_home / "cortex.log")
    dangling = cortex_home / "dangling.log"
    dangling.symlink_to(tmp_path / "nowhere.log")

    for path in (cortex_home / "linked.log", cortex_home / "cortex.log", dangling):
        keep_private(path)

    assert (_mode(target), _mode(shared)) == ("0o644", "0o664")
    assert not (tmp_path / "nowhere.log").exists()


def test_a_pipe_is_not_opened(cortex_home):
    """A reader of the pipe takes a writer that comes and goes for the end
    of the log, and stops before the log is written."""
    pipe = cortex_home / "pipe.log"
    os.mkfifo(pipe)
    pipe.chmod(0o644)
    reader = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
    try:
        keep_private(pipe)
        watching = select.poll()
        watching.register(reader, select.POLLHUP)
        assert watching.poll(0) == []  # no writer has come and gone
    finally:
        os.close(reader)
    assert _mode(pipe) == "0o644"


@pytest.mark.skipif(os.getuid() != 0, reason="needs to make a device file")
def test_a_device_is_left_as_it_is(cortex_home):
    device = cortex_home / "null"
    os.mknod(device, 0o666 | stat.S_IFCHR, os.makedev(1, 3))
    device.chmod(0o666)
    keep_private(device)
    assert _mode(device) == "0o666"


def test_inside_means_where_the_path_leads(cortex_home, tmp_path, monkeypatch):
    outside = tmp_path / "out.log"
    outside.write_text("", encoding="utf-8")
    outside.chmod(0o664)
    keep_private(cortex_home / ".." / ".." / "out.log")
    assert _mode(outside) == "0o664"

    real = tmp_path / "real-cortex"
    real.mkdir()
    (tmp_path / "linked-home").mkdir()
    (tmp_path / "linked-home" / ".cortex").symlink_to(real)
    monkeypatch.setenv("HOME", str(tmp_path / "linked-home"))
    log = real / "cortex.log"
    log.write_text("", encoding="utf-8")
    log.chmod(0o620)
    keep_private(tmp_path / "linked-home" / ".cortex" / "cortex.log")
    assert _mode(log) == "0o600"


@pytest.mark.skipif(os.getuid() != 0, reason="needs to give a file to another user")
def test_another_users_file_is_not_changed(cortex_home):
    path = cortex_home / "cortex.log"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    os.chown(path, 65534, -1)
    keep_private(path)
    assert _mode(path) == "0o644"


def test_a_path_that_cannot_be_opened_is_left_to_its_user(cortex_home):
    keep_private(cortex_home / "missing-folder" / "cortex.log")
    keep_private(cortex_home)  # a folder
