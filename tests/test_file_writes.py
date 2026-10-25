import errno
import os
import shutil
import subprocess

import pytest

from cortex.tooling.builtin.common import write_text_atomic


def test_a_write_that_fails_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    target = tmp_path / "app.py"
    target.write_text("original\n")

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        write_text_atomic(target, "new content\n")

    assert target.read_text() == "original\n"
    assert [p.name for p in tmp_path.iterdir()] == ["app.py"]


def test_a_new_file_gets_the_default_mode(tmp_path):
    umask = os.umask(0o022)
    os.umask(umask)
    target = tmp_path / "new.txt"

    write_text_atomic(target, "x")

    assert target.stat().st_mode & 0o777 == 0o666 & ~umask


def test_a_file_that_cannot_be_replaced_is_written_in_place(tmp_path, monkeypatch):
    target = tmp_path / "mounted.txt"
    target.write_text("original\n")

    def busy(*_args):
        raise OSError(errno.EBUSY, "Device or resource busy")

    monkeypatch.setattr(os, "replace", busy)
    write_text_atomic(target, "new\n")

    assert target.read_text() == "new\n"
    assert [p.name for p in tmp_path.iterdir()] == ["mounted.txt"]


def test_the_new_content_is_private_until_it_replaces_the_file(tmp_path, monkeypatch):
    target = tmp_path / "secret.env"
    target.write_text("old\n")
    target.chmod(0o600)
    seen = []
    real_replace = os.replace

    def spy(source, destination):
        seen.append(os.stat(source).st_mode & 0o777)
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", spy)
    write_text_atomic(target, "new\n")

    assert seen == [0o600]


def test_a_long_file_name_is_written(tmp_path):
    target = tmp_path / ("n" * 250)

    write_text_atomic(target, "x")

    assert target.read_text() == "x"


@pytest.mark.skipif(not hasattr(os, "setxattr"), reason="extended attributes are Linux-only here")
def test_extended_attributes_and_acls_are_kept(tmp_path):
    target = tmp_path / "shared.txt"
    target.write_text("old\n")
    try:
        os.setxattr(target, "user.tag", b"blue")
    except OSError:
        pytest.skip("the filesystem has no user attributes")
    acl = shutil.which("setfacl")
    if acl:
        subprocess.run([acl, "-m", "u:1000:rw", str(target)], check=True)
    before = {name: os.getxattr(target, name) for name in os.listxattr(target)}

    write_text_atomic(target, "new\n")

    assert target.read_text() == "new\n"
    assert {name: os.getxattr(target, name) for name in os.listxattr(target)} == before


@pytest.mark.skipif(not shutil.which("setfacl"), reason="needs setfacl")
def test_a_directory_default_acl_does_not_reach_an_edited_file(tmp_path):
    target = tmp_path / "private.txt"
    target.write_text("old\n")
    target.chmod(0o640)
    subprocess.run(["setfacl", "-d", "-m", "u:1000:rw", str(tmp_path)], check=True)

    write_text_atomic(target, "new\n")

    assert "system.posix_acl_access" not in os.listxattr(target)
    assert target.stat().st_mode & 0o777 == 0o640


@pytest.mark.skipif(not shutil.which("setfacl"), reason="needs setfacl")
def test_a_new_file_gets_the_permissions_any_new_file_would(tmp_path):
    subprocess.run(["setfacl", "-d", "-m", "u:1000:rw,g::rwx", str(tmp_path)], check=True)
    plain = tmp_path / "plain.txt"
    plain.write_text("x")

    write_text_atomic(tmp_path / "written.txt", "x")

    def acl(path):
        return subprocess.run(
            ["getfacl", "-c", "-p", str(path)], capture_output=True, text=True, check=True
        ).stdout

    assert acl(tmp_path / "written.txt") == acl(plain)
