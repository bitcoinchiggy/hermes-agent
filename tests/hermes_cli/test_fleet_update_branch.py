"""bitcoinchiggy/hermes-agent follows git branch fleet.

Plain update and ``--check`` resolve to ``fleet`` when the main channel
record is missing and when a record is published. ``--branch fleet`` is the
one-run override. Any other branch or channel fails closed. The official
repository still follows its record, including an unpublished ``main``.

The hermes_cli conftest supplies a source-branch record for every channel
unless a test replaces ``_resolve_channel``. That stub is the present-record
case.
"""

import argparse
import os
from copy import deepcopy
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main, source_check, source_releases, update_cmd
from hermes_cli.release_channels import ChannelNotFound
from hermes_cli.source_releases import (
    FLEET_BRANCH,
    FLEET_REPOSITORY,
    OFFICIAL_REPOSITORY,
    reject_fleet_branch,
    resolve_source_target,
    source_repository,
)
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_channel import channel_record, handle_metadata_args, set_install_channel
from hermes_cli.config import require_readable_config_before_write

FLEET_URL = "https://github.com/bitcoinchiggy/hermes-agent.git"
PINNED = "a" * 40


def _git(root, *args):
    env = os.environ.copy()
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_AUTHOR_NAME"] = "Fleet Fixture"
    env["GIT_AUTHOR_EMAIL"] = "fixture@example.invalid"
    env["GIT_COMMITTER_NAME"] = "Fleet Fixture"
    env["GIT_COMMITTER_EMAIL"] = "fixture@example.invalid"
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()


def _missing(name, repository):
    raise ChannelNotFound(f"Channel object not found: releases/channels/{name}.json")


def _pinned(commit, repository):
    return SimpleNamespace(
        terminal={
            "repository": repository, "name": "main", "policy": "preview",
            "head": {"sequence": 1},
        },
        requested={"state": "active"},
        manifest={"request": {
            "commit": commit, "sourceVersion": "1.2.3", "buildId": "b" * 32,
        }},
    )


@pytest.fixture
def tree(tmp_path, monkeypatch):
    home, origin, checkout = (tmp_path / name for name in ("home", "origin", "checkout"))
    home.mkdir()
    origin.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    _git(origin, "init", "-b", "main")
    _git(origin, "config", "user.name", "Fleet Fixture")
    _git(origin, "config", "user.email", "fixture@example.invalid")
    _git(origin, "config", "commit.gpgsign", "false")
    shas = {}
    for label in ("installed", "published", "main-tip"):
        (origin / "marker.txt").write_text(label)
        _git(origin, "add", "marker.txt")
        _git(origin, "commit", "-m", label)
        shas[label] = _git(origin, "rev-parse", "HEAD")
    _git(origin, "branch", "fleet")
    _git(origin, "checkout", "fleet")
    (origin / "marker.txt").write_text("fleet-tip")
    _git(origin, "add", "marker.txt")
    _git(origin, "commit", "-m", "fleet-tip")
    shas["fleet"] = _git(origin, "rev-parse", "HEAD")
    _git(origin, "checkout", "main")
    _git(origin, "branch", "review")
    _git(tmp_path, "clone", str(origin), str(checkout))
    _git(checkout, "checkout", "--detach", shas["installed"])
    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: checkout)
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=main.cmd_update)
    opts = update_cmd._UpdateOptions(
        pre_update_version=None, gw_input_fn=None, assume_yes=True, keep_stash=False,
        switch_branch=False, discard_local_changes=False, no_gateway_restart=True,
    )
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], False))
    return SimpleNamespace(
        home=home, origin=origin, root=checkout, shas=shas, parser=parser,
    )


def _set_fleet_url(tree):
    """Store the fork URL without rewriting it to the local origin.

    ``git remote get-url`` expands ``insteadOf``. A local mirror would hide
    the repository from the passive check. Fetching tests add that rewrite
    separately, and read the stored URL the way ``source_repository`` does.
    """
    _git(tree.root, "config", "remote.origin.url", FLEET_URL)
    _git(tree.root, "config", "--local", "credential.helper", "")


def _point_fleet(tree):
    _set_fleet_url(tree)
    _git(tree.root, "config", "--local", f"url.{tree.origin}.insteadOf", FLEET_URL)


def _saved(tree):
    return channel_record(
        require_readable_config_before_write(tree.home / "config.yaml"), tree.root,
    )


def test_absent_main_record_follows_fleet(monkeypatch):
    monkeypatch.setattr(source_releases, "_resolve_channel", _missing)
    target = resolve_source_target("main", repository=FLEET_REPOSITORY)
    assert target.commit is None
    assert target.branch == FLEET_BRANCH
    assert target.repository == FLEET_REPOSITORY


def test_absent_main_record_follows_main_for_the_official_repository(monkeypatch):
    monkeypatch.setattr(source_releases, "_resolve_channel", _missing)
    target = resolve_source_target("main", repository=OFFICIAL_REPOSITORY)
    assert target.commit is None
    assert target.branch == "main"


def test_channel_lookup_failure_does_not_block_fleet(monkeypatch):
    def down(name, repository):
        raise RuntimeError("cdn down")

    monkeypatch.setattr(source_releases, "_resolve_channel", down)
    target = resolve_source_target("main", repository=FLEET_REPOSITORY)
    assert target.branch == FLEET_BRANCH and target.commit is None


def test_channel_lookup_failure_still_propagates_upstream(monkeypatch):
    def down(name, repository):
        raise RuntimeError("cdn down")

    monkeypatch.setattr(source_releases, "_resolve_channel", down)
    with pytest.raises(RuntimeError, match="cdn down"):
        resolve_source_target("main", repository=OFFICIAL_REPOSITORY)


def test_present_source_branch_record_does_not_override_fleet():
    """The conftest record delivers branch ``main``. This fork stays on fleet."""
    target = resolve_source_target("main", repository=FLEET_REPOSITORY)
    assert target.branch == FLEET_BRANCH
    assert target.commit is None


def test_present_source_branch_record_is_honored_upstream():
    target = resolve_source_target("main", repository=OFFICIAL_REPOSITORY)
    assert target.branch == "main"
    assert target.commit is None


def test_present_commit_does_not_detach_fleet(monkeypatch):
    monkeypatch.setattr(
        source_releases, "_resolve_channel",
        lambda name, repository: _pinned(PINNED, FLEET_REPOSITORY),
    )
    target = resolve_source_target("main", repository=FLEET_REPOSITORY)
    assert target.commit is None
    assert target.branch == FLEET_BRANCH


def test_present_commit_is_honored_upstream(monkeypatch):
    monkeypatch.setattr(
        source_releases, "_resolve_channel",
        lambda name, repository: _pinned(PINNED, OFFICIAL_REPOSITORY),
    )
    target = resolve_source_target("main", repository=OFFICIAL_REPOSITORY)
    assert target.commit == PINNED
    assert target.branch is None


@pytest.mark.parametrize("name", ["stable", "canary", "fleet"])
def test_other_channel_fails_before_a_record_can_apply(monkeypatch, name):
    def looked_up(channel, repository):
        raise AssertionError(f"looked up {channel}")

    monkeypatch.setattr(source_releases, "_resolve_channel", looked_up)
    with pytest.raises(ValueError, match="tracks git branch fleet") as caught:
        resolve_source_target(name, repository=FLEET_REPOSITORY)
    assert name in str(caught.value)
    assert "different commit or branch" in str(caught.value)


def test_official_named_channel_still_follows_its_record():
    target = resolve_source_target("stable", repository=OFFICIAL_REPOSITORY)
    assert target.branch == "stable"
    assert target.commit is None


def test_explicit_fleet_branch_is_allowed():
    reject_fleet_branch(FLEET_BRANCH, FLEET_REPOSITORY)


def test_explicit_other_branch_fails_on_this_fork():
    with pytest.raises(ValueError, match="would leave that track") as caught:
        reject_fleet_branch("main", FLEET_REPOSITORY)
    assert "--branch main" in str(caught.value)
    assert "--branch fleet" in str(caught.value)


def test_explicit_branch_is_unrestricted_upstream():
    reject_fleet_branch("main", OFFICIAL_REPOSITORY)
    reject_fleet_branch("review", OFFICIAL_REPOSITORY)


def test_credentialed_https_origin_selects_this_fork(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init")
    url = "https://x-access-token:not-a-real-token@github.com/bitcoinchiggy/hermes-agent.git"
    _git(checkout, "remote", "add", "origin", url)
    assert source_repository(["git"], checkout) == FLEET_REPOSITORY


def test_non_github_origin_keeps_the_official_repository(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init")
    _git(checkout, "remote", "add", "origin", str(tmp_path / "not-github.git"))
    assert source_repository(["git"], checkout) == OFFICIAL_REPOSITORY


def test_plain_check_follows_fleet_when_a_record_is_present(tree, capsys):
    _point_fleet(tree)
    before = _git(tree.root, "rev-parse", "HEAD")
    update_cmd._cmd_update_check()
    out = capsys.readouterr().out
    assert "behind origin/fleet" in out
    assert "A published channel record is not used." in out
    assert "behind origin/main" not in out
    assert _git(tree.root, "rev-parse", "HEAD") == before


def test_plain_check_follows_fleet_when_the_record_is_absent(tree, monkeypatch, capsys):
    _point_fleet(tree)
    monkeypatch.setattr(source_releases, "_resolve_channel", _missing)
    update_cmd._cmd_update_check()
    out = capsys.readouterr().out
    assert "behind origin/fleet" in out
    assert "behind origin/main" not in out


def test_plain_update_lands_on_fleet_despite_a_pinned_record(tree, monkeypatch):
    _point_fleet(tree)
    monkeypatch.setattr(
        source_releases, "_resolve_channel",
        lambda name, repository: _pinned(tree.shas["published"], repository),
    )
    completed = []
    monkeypatch.setattr(
        update_cmd, "_complete_source_update", lambda request: completed.append(deepcopy(request)),
    )
    args = tree.parser.parse_args(["update", "--yes", "--no-gateway-restart"])
    update_cmd._cmd_update_impl(args, False)
    assert _git(tree.root, "rev-parse", "HEAD") == tree.shas["fleet"]
    assert (tree.root / "marker.txt").read_text() == "fleet-tip"
    assert completed[0]["expected_sha"] == tree.shas["fleet"]
    assert completed[0]["branch"] == FLEET_BRANCH
    assert completed[0]["expected_sha"] != tree.shas["published"]


def test_plain_update_lands_on_fleet_when_the_record_is_absent(tree, monkeypatch):
    _point_fleet(tree)
    monkeypatch.setattr(source_releases, "_resolve_channel", _missing)
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda request: None)
    args = tree.parser.parse_args(["update", "--yes", "--no-gateway-restart"])
    update_cmd._cmd_update_impl(args, False)
    assert _git(tree.root, "rev-parse", "HEAD") == tree.shas["fleet"]
    assert (tree.root / "marker.txt").read_text() == "fleet-tip"


def test_explicit_branch_fleet_is_the_one_run_override(tree, capsys):
    _point_fleet(tree)
    update_cmd._cmd_update_check(branch=FLEET_BRANCH, branch_explicit=True)
    out = capsys.readouterr().out
    assert "behind origin/fleet" in out
    assert "behind origin/main" not in out


def test_explicit_other_branch_does_not_fetch(tree, capsys):
    _point_fleet(tree)
    before = _git(tree.root, "rev-parse", "HEAD")
    with pytest.raises(SystemExit) as caught:
        update_cmd._cmd_update_check(branch="main", branch_explicit=True)
    assert caught.value.code == 1
    out = capsys.readouterr().out
    assert "would leave that track" in out
    assert "Fetching" not in out
    assert "behind origin/main" not in out
    assert _git(tree.root, "rev-parse", "HEAD") == before


def test_explicit_other_branch_does_not_apply(tree, capsys):
    _point_fleet(tree)
    before = _git(tree.root, "rev-parse", "HEAD")
    args = tree.parser.parse_args(["update", "--branch", "main", "--yes"])
    with pytest.raises(SystemExit) as caught:
        update_cmd._cmd_update_impl(args, False)
    assert caught.value.code == 1
    out = capsys.readouterr().out
    assert "would leave that track" in out
    assert "No update was applied." in out
    assert _git(tree.root, "rev-parse", "HEAD") == before
    assert (tree.root / "marker.txt").read_text() == "installed"


def test_channel_override_does_not_change_tracks(tree, capsys):
    _point_fleet(tree)
    before = _git(tree.root, "rev-parse", "HEAD")
    with pytest.raises(SystemExit) as caught:
        update_cmd._cmd_update_check(channel="canary")
    assert caught.value.code == 1
    out = capsys.readouterr().out
    assert "Channel 'canary'" in out
    assert "tracks git branch fleet" in out
    assert "Fetching" not in out
    assert _git(tree.root, "rev-parse", "HEAD") == before


def test_set_channel_other_than_main_is_not_stored(tree, capsys):
    _point_fleet(tree)
    config = tree.home / "config.yaml"
    before = config.read_bytes() if config.exists() else b""
    with pytest.raises(ValueError, match="tracks git branch fleet"):
        set_install_channel("stable", tree.root)
    assert (config.read_bytes() if config.exists() else b"") == before
    args = SimpleNamespace(set_channel="canary", install_id=False)
    with pytest.raises(SystemExit) as caught:
        handle_metadata_args(args, tree.root)
    assert caught.value.code == 2
    assert "Channel 'canary'" in capsys.readouterr().out
    assert (config.read_bytes() if config.exists() else b"") == before
    key = set_install_channel("main", tree.root)
    assert _saved(tree)["channel"] == "main"
    args = SimpleNamespace(set_channel="main", install_id=False)
    assert handle_metadata_args(args, tree.root) is True
    assert f"Channel main on this installation follows git branch {FLEET_BRANCH}." in capsys.readouterr().out
    assert key


def test_stored_non_main_channel_cannot_move_the_checkout(tree, capsys):
    set_install_channel("canary", tree.root)
    assert _saved(tree)["channel"] == "canary"
    _point_fleet(tree)
    before = _git(tree.root, "rev-parse", "HEAD")
    with pytest.raises(SystemExit):
        update_cmd._cmd_update_check()
    out = capsys.readouterr().out
    assert "Channel 'canary'" in out
    assert _git(tree.root, "rev-parse", "HEAD") == before
    assert _saved(tree)["channel"] == "canary"


def test_official_check_honors_an_explicit_branch(tree, capsys):
    update_cmd._cmd_update_check(branch="review", branch_explicit=True)
    out = capsys.readouterr().out
    assert "behind origin/review" in out
    assert "would leave that track" not in out


def test_official_update_checks_out_a_published_commit(tree, monkeypatch):
    monkeypatch.setattr(
        source_releases, "_resolve_channel",
        lambda name, repository: _pinned(tree.shas["published"], OFFICIAL_REPOSITORY),
    )
    completed = []
    monkeypatch.setattr(
        update_cmd, "_complete_source_update", lambda request: completed.append(deepcopy(request)),
    )
    args = tree.parser.parse_args(["update", "--yes", "--no-gateway-restart"])
    update_cmd._cmd_update_impl(args, False)
    assert _git(tree.root, "rev-parse", "HEAD") == tree.shas["published"]
    assert (tree.root / "marker.txt").read_text() == "published"
    assert completed[0]["expected_sha"] == tree.shas["published"]


def test_passive_check_reports_fleet_from_another_branch(tree, monkeypatch):
    _set_fleet_url(tree)
    _git(tree.root, "checkout", "-B", "parked", tree.shas["installed"])
    monkeypatch.setattr(
        source_releases, "_resolve_channel",
        lambda name, repository: _pinned(tree.shas["published"], repository),
    )

    def tip(repository, branch, root, git, remote="origin"):
        assert repository == FLEET_REPOSITORY
        if branch == FLEET_BRANCH:
            return tree.shas["fleet"], False, None
        if branch == "main":
            return tree.shas["main-tip"], False, None
        raise AssertionError(branch)

    monkeypatch.setattr(source_check, "_branch_tip", tip)
    monkeypatch.setattr(source_check, "_github_compare", lambda *args, **kwargs: None)
    status = source_check.check_for_updates(
        install_root=tree.root, home=tree.home, force=True,
    )
    assert status.get("error") is None, status
    assert status["branch"] == FLEET_BRANCH, status
    assert status["targetSha"] == tree.shas["fleet"]
    assert status["targetSha"] != tree.shas["published"]
    assert status["targetSha"] != tree.shas["main-tip"]


def test_passive_check_rejects_an_explicit_other_branch(tree):
    _set_fleet_url(tree)
    status = source_check.check_for_updates(
        install_root=tree.root, home=tree.home, branch="main", force=True,
    )
    assert status.get("error") == "release-unavailable", status
    assert "would leave that track" in status["message"]
    assert "targetSha" not in status
