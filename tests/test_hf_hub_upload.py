from unittest.mock import MagicMock

from kvtrace.hf_hub.download import download_dataset_file
from kvtrace.hf_hub.upload import resolve_repo_id, upload_dataset_file


def test_resolve_repo_id_from_env(monkeypatch):
    monkeypatch.setenv("HF_REPO_ID", "me/my-ds")
    assert resolve_repo_id() == "me/my-ds"


def test_resolve_repo_id_from_user(monkeypatch):
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.setenv("HF_USER", "me")
    assert resolve_repo_id() == "me/kv-trace-study"


def test_resolve_repo_id_none_when_unset(monkeypatch):
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("HF_USER", raising=False)
    assert resolve_repo_id() is None


def test_upload_skipped_when_no_repo(tmp_path, monkeypatch):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("HF_USER", raising=False)
    result = upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    assert result is False
    api.upload_file.assert_not_called()


def test_upload_calls_api(tmp_path, monkeypatch):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    # list_repo_refs returns no branches matching → upload proceeds.
    api.list_repo_refs.return_value = MagicMock(branches=[])
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    result = upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    assert result is True
    api.create_repo.assert_called_once()
    api.create_branch.assert_called_once()
    api.upload_file.assert_called_once()


def test_upload_creates_dataset_repo_with_exist_ok(tmp_path, monkeypatch):
    """First-run case: dataset repo doesn't exist on Hub yet."""
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    api.list_repo_refs.return_value = MagicMock(branches=[])
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    _, kwargs = api.create_repo.call_args
    assert kwargs["repo_id"] == "me/kv"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["exist_ok"] is True


def test_upload_skips_when_create_repo_fails(tmp_path, monkeypatch):
    """If repo can't be created (e.g. auth), don't crash — just skip."""
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    api.create_repo.side_effect = RuntimeError("403 forbidden")
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    result = upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    assert result is False
    api.create_branch.assert_not_called()
    api.upload_file.assert_not_called()


def test_upload_swallows_upload_errors(tmp_path, monkeypatch):
    """A flaky 5xx on upload_file must not propagate and kill Phase 1."""
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    api.list_repo_refs.return_value = MagicMock(branches=[])
    api.upload_file.side_effect = RuntimeError("503 service unavailable")
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    result = upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    assert result is False  # logged-and-swallowed, no exception raised


def test_upload_idempotent_when_revision_exists(tmp_path, monkeypatch):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    # MagicMock(name=...) sets the *mock's* repr name, not the .name attribute,
    # so we have to assign .name explicitly for the branch lookup to match.
    branch = MagicMock()
    branch.name = "traces-m-bf16"
    api.list_repo_refs.return_value = MagicMock(branches=[branch])
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    result = upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    # idempotent short-circuit
    assert result is False
    api.upload_file.assert_not_called()


def test_download_calls_hf_hub_download(tmp_path, monkeypatch):
    api = MagicMock()
    api.hf_hub_download.return_value = str(tmp_path / "downloaded.jsonl")
    (tmp_path / "downloaded.jsonl").write_text("ok")
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    local = download_dataset_file("data.jsonl", revision_tag="traces-m-bf16", api=api)
    assert local is not None
    api.hf_hub_download.assert_called_once()
