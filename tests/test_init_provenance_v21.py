import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_init_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "create_student_init.py"
    spec = importlib.util.spec_from_file_location("create_student_init_v21", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "ci@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "CI"], check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("frozen\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    return repo, tracked


def test_init_provenance_allows_untracked_experiment_outputs(tmp_path):
    module = _load_init_script()
    repo, _ = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / "experiment_output.pt").write_bytes(b"generated")
    assert module.git_provenance(repo) == expected


def test_init_provenance_rejects_modified_tracked_source(tmp_path):
    module = _load_init_script()
    repo, tracked = _repo(tmp_path)
    tracked.write_text("mutated\n")
    with pytest.raises(RuntimeError, match="clean tracked source"):
        module.git_provenance(repo)


def test_init_provenance_rejects_staged_tracked_source(tmp_path):
    module = _load_init_script()
    repo, tracked = _repo(tmp_path)
    tracked.write_text("mutated\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    with pytest.raises(RuntimeError, match="clean tracked source"):
        module.git_provenance(repo)
