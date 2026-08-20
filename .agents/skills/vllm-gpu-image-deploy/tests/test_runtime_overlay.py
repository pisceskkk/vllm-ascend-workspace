from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tarfile


SKILL = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"


def run(*command: str, cwd: pathlib.Path) -> str:
    process = subprocess.run(
        list(command), cwd=cwd, check=True, stdout=subprocess.PIPE, text=True
    )
    return process.stdout


def test_prepare_source_records_checksums_and_image_relative_deletions(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "vllm"
    package = repo / "vllm"
    package.mkdir(parents=True)
    run("git", "init", "-q", cwd=repo)
    run("git", "config", "user.email", "test@example.invalid", cwd=repo)
    run("git", "config", "user.name", "test", cwd=repo)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "generated.py").write_text("IMAGE_ONLY = True\n", encoding="utf-8")
    run("git", "add", "vllm", cwd=repo)
    run("git", "commit", "-qm", "image base", cwd=repo)
    image_commit = run("git", "rev-parse", "HEAD", cwd=repo).strip()

    (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    (package / "generated.py").unlink()
    (package / "runtime.json").write_text('{"candidate": true}\n', encoding="utf-8")
    (package / "local.so").write_bytes(b"must not be archived")
    output = tmp_path / "source.tar.gz"
    payload = json.loads(
        run(
            sys.executable,
            str(SCRIPTS / "prepare_source.py"),
            "--vllm-repo",
            str(repo),
            "--image-build-commit",
            image_commit,
            "--output",
            str(output),
            cwd=repo,
        )
    )

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        python_manifest = archive.extractfile(".vaws-python-sha256sums").read()
        deleted = archive.extractfile(".vaws-deleted-paths").read().decode().splitlines()
        metadata = json.load(archive.extractfile(".vaws-source-metadata.json"))

    assert "vllm/__init__.py" in names
    assert "vllm/runtime.json" in names
    assert "vllm/local.so" not in names
    assert deleted == ["vllm/generated.py"]
    assert metadata["image_build_commit"] == image_commit
    assert payload["python_sha256"] == hashlib.sha256(python_manifest).hexdigest()


def test_deploy_script_uses_no_clobber_runtime_seed_and_source_checksums() -> None:
    spec = importlib.util.spec_from_file_location(
        "gpu_deploy_fleet", SCRIPTS / "deploy_fleet.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        remote_image_tar="/tmp/image.tar",
        image_tar_sha256="1" * 64,
        image_id="sha256:" + "2" * 64,
        image_ref="vllm:test",
        rootfs_sha256="3" * 64,
        image_build_commit="4" * 40,
        remote_source_tar="/tmp/source.tar.gz",
        source_sha256="5" * 64,
        python_sha256="6" * 64,
        source_head="7" * 40,
        container="vllm-managed",
        workspace_root="/tmp/workspaces",
        model_root="/home/weight",
        expected_gpus=8,
        expected_gpu_model="NVIDIA H20",
    )

    script = module.render_remote_script(args)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert 'cp -a -n "$vaws_image_pkg"/. "$vaws_workspace_pkg"/' in script
    assert "sha256sum -c .vaws-source-sha256sums" in script
    assert "done < /workspace/vllm/.vaws-deleted-paths" in script
    assert 'vaws_workspace_stage="${vaws_workspace}.part-$$"' in script
    assert ".incomplete-$(date -u +%Y%m%dT%H%M%SZ)" in script
    assert 'mv "$vaws_image_pkg" "${vaws_image_pkg}.image"' in script
    assert 'ln -s "$vaws_workspace_pkg" "$vaws_image_pkg"' in script
    assert '[ "$vaws_current_workspace" = "$vaws_workspace" ]' in script
    assert "fp8_fp4_paged_mqa_logits" in script
    assert "vaws.skill=vllm-gpu-image-deploy" in script
    assert 'test "$vaws_gpu_count" -gt 0' in script
    assert 'test "$vaws_gpu_models" = "$vaws_expected_gpu_model"' in script
    assert "gpu_models=%s" in script
    assert "vaws.skill=h20-" not in script


def test_generic_gpu_validation_has_no_h20_or_eight_gpu_default() -> None:
    spec = importlib.util.spec_from_file_location(
        "gpu_deploy_generic", SCRIPTS / "deploy_fleet.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        remote_image_tar="/tmp/image.tar",
        image_tar_sha256="1" * 64,
        image_id="sha256:" + "2" * 64,
        image_ref="vllm:test",
        rootfs_sha256="3" * 64,
        image_build_commit="4" * 40,
        remote_source_tar="/tmp/source.tar.gz",
        source_sha256="5" * 64,
        python_sha256="6" * 64,
        source_head="7" * 40,
        container="vllm-managed",
        workspace_root="/tmp/workspaces",
        model_root="/home/weight",
        expected_gpus=None,
        expected_gpu_model=None,
    )

    script = module.render_remote_script(args)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "vaws_expected_gpus=''" in script
    assert "vaws_expected_gpu_model=''" in script
    assert "NVIDIA H20" not in script
    assert 'test "$vaws_gpu_count" -gt 0' in script


def test_deploy_ssh_config_override_is_used() -> None:
    spec = importlib.util.spec_from_file_location(
        "gpu_deploy_ssh", SCRIPTS / "deploy_fleet.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ssh_command_prefix = lambda: ["ssh"]
    args = argparse.Namespace(
        identity_file=None,
        ssh_config=pathlib.Path("/dev/null"),
        port=22,
        user="root",
    )
    command = module.ssh_base(args, "192.0.2.1")
    assert command[command.index("-F") + 1] == "/dev/null"


def test_gpu_workspace_entrypoint_is_repository_level_gpu_tool() -> None:
    spec = importlib.util.spec_from_file_location(
        "gpu_deploy_workspace_entrypoint", SCRIPTS / "deploy_fleet.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.GPU_WORKSPACE_TOOL.name == "gpu_workspace.py"
    assert module.GPU_WORKSPACE_TOOL.parent.name == "scripts"
    assert module.GPU_WORKSPACE_TOOL.parent.parent.name == ".agents"
    args = argparse.Namespace(
        user="gpu-user",
        port=2202,
        container="managed-vllm-gpu",
        identity_file=None,
        ssh_config=pathlib.Path("/dev/null"),
    )
    command = module.gpu_workspace_setup_command(args, "192.0.2.8")
    assert command[1].endswith("/.agents/scripts/gpu_workspace.py")
    assert command[command.index("--host") + 1] == "192.0.2.8"
    assert command[command.index("--container") + 1] == "managed-vllm-gpu"
    assert command[command.index("--ssh-config") + 1] == "/dev/null"


def test_fleet_stream_separates_binary_transfer_from_finalize_script() -> None:
    spec = importlib.util.spec_from_file_location(
        "gpu_fleet_stream_quoting", SCRIPTS / "fleet_stream.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ssh_command_prefix = lambda: ["ssh"]
    args = argparse.Namespace(
        identity_file=None,
        ssh_config=pathlib.Path("/dev/null"),
        port=22,
        user="root",
    )
    prepare = module.remote_prepare("/tmp/image.tar")
    command = module.remote_transfer_command(args, "192.0.2.8", "/tmp/image.tar")
    finalize = module.remote_finalize("/tmp/image.tar", "a" * 64)
    subprocess.run(["sh", "-n"], input=prepare, text=True, check=True)
    subprocess.run(["sh", "-n"], input=finalize, text=True, check=True)
    assert command[-1] == "dd of=/tmp/image.tar.part bs=4M status=none"
    assert "mkdir -p /tmp" in prepare
    assert "sha256sum /tmp/image.tar.part" in finalize
    assert "mv -f -- /tmp/image.tar.part /tmp/image.tar" in finalize
