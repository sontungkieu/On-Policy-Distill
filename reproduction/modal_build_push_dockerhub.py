"""Build and push the DPCA image from a remote Modal VM Sandbox."""

from __future__ import annotations

import json
import re
import shlex

import modal
from docker_build_contract import (
    LOCAL_ROOT,
    audit_local_context,
    normalize_image_ref,
    validate_secret_name,
)

APP_NAME = "dpca-breaking-token-barrier-docker-builder"
DOCKERFILE = "/build/context/reproduction/Dockerfile.dpca-2gpu"
CONTEXT = "/build/context"
CONTEXT_IGNORE = modal.FilePatternMatcher.from_file(LOCAL_ROOT / ".dockerignore")


def sandbox_image() -> modal.Image:
    return (
        modal.Image.from_registry("ubuntu:24.04")
        .env({"DEBIAN_FRONTEND": "noninteractive"})
        .apt_install("ca-certificates", "docker.io", "docker-buildx")
        .run_commands("mkdir -p /build/context")
        .add_local_dir(LOCAL_ROOT, CONTEXT, copy=True, ignore=CONTEXT_IGNORE)
    )


def wait_checked(process: modal.container_process.ContainerProcess, label: str) -> None:
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        detail = process.stderr.read().strip()
        if detail:
            print(detail)
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")


app = modal.App(APP_NAME)


@app.local_entrypoint()
def main(
    image: str,
    registry_secret: str,
    source_commit: str,
    context_sha256: str,
    timeout_seconds: int = 10_800,
) -> None:
    image_ref = normalize_image_ref(image)
    secret_name = validate_secret_name(registry_secret)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be an exact lowercase SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", context_sha256):
        raise ValueError("context sha256 must be an exact lowercase digest")
    if not 600 <= timeout_seconds <= 21_600:
        raise ValueError("timeout must be between 600 and 21600 seconds")
    audit = audit_local_context()
    if audit["context_sha256"] != context_sha256:
        raise RuntimeError(
            f"Docker context drifted: expected={context_sha256} actual={audit}"
        )
    print("DPCA_DOCKER_CONTEXT_JSON=" + json.dumps(audit, sort_keys=True))

    registry_secret_object = modal.Secret.from_name(secret_name)
    sandbox: modal.Sandbox | None = None
    logged_in = False
    try:
        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                "/usr/bin/dockerd",
                "-D",
                app=app,
                image=sandbox_image(),
                cpu=8,
                memory=32_768,
                timeout=timeout_seconds,
                secrets=[registry_secret_object],
                experimental_options={"vm_runtime": True},
            )
        ready = sandbox.exec(
            "sh",
            "-lc",
            "for i in $(seq 1 180); do "
            "if [ -S /var/run/docker.sock ] && docker info >/dev/null 2>&1; then "
            "echo DPCA_DOCKER_DAEMON=ready; exit 0; fi; sleep 1; done; exit 1",
        )
        wait_checked(ready, "Docker daemon readiness")
        login = sandbox.exec(
            "sh",
            "-lc",
            'test -n "${REGISTRY_USERNAME:-}" && test -n "${REGISTRY_PASSWORD:-}" '
            '&& printf %s "$REGISTRY_PASSWORD" | docker login docker.io '
            '--username "$REGISTRY_USERNAME" --password-stdin >/dev/null 2>&1',
        )
        login.wait()
        if login.returncode != 0:
            raise RuntimeError("Docker Hub authentication failed")
        logged_in = True
        print("DPCA_DOCKER_REGISTRY_AUTH=pass")

        builder = sandbox.exec(
            "sh",
            "-lc",
            "docker buildx create --name dpca-b200-builder --driver docker-container --use "
            ">/dev/null 2>&1 || docker buildx use dpca-b200-builder; "
            "docker buildx inspect --bootstrap >/dev/null 2>&1; echo DPCA_BUILDX=ready",
        )
        wait_checked(builder, "buildx bootstrap")
        quoted_ref = shlex.quote(image_ref)
        build_command = (
            "set -o pipefail; docker buildx build --platform linux/amd64 "
            f"--file {shlex.quote(DOCKERFILE)} --tag {quoted_ref} "
            f"--build-arg REPO_COMMIT={shlex.quote(source_commit)} "
            f"--build-arg CONTEXT_SHA256={shlex.quote(context_sha256)} "
            "--provenance=mode=max --sbom=true "
            "--output type=registry,compression=estargz,force-compression=true,oci-mediatypes=true "
            "--metadata-file /tmp/dpca-build-metadata.json "
            f"{shlex.quote(CONTEXT)} 2>&1"
        )
        build = sandbox.exec("bash", "-lc", build_command)
        wait_checked(build, "DPCA image build and push")

        inspect_process = sandbox.exec(
            "sh",
            "-lc",
            f"docker buildx imagetools inspect {quoted_ref} --format '{{{{json .Manifest}}}}' 2>/dev/null",
        )
        inspect_process.wait()
        if inspect_process.returncode != 0:
            raise RuntimeError("pushed image inspection failed")
        manifest = json.loads(inspect_process.stdout.read().strip())
        print(
            "DPCA_DOCKER_PUSH_JSON="
            + json.dumps(
                {
                    "status": "pushed",
                    "image": image_ref,
                    "source_commit": source_commit,
                    "context_sha256": context_sha256,
                    "manifest": manifest,
                },
                sort_keys=True,
            )
        )
    finally:
        if sandbox is not None:
            if logged_in:
                logout = sandbox.exec(
                    "sh", "-lc", "docker logout docker.io >/dev/null 2>&1 || true"
                )
                logout.wait()
            sandbox.terminate(wait=True)
