"""Ensure the Docker Hub target exists or verify first-push creation access."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    authorization: str | None = None,
) -> tuple[int, dict[str, object]]:
    headers = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return 404, {}
        detail = ""
        try:
            response = json.loads(error.read().decode("utf-8"))
            detail = str(
                response.get("detail")
                or response.get("message")
                or response.get("text")
                or ""
            )[:300]
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"Docker Hub API returned HTTP {error.code}{suffix}"
        ) from error


def _registry_push_preflight(
    username: str, token: str, repository: str
) -> dict[str, object]:
    repository_path = f"{username}/{repository}"
    auth_query = urllib.parse.urlencode(
        {
            "service": "registry.docker.io",
            "scope": f"repository:{repository_path}:pull,push",
        }
    )
    basic = base64.b64encode(f"{username}:{token}".encode()).decode("ascii")
    auth_request = urllib.request.Request(
        f"https://auth.docker.io/token?{auth_query}",
        headers={"Authorization": f"Basic {basic}"},
    )
    try:
        with urllib.request.urlopen(auth_request, timeout=30) as response:
            registry_auth = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Docker registry token request returned HTTP {error.code}"
        ) from error
    registry_token = registry_auth.get("token") or registry_auth.get("access_token")
    if not registry_token:
        raise RuntimeError("Docker registry did not return a scoped bearer token")

    upload_request = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{repository_path}/blobs/uploads/",
        headers={"Authorization": f"Bearer {registry_token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(upload_request, timeout=30) as response:
            upload_status = response.status
            upload_location = response.headers.get("Location", "")
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Docker registry push preflight returned HTTP {error.code}"
        ) from error
    if upload_status != 202 or not upload_location:
        raise RuntimeError(
            f"Unexpected Docker registry upload response: HTTP {upload_status}"
        )

    cancel_url = urllib.parse.urljoin("https://registry-1.docker.io", upload_location)
    cancel_request = urllib.request.Request(
        cancel_url,
        headers={"Authorization": f"Bearer {registry_token}"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(cancel_request, timeout=30) as response:
            cancel_status = response.status
    except urllib.error.HTTPError as error:
        if error.code not in (401, 404):
            raise RuntimeError(
                f"Docker registry upload cleanup returned HTTP {error.code}"
            ) from error
        cancel_status = error.code
    if cancel_status not in (204, 401, 404):
        raise RuntimeError(
            f"Unexpected Docker registry upload cleanup response: HTTP {cancel_status}"
        )
    return {
        "upload_status": upload_status,
        "cleanup_status": cancel_status,
        "cleanup_confirmed": cancel_status == 204,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--description", default="DPCA two-B200 reproduction runtime")
    args = parser.parse_args()

    lines = [
        line.strip()
        for line in args.credential_file.read_text().splitlines()
        if line.strip()
    ]
    if len(lines) != 2:
        raise RuntimeError(
            "Docker Hub credential file must contain username then token"
        )
    username, token = lines
    status, login = _json_request(
        "https://hub.docker.com/v2/auth/token",
        method="POST",
        payload={"identifier": username, "secret": token},
    )
    if status != 200 or not login.get("access_token"):
        raise RuntimeError("Docker Hub authentication failed")
    authorization = f"Bearer {login['access_token']}"
    repository_url = f"https://hub.docker.com/v2/namespaces/{username}/repositories/{args.repository}"
    status, _ = _json_request(repository_url, authorization=authorization)
    created = False
    creation_mode = "already-exists"
    registry_preflight: dict[str, object] | None = None
    if status == 404:
        try:
            create_status, _ = _json_request(
                f"https://hub.docker.com/v2/namespaces/{username}/repositories",
                method="POST",
                payload={
                    "namespace": username,
                    "name": args.repository,
                    "description": args.description,
                    "registry": "docker.io",
                    "is_private": False,
                },
                authorization=authorization,
            )
        except RuntimeError as error:
            if "insufficient scope" not in str(error):
                raise
            registry_preflight = _registry_push_preflight(
                username, token, args.repository
            )
            creation_mode = "automatic-on-first-push"
        else:
            if create_status not in (200, 201):
                raise RuntimeError(
                    f"Docker Hub repository creation returned {create_status}"
                )
            created = True
            creation_mode = "hub-api"
    print(
        "DOCKERHUB_REPOSITORY_READY_JSON="
        + json.dumps(
            {
                "namespace": username,
                "repository": args.repository,
                "created": created,
                "creation_mode": creation_mode,
                "registry_push_authorized": (
                    True if registry_preflight is not None or created else None
                ),
                "registry_preflight": registry_preflight,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
