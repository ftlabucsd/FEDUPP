#!/usr/bin/env python3
"""Create and optionally publish a Zenodo software deposition via REST API."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive the current Git HEAD as a ZIP file, create a Zenodo "
            "deposition, upload the ZIP, and optionally publish it."
        )
    )
    parser.add_argument(
        "--metadata",
        default=REPO_ROOT / "zenodo_metadata.json",
        type=Path,
        help="Path to Zenodo deposition metadata JSON.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Override metadata.metadata.version and use it in the archive filename.",
    )
    parser.add_argument(
        "--base-url",
        default="https://sandbox.zenodo.org",
        help="Zenodo base URL. Use https://zenodo.org only for the final deposit.",
    )
    parser.add_argument(
        "--token-env",
        default="ZENODO_TOKEN",
        help="Environment variable containing the Zenodo access token.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use an existing ZIP archive instead of creating one with git archive.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the generated ZIP archive under dist/.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty working tree. The ZIP still archives committed HEAD only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata and create the archive without contacting Zenodo.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the deposition after upload. Publishing is permanent.",
    )
    parser.add_argument(
        "--publish-existing",
        metavar="DEPOSITION_ID",
        help="Publish an existing draft deposition by ID without creating or uploading files.",
    )
    parser.add_argument(
        "--upload-existing",
        metavar="DEPOSITION_ID",
        help="Upload the archive to an existing draft deposition by ID.",
    )
    parser.add_argument(
        "--update-existing",
        metavar="DEPOSITION_ID",
        help="Update metadata on an existing draft deposition by ID without uploading files.",
    )
    return parser.parse_args()


def run(command: list[str], cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def require_clean_tree(allow_dirty: bool) -> None:
    status = run(["git", "status", "--porcelain"])
    if status and not allow_dirty:
        raise SystemExit(
            "Working tree is dirty. Commit/stash changes first, or pass "
            "--allow-dirty if you intentionally want to archive committed HEAD."
        )


def load_metadata(path: Path, version: str | None) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if "metadata" not in payload:
        payload = {"metadata": payload}

    metadata = payload["metadata"]
    required = ["upload_type", "publication_date", "title", "creators", "description"]
    missing = [field for field in required if not metadata.get(field)]
    if missing:
        raise SystemExit(f"Missing required Zenodo metadata fields: {', '.join(missing)}")

    if version:
        metadata["version"] = version

    return payload


def archive_name(metadata: dict[str, Any]) -> str:
    version = metadata["metadata"].get("version", "snapshot")
    safe_version = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in version)
    return f"FEDUPP-v{safe_version}.zip"


def create_archive(name: str, keep_archive: bool) -> Path:
    if keep_archive:
        archive_dir = REPO_ROOT / "dist"
        archive_dir.mkdir(exist_ok=True)
    else:
        archive_dir = Path(tempfile.mkdtemp(prefix="fedupp-zenodo-"))

    archive_path = archive_dir / name
    prefix = name.removesuffix(".zip") + "/"
    run(["git", "archive", "--format=zip", "--prefix", prefix, "-o", str(archive_path), "HEAD"])
    return archive_path


def api_request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request) as response:
            data = response.read().decode("utf-8")
            return json.loads(data) if data else {}
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {url} failed: HTTP {error.code}\n{detail}") from error
    except URLError as error:
        raise SystemExit(f"{method} {url} failed: {error}") from error


def upload_archive(bucket_url: str, archive_path: Path, token: str) -> dict[str, Any]:
    filename = archive_path.name
    target = f"{bucket_url.rstrip('/')}/{quote(filename)}"
    parsed = urlparse(target)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(archive_path.stat().st_size),
    }

    connection = http.client.HTTPSConnection(parsed.netloc)
    try:
        with archive_path.open("rb") as handle:
            connection.request("PUT", parsed.path, body=handle, headers=headers)
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
    finally:
        connection.close()

    if response.status not in {200, 201}:
        raise SystemExit(f"PUT {target} failed: HTTP {response.status}\n{body}")
    return json.loads(body) if body else {}


def main() -> int:
    args = parse_args()

    token = os.environ.get(args.token_env)
    if args.publish_existing:
        if not token:
            raise SystemExit(f"Set {args.token_env} to a Zenodo access token first.")
        base_url = args.base_url.rstrip("/")
        published = api_request(
            "POST",
            f"{base_url}/api/deposit/depositions/{args.publish_existing}/actions/publish",
            token,
        )
        print(f"Published record: {published.get('record_url') or published.get('links', {}).get('html')}")
        print(f"DOI: {published.get('doi')}")
        return 0

    metadata_payload = load_metadata(args.metadata, args.version)

    if args.update_existing:
        if not token:
            raise SystemExit(f"Set {args.token_env} to a Zenodo access token first.")
        base_url = args.base_url.rstrip("/")
        updated = api_request(
            "PUT",
            f"{base_url}/api/deposit/depositions/{args.update_existing}",
            token,
            metadata_payload,
        )
        print(f"Updated draft: {updated.get('id', args.update_existing)}")
        print(f"Draft URL: {updated.get('links', {}).get('html')}")
        return 0

    require_clean_tree(args.allow_dirty)

    archive_path = args.archive or create_archive(archive_name(metadata_payload), args.keep_archive)
    print(f"Archive: {archive_path} ({archive_path.stat().st_size:,} bytes)")

    if args.dry_run:
        print("Dry run complete. No Zenodo requests were made.")
        return 0

    if not token:
        raise SystemExit(f"Set {args.token_env} to a Zenodo access token first.")

    base_url = args.base_url.rstrip("/")
    if args.upload_existing:
        deposition = api_request(
            "GET",
            f"{base_url}/api/deposit/depositions/{args.upload_existing}",
            token,
        )
        upload = upload_archive(deposition["links"]["bucket"], archive_path, token)
        print(f"Uploaded: {upload.get('key', archive_path.name)}")
        if upload.get("checksum"):
            print(f"Zenodo checksum: {upload['checksum']}")
        print("Existing draft updated but not published. Review it in Zenodo before publishing.")
        return 0

    deposition = api_request(
        "POST",
        f"{base_url}/api/deposit/depositions",
        token,
        metadata_payload,
    )
    deposition_id = deposition["id"]
    print(f"Created deposition: {deposition_id}")
    print(f"Draft URL: {deposition['links'].get('html')}")

    reserved = deposition.get("metadata", {}).get("prereserve_doi", {})
    if reserved.get("doi"):
        print(f"Reserved DOI: {reserved['doi']}")

    upload = upload_archive(deposition["links"]["bucket"], archive_path, token)
    print(f"Uploaded: {upload.get('key', archive_path.name)}")
    if upload.get("checksum"):
        print(f"Zenodo checksum: {upload['checksum']}")

    if args.publish:
        published = api_request(
            "POST",
            f"{base_url}/api/deposit/depositions/{deposition_id}/actions/publish",
            token,
        )
        print(f"Published record: {published.get('record_url') or published.get('links', {}).get('html')}")
        print(f"DOI: {published.get('doi')}")
    else:
        print("Draft created but not published. Review it in Zenodo before publishing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
