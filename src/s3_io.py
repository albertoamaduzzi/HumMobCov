"""
s3_io.py
========
Thin boto3 wrapper for S3-compatible object stores (AWS, MinIO, Ceph, etc.).

Replaces all ``aws s3 cp`` / ``aws s3 ls`` subprocess calls so the project
runs on any machine with ``boto3`` and valid credentials — no AWS CLI needed.

Credentials are resolved in the standard boto3 order:
  1. Environment variables (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``)
  2. ``~/.aws/credentials`` / ``~/.aws/config``
  3. IAM instance profile (EC2)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

try:
    import boto3
    from botocore.exceptions import ClientError
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


# ---------------------------------------------------------------------------
# Internal: client factory
# ---------------------------------------------------------------------------

def _client(endpoint_url: str):
    if not _HAS_BOTO3:
        raise ImportError(
            "boto3 is required for S3 operations.  "
            "Install with:  pip install boto3"
        )
    return boto3.client("s3", endpoint_url=endpoint_url)


def _is_not_found(exc) -> bool:
    """Return True if the boto3 exception represents a 'key not found' error."""
    if not hasattr(exc, "response"):
        return False
    code = exc.response.get("Error", {}).get("Code", "")
    return code in ("NoSuchKey", "NoSuchBucket", "404")


# ---------------------------------------------------------------------------
# Existence check
# ---------------------------------------------------------------------------

def s3_exists(bucket: str, key: str, endpoint_url: str) -> bool:
    """Return ``True`` if ``s3://bucket/key`` exists."""
    s3 = _client(endpoint_url)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def s3_upload(
    local_path: Path | str,
    bucket: str,
    key: str,
    endpoint_url: str,
    delete_after: bool = False,
) -> bool:
    """
    Upload *local_path* to ``s3://bucket/key``.

    Returns
    -------
    bool
        ``True`` on success.  On failure the local file is kept even when
        *delete_after* is ``True``.
    """
    s3 = _client(endpoint_url)
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"  [S3 upload] File not found, skipping: {local_path}")
        return False
    try:
        s3.upload_file(str(local_path), bucket, key)
        if delete_after:
            local_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(
            f"  [S3 upload] FAILED {local_path.name} → s3://{bucket}/{key}\n"
            f"  {exc}"
        )
        return False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def s3_download(
    bucket: str,
    key: str,
    local_path: Path | str,
    endpoint_url: str,
) -> bool:
    """
    Download ``s3://bucket/key`` to *local_path*.

    Returns
    -------
    bool
        ``True`` on success.
    """
    s3 = _client(endpoint_url)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, str(local_path))
        return True
    except Exception as exc:
        print(
            f"  [S3 download] FAILED s3://{bucket}/{key} → {local_path}\n"
            f"  {exc}"
        )
        return False


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def s3_list(
    bucket: str,
    prefix: str,
    endpoint_url: str,
    suffix: str = "",
) -> list[str]:
    """
    List all object keys under *prefix* (recursive, paginated).

    Parameters
    ----------
    suffix : str
        Only return keys ending with this string (e.g. ``".parquet"``).

    Returns
    -------
    list[str]
        Full S3 object keys (no ``s3://bucket/`` prefix).
    """
    s3 = _client(endpoint_url)
    prefix = prefix.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if not suffix or k.endswith(suffix):
                keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def s3_delete(bucket: str, key: str, endpoint_url: str) -> bool:
    """Delete a single object.  Return ``True`` on success."""
    s3 = _client(endpoint_url)
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        print(f"  [S3 delete] FAILED s3://{bucket}/{key}\n  {exc}")
        return False


# ---------------------------------------------------------------------------
# JSON helpers (fully in-memory — no temp file)
# ---------------------------------------------------------------------------

def s3_read_json(bucket: str, key: str, endpoint_url: str) -> dict | None:
    """
    Download and parse a JSON object from S3.

    Returns ``None`` if the key does not exist; raises on other errors.
    """
    s3 = _client(endpoint_url)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read())
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise


def s3_write_json(
    data: dict,
    bucket: str,
    key: str,
    endpoint_url: str,
) -> bool:
    """
    Serialise *data* as JSON and upload directly (no temp file).

    Returns ``True`` on success.
    """
    s3 = _client(endpoint_url)
    try:
        body = json.dumps(data, indent=2).encode()
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        return True
    except Exception as exc:
        print(f"  [S3 write_json] FAILED s3://{bucket}/{key}\n  {exc}")
        return False


# ---------------------------------------------------------------------------
# Parquet helpers (in-memory via io.BytesIO — no temp file for reads)
# ---------------------------------------------------------------------------

def s3_read_parquet(
    bucket: str,
    key: str,
    endpoint_url: str,
) -> "pl.DataFrame | None":
    """
    Download a parquet file from S3 and return a Polars DataFrame.

    No temp file is written — bytes are read directly from the response body
    via :class:`io.BytesIO`.

    Returns ``None`` if the key does not exist or the read fails.
    """
    import polars as pl

    s3 = _client(endpoint_url)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        data = resp["Body"].read()
    except Exception as exc:
        if _is_not_found(exc):
            return None
        print(f"  [s3_read_parquet] FAILED s3://{bucket}/{key}\n  {exc}")
        return None
    try:
        return pl.read_parquet(io.BytesIO(data))
    except Exception as exc:
        print(f"  [s3_read_parquet] Parse failed for s3://{bucket}/{key}: {exc}")
        return None


def s3_read_parquet_schema(
    bucket: str,
    key: str,
    endpoint_url: str,
) -> dict | None:
    """
    Download a parquet file from S3 and return its Polars schema dict.

    Returns ``None`` if the key does not exist or the read fails.
    """
    import polars as pl

    s3 = _client(endpoint_url)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        data = resp["Body"].read()
    except Exception as exc:
        if _is_not_found(exc):
            return None
        print(f"  [s3_read_parquet_schema] FAILED s3://{bucket}/{key}\n  {exc}")
        return None
    try:
        return pl.read_parquet_schema(io.BytesIO(data))
    except Exception as exc:
        print(
            f"  [s3_read_parquet_schema] Parse failed for s3://{bucket}/{key}: {exc}"
        )
        return None
