"""
Storage helpers for uploaded input files and generated project paths.

The helpers in this module keep filesystem rules out of the API layer so later
changes to storage layout do not ripple through unrelated modules.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "input" / "uploads"


def ensure_input_upload_directory() -> None:
    """Create the upload directory used by the frontend file picker flow."""

    INPUT_UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Reduce a user-provided filename to a safe local basename.

    The rule is intentionally simple for the MVP: keep only the basename and
    replace spaces with underscores. A unique prefix prevents collisions.
    """

    basename = Path(filename).name.strip()
    basename = basename or "upload.bin"
    return basename.replace(" ", "_")


def save_uploaded_input_file(filename: str, content: bytes) -> tuple[Path, str]:
    """
    Persist an uploaded input file under the project input directory.

    Returns both the absolute path and the project-relative path string used by
    the task API contract.
    """

    ensure_input_upload_directory()
    safe_name = sanitize_filename(filename)
    stored_name = f"{uuid4().hex}_{safe_name}"
    absolute_path = INPUT_UPLOAD_DIRECTORY / stored_name
    absolute_path.write_bytes(content)

    relative_path = absolute_path.relative_to(PROJECT_ROOT).as_posix()
    return absolute_path, relative_path


def list_uploaded_input_files() -> list[dict[str, str | int]]:
    """
    Return uploaded input files in newest-first order for frontend reuse.

    The response intentionally mirrors the upload response shape so the frontend
    can treat freshly uploaded files and historical files through one contract.
    """

    ensure_input_upload_directory()
    entries: list[dict[str, str | int]] = []

    for file_path in sorted(
        INPUT_UPLOAD_DIRECTORY.iterdir(),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    ):
        if not file_path.is_file():
            continue

        entries.append(
            {
                "filename": file_path.name,
                "stored_path": file_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": file_path.stat().st_size,
            }
        )

    return entries
