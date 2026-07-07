"""Archive extraction fixture with a deliberate security boundary."""

from pathlib import Path
from zipfile import ZipFile


def safe_destination(base_dir: Path, member_name: str) -> Path:
    """Return a safe destination path for an archive member."""
    destination = (base_dir / member_name).resolve()
    base = base_dir.resolve()
    if destination != base and base not in destination.parents:
        raise ValueError(f"archive member escapes output directory: {member_name}")
    return destination


def extract_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    """Extract a zip archive without allowing path traversal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            destination = safe_destination(output_dir, member.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
            extracted.append(destination)
    return extracted
