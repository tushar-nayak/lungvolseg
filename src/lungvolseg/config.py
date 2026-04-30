from pathlib import Path

CLASS_NAMES = {
    0: "background",
    1: "lung",
}

DEFAULT_SPACING = (1.5, 1.5, 1.5)
DEFAULT_PATCH_SIZE = (96, 96, 96)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
