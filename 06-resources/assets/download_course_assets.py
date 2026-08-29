"""Download course media while preserving files and reporting SHA-256."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


ASSETS = {
    "bus": ("bus.jpg", "https://ultralytics.com/images/bus.jpg"),
    "zidane": ("zidane.jpg", "https://ultralytics.com/images/zidane.jpg"),
    "vehicles": (
        "vehicles.mp4",
        "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    ),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(name: str, output_dir: Path, overwrite: bool) -> None:
    filename, url = ASSETS[name]
    destination = output_dir / filename
    if destination.exists() and not overwrite:
        print(f"Preserved existing file: {destination}")
        print(f"SHA-256: {digest(destination)}")
        return

    request = Request(url, headers={"User-Agent": "SAM3-Learning-Journey/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    temporary.replace(destination)
    print(f"Downloaded: {destination} ({destination.stat().st_size:,} bytes)")
    print(f"SHA-256: {digest(destination)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=["all", *ASSETS], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("input"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = ASSETS if args.asset == "all" else {args.asset: ASSETS[args.asset]}
    for name in selected:
        download(name, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
