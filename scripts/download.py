import hashlib
import json
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectlab import CATEGORIES, MODEL_ID, REVISION

def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        temporary.replace(destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main():
    import re
    page = urlopen("https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads", timeout=60).read().decode()
    links = re.findall(r'href="([^"]+\.tar\.xz)"', page)
    urls = {category: next(link for link in links if link.endswith("/" + category + ".tar.xz")) for category in CATEGORIES}
    manifest = {}
    def one(category):
        archive = Path(".cache/archives") / (category + ".tar.xz")
        digest = download(urls[category], archive)
        target = Path(".cache/data")
        target.mkdir(parents=True, exist_ok=True)
        if not (target / category / "train").exists():
            with tarfile.open(archive) as tar:
                tar.extractall(target, filter="data")
        count = len(list((target / category).rglob("*.png")))
        print(category, archive.stat().st_size, count, flush=True)
        return category, {"url": urls[category], "sha256": digest, "bytes": archive.stat().st_size, "png_files_including_masks": count}
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest.update(pool.map(one, CATEGORIES))
    model = Path(".cache/model")
    for name in ["config.json", "model.safetensors", "preprocessor_config.json"]:
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        digest = download(url, model / name)
        manifest["model/" + name] = {"url": url, "sha256": digest}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/downloads.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
