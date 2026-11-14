
from pathlib import Path
import hashlib

folder = Path("results/scores/calib")
manifest = folder / "manifest_sha256.txt"

lines = manifest.read_text(encoding="utf-8").splitlines()
files = list(folder.glob("*.npy"))

print("Manifest entries:", len(lines))
print("Score files:", len(files), flush=True)

verified = 0
failed = []

for index, line in enumerate(lines, start=1):
    expected_hash, filename = line.split("  ", 1)
    path = folder / filename

    if not path.is_file():
        failed.append(filename)
        continue

    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    if digest.hexdigest() == expected_hash:
        verified += 1
    else:
        failed.append(filename)

    if index % 100 == 0:
        print(f"Verified {index}/{len(lines)}", flush=True)

print("\nVERIFICATION RESULTS")
print("Manifest entries:", len(lines))
print("Score files:", len(files))
print("Hashes verified:", verified)
print("Failed:", len(failed))
print(
    "All checks passed:",
    len(lines) == 1186
    and len(files) == 1186
    and verified == 1186
    and not failed
)
