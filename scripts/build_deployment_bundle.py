from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deployment_bundle import build_bundle

if __name__ == "__main__":
    bundle, reused, manifest = build_bundle(Path(__file__).resolve().parents[1])
    print(f"bundle={bundle.name} reused={reused} files={len(manifest['files'])}")
