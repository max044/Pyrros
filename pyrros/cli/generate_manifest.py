import json
from pathlib import Path

# ----- Configuration -----
CLI_DIR        = Path(__file__).parent
PROJECT_ROOT   = CLI_DIR.parent.parent
REGISTRY_DIR   = PROJECT_ROOT / 'registry'
OUTPUT_MANIFEST= CLI_DIR / 'manifest.json'
CATEGORIES     = ['algorithms', 'models']

def scan_modules() -> dict[str, dict]:
    """
    Parcourt registry/{category} et renvoie :
    {
      "algorithms": {
        "module1": {"files": [...]},
        ...
      },
      "models": {
        "moduleA": {"files": [...]},
        ...
      }
    }
    """
    manifest: dict[str, dict] = {}
    for category in CATEGORIES:
        cat_dir = REGISTRY_DIR / category
        if not cat_dir.exists():
            continue
        modules: dict[str, dict] = {}
        for module_dir in sorted(cat_dir.iterdir()):
            if not module_dir.is_dir():
                continue
            files = []
            for py in module_dir.rglob("*.py"):
                if py.name == "__init__.py":
                    continue
                rel = py.relative_to(module_dir)
                files.append(str(rel))
            if files:
                modules[module_dir.name] = {"files": sorted(files)}
        manifest[category] = modules
    return manifest

def write_manifest(data: dict[str, dict]) -> None:
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Manifest généré: {OUTPUT_MANIFEST} "
          f"({sum(len(m) for m in data.values())} modules)")

if __name__ == "__main__":
    m = scan_modules()
    write_manifest(m)
