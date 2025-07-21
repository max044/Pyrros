import json
from pathlib import Path

# Configuration
REPO_ROOT = Path(__file__).parent.parent
REGISTRY_DIR = REPO_ROOT / 'registry'
OUTPUT_MANIFEST = REPO_ROOT / 'pyrros' / 'cli' / 'manifest.json'
CATEGORIES = ['algorithms', 'models']


def scan_modules() -> dict[str, dict]:
    """
    Parcours les dossiers registry/{category} et génère la structure du manifeste.
    Retourne un dict de la forme:
    {
      "module_name": {"category": ..., "files": [...]},
      ...
    }
    """
    manifest: dict[str, dict] = {}
    for category in CATEGORIES:
        cat_dir = REGISTRY_DIR / category
        if not cat_dir.exists():
            continue
        for module_dir in sorted(cat_dir.iterdir()):
            if not module_dir.is_dir():
                continue
            module_name = module_dir.name
            # Collecter tous les fichiers .py (sans __init__.py)
            files = []
            for py_file in module_dir.rglob('*.py'):
                if py_file.name == '__init__.py':
                    continue
                # chemin relatif au dossier module
                rel_path = py_file.relative_to(module_dir)
                files.append(str(rel_path))
            if files:
                manifest[module_name] = {
                    'category': category,
                    'files': sorted(files)
                }
    return manifest


def write_manifest(data: dict[str, dict]) -> None:
    """Écrit le JSON formaté dans OUTPUT_MANIFEST."""
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Manifest généré: {OUTPUT_MANIFEST} ({len(data)} modules)")


if __name__ == '__main__':
    manifest = scan_modules()
    write_manifest(manifest)
