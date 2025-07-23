import json
from pathlib import Path

# ----- Configuration -----
# Ce script se situe dans pyrros/cli/generate_manifest.py
CLI_DIR = Path(__file__).parent
PROJECT_ROOT = CLI_DIR.parent.parent  # remonte à la racine du repo
REGISTRY_DIR = PROJECT_ROOT / 'registry'
OUTPUT_MANIFEST = CLI_DIR / 'manifest.json'
CATEGORIES = ['algorithms', 'models']


def scan_modules() -> dict[str, dict]:
    """
    Parcourt registry/{category} pour générer le manifeste.
    Renvoie un dict {module_name: {'category':..., 'files':[...]}}
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
            files: list[str] = []
            for py_file in module_dir.rglob('*.py'):
                if py_file.name == '__init__.py':
                    continue
                rel = py_file.relative_to(module_dir)
                files.append(str(rel))
            if files:
                manifest[module_name] = {
                    'category': category,
                    'files': sorted(files)
                }
    return manifest


def write_manifest(data: dict[str, dict]) -> None:
    """Écrit le JSON indenté dans le fichier OUTPUT_MANIFEST."""
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Manifest généré: {OUTPUT_MANIFEST} ({len(data)} modules)")


if __name__ == '__main__':
    manifest = scan_modules()
    write_manifest(manifest)