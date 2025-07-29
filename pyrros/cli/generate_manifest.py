import json
from pathlib import Path

# ----- Configuration -----
CLI_DIR         = Path(__file__).parent
PROJECT_ROOT    = CLI_DIR.parent.parent
REGISTRY_DIR    = PROJECT_ROOT / 'registry'
RECIPES_DIR     = PROJECT_ROOT / 'recipes'
OUTPUT_MANIFEST = CLI_DIR / 'manifest.json'
CATEGORIES      = ['algorithms', 'models', 'utils']

def scan_registry() -> dict[str, dict]:
    """
    Renvoie :
    {
      "algorithms": {
        "grpo": { "files": [...] }
      },
      "models": {
        "grpo": { "files": [...] }
      },
      ...
    }
    """
    registry_manifest: dict[str, dict] = {}
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
        if modules:
            registry_manifest[category] = modules
    return registry_manifest

def scan_recipes() -> dict[str, dict]:
    """
    Renvoie :
    {
      "grpo": { "files": [...] }
    }
    """
    recipes_manifest: dict[str, dict] = {}
    if not RECIPES_DIR.exists():
        return recipes_manifest

    for recipe_dir in sorted(RECIPES_DIR.iterdir()):
        if not recipe_dir.is_dir():
            continue
        files = []
        for py in recipe_dir.rglob("*.py"):
            rel = py.relative_to(recipe_dir)
            files.append(str(rel))
        if files:
            recipes_manifest[recipe_dir.name] = {"files": sorted(files)}
    return recipes_manifest

def write_manifest(data: dict[str, dict]) -> None:
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Manifest généré: {OUTPUT_MANIFEST} "
          f"({len(data.get('registry', {}))} catégories registry, "
          f"{len(data.get('recipes', {}))} recettes)")

if __name__ == "__main__":
    manifest = {
        "registry": scan_registry(),
        "recipes": scan_recipes()
    }
    write_manifest(manifest)
