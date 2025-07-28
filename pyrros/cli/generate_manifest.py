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
    Parcourt registry/{category} et renvoie la structure :
    {
      "algorithms": {
        "module1": {"files": [...]},
        ...
      },
      ...
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

def scan_recipes() -> dict[str, dict]:
    """
    Parcourt recipes/<recipe_name> et renvoie :
    {
      "<recipe_name>": {"files": [...paths relative to recipe dir...]},
      ...
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
            # Si vous voulez exclure __init__.py, décommentez la ligne ci-dessous
            # if py.name == "__init__.py": continue
            rel = py.relative_to(recipe_dir)
            files.append(str(rel))
        if files:
            recipes_manifest[recipe_dir.name] = {"files": sorted(files)}
    return recipes_manifest

def write_manifest(data: dict[str, dict]) -> None:
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    total_modules = sum(len(mods) for mods in data.values())
    print(f"Manifest généré: {OUTPUT_MANIFEST} ({total_modules} modules)")

if __name__ == "__main__":
    manifest = {}
    # 1) modules registry
    manifest.update(scan_registry())
    # 2) modules recipes
    manifest["recipes"] = scan_recipes()
    # 3) write to disk
    write_manifest(manifest)
