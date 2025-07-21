import os
import json
from pathlib import Path
from typing import List, Dict

import typer
import questionary
import requests
from rich.console import Console
from dotenv import load_dotenv

# Init Typer app
typing_app = typer.Typer(help="CLI de gestion des modules Pyrros")
console = Console()
load_dotenv()

# Constants
BASE_URL = "https://raw.githubusercontent.com/max044/Pyrros"
VERSION = os.getenv("PYRROS_VERSION", "main")
MANIFEST_PATH = Path(__file__).parent / "manifest.json"

# Types
class ModuleInfo(typer.models.BaseModel):
    category: str
    path: str
    files: List[str]

# --- Manifest loading ---
def load_manifest() -> Dict[str, ModuleInfo]:
    """Charge le manifeste statique des modules depuis manifest.json."""
    if not MANIFEST_PATH.exists():
        console.print(f"[bold red]Manifeste introuvable: {MANIFEST_PATH}[/]")
        raise typer.Exit(code=1)
    data = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    modules = {name: ModuleInfo(**info) for name, info in data.items()}
    return modules

# --- Installation logic ---
def install_module(name: str, info: ModuleInfo) -> None:
    """Télécharge les fichiers du module depuis GitHub raw."""
    console.print(f"[bold magenta]Installation du module:[/] [yellow]{name}[/]")
    base = f"{BASE_URL}/{VERSION}/pyrros/{info.category}/{name}"
    for file in info.files:
        rel = Path(info.category) / name / file
        url = f"{base}/{file}"
        target = Path("pyrros") / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            target.write_text(resp.text, encoding='utf-8')
            console.print(f"[green]  ✓ {rel}")
        except requests.HTTPError as e:
            console.print(f"[bold red]Erreur HTTP {e.response.status_code} pour {url}[/]")
    console.print(f"[bold green]Module {name} installé ![/]")

# --- CLI Commands ---
@typing_app.command()
def add(
    name: str = typer.Argument(
        None, help="Nom du module à installer. Sans argument, lance le mode browse."),
):
    """
    Installe un module spécifique ou ouvre l'explorateur interactif.
    """
    modules = load_manifest()
    if name:
        if name not in modules:
            console.print(f"[bold red]Module inconnu: {name}[/]")
            raise typer.Exit(code=1)
        install_module(name, modules[name])
    else:
        browse(modules)

@typing_app.command()
def browse(modules: Dict[str, ModuleInfo]) -> None:
    """
    Liste les modules disponibles et permet leur sélection.
    """
    choix = questionary.checkbox(
        "Modules disponibles:",
        choices=list(modules.keys())
    ).ask()
    if not choix:
        console.print("[bold yellow]Aucun module sélectionné.[/]")
        raise typer.Exit()
    for name in choix:
        install_module(name, modules[name])

if __name__ == "__main__":
    typing_app()
