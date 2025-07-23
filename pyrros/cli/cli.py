#!/usr/bin/env python3
# pyrros/cli/cli.py

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import typer
import questionary
import requests
from rich.console import Console
from typer import Context

app = typer.Typer(help="Pyrros CLI – Module Management Tool")
console = Console()

BASE_URL = "https://raw.githubusercontent.com/max044/Pyrros"
VERSION = os.getenv("PYRROS_VERSION", "main")
MANIFEST_PATH = Path(__file__).parent / "manifest.json"


@dataclass(frozen=True)
class ModuleInfo:
    """
    Metadata for a Pyrros module, as defined in the manifest.
    """
    category: str
    files: List[str]

@app.callback(invoke_without_command=True)
def _callback(ctx: Context):
    """
    Show help if no subcommand is provided.
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())

def load_manifest() -> Dict[str, List[ModuleInfo]]:
    """
    Load the manifest.json, which is structured by category, and
    return a flat mapping from module name to its ModuleInfo list.

    Raises:
        typer.Exit: if the manifest is missing or invalid JSON.
    """
    if not MANIFEST_PATH.exists():
        console.print(f"[bold red]Manifest not found:[/] {MANIFEST_PATH}")
        raise typer.Exit(code=1)

    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error parsing manifest.json:[/] {e}")
        raise typer.Exit(code=1)

    modules: Dict[str, List[ModuleInfo]] = {}
    for category, entries in raw.items():
        for name, info in entries.items():
            modules.setdefault(name, []).append(
                ModuleInfo(category=category, files=info.get("files", []))
            )
    return modules


def install_module(name: str, infos: List[ModuleInfo]) -> None:
    """
    Download and write all files for the given module to pyrros/{category}/{name}/.

    Args:
        name: The module identifier.
        infos: List of ModuleInfo instances for this module.
    """
    console.print(f"[bold magenta]Installing module:[/] [yellow]{name}[/]")
    for info in infos:
        base_url = f"{BASE_URL}/{VERSION}/registry/{info.category}/{name}"
        for fname in info.files:
            file_url = f"{base_url}/{fname}"
            target = Path("pyrros") / info.category / name / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                resp = requests.get(file_url, timeout=10)
                resp.raise_for_status()
                target.write_text(resp.text, encoding="utf-8")
                console.print(f"[green]✓ {target}[/]")
            except requests.RequestException as e:
                console.print(f"[bold red]Failed to download {file_url}:[/] {e}")
    console.print(f"[bold green]Module {name} installed successfully![/]\n")


@app.command("add")
def add(
    module: Optional[str] = typer.Argument(
        None,
        help="Name of the module to install. Omit to browse interactively."
    )
) -> None:
    """
    Install a specific module, or enter interactive mode if no module is provided.
    """
    modules = load_manifest()
    available = sorted(modules.keys())

    if module:
        if module not in modules:
            console.print(f"[bold red]Unknown module:[/] {module}")
            raise typer.Exit(code=1)
        install_module(module, modules[module])
    else:
        choice = questionary.checkbox(
            "Select modules to install:",
            choices=available
        ).ask()
        if not choice:
            console.print("[bold yellow]No modules selected. Exiting.[/]")
            raise typer.Exit()
        for name in choice:
            install_module(name, modules[name])


@app.command("list")
def list_modules() -> None:
    """
    List all modules available in the manifest.
    """
    modules = load_manifest()
    console.print("[bold cyan]Available modules:[/]")
    for name in sorted(modules.keys()):
        console.print(f"  • {name}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
