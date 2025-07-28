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
    category: str
    files: List[str]


@app.callback(invoke_without_command=True)
def _callback(ctx: Context):
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def load_manifest() -> Dict[str, List[ModuleInfo]]:
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
    console.print(f"[bold magenta]Installing module:[/] [yellow]{name}[/]")
    for info in infos:
        base_url = f"{BASE_URL}/{VERSION}/registry/{info.category}/{name}"
        for fname in info.files:
            file_url = f"{base_url}/{fname}"
            target = Path("registry") / info.category / name / fname
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
    Install a specific module, or enter interactive mode if none is provided.
    """
    modules = load_manifest()
    available = sorted(modules.keys())

    # determine selection
    if module:
        selection = [module]
    else:
        selection = questionary.checkbox(
            "Select modules to install:",
            choices=available,
            use_arrow_keys=True,
            instruction="Use space to select, enter to confirm.",
            validate=lambda x: len(x) > 0 or "You must select at least one module."
        ).ask() or []
    if not selection:
        console.print("[bold yellow]No modules selected. Exiting.[/]")
        raise typer.Exit()

    for name in selection:
        if name not in modules:
            console.print(f"[bold red]Unknown module:[/] {name}")
            raise typer.Exit(code=1)

        infos = modules[name]
        # check for existing installation paths
        existing = [
            Path("registry") / info.category / name
            for info in infos
            if (Path("registry") / info.category / name).exists()
        ]
        if existing:
            dirs = ", ".join(str(p) for p in existing)
            confirm = questionary.confirm(
                f"Module '{name}' already installed at {dirs}. Overwrite?"
            ).ask()
            if not confirm:
                console.print(f"[yellow]Skipping {name}[/]")
                continue
        install_module(name, infos)


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
