from rich.console import Console
import questionary
from sys import argv, exit
from enum import Enum
from requests import get
from dotenv import load_dotenv
from os import getenv

class CLIMode(Enum):
    ADD = 1
    BROWSE = 2

class CLI():
    def __init__(self):
        self.mode = None
        self.target_module = None
        self.console = Console()
        self.choices = []
        self.urls = [
            "https://api.github.com/repos/max044/Pyrros/contents/pyrros/algorithms",
            "https://api.github.com/repos/max044/Pyrros/contents/pyrros/models"
        ]
        load_dotenv()

    def fetch_available_modules(self) -> list[str]:
        token = getenv("GITHUB_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        sets = []
        for url in self.urls:
            response = get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                names = set(item['name'] for item in data if item['type'] == 'dir')
                sets.append(names)
            else:
                self.console.print(f"[bold red]Error fetching available modules from {url}.[/]")
                exit(1)
        if sets:
            common = set.intersection(*sets)
            return sorted(common)
        return []

    def fetch_content(self, module_name: str) -> dict:
        import base64
        from pathlib import Path
        token = getenv("GITHUB_TOKEN")
        headers = {"Authorization": f"token {token}"} if token else {}
        base_urls = {
            "algorithms": f"https://api.github.com/repos/max044/Pyrros/contents/pyrros/algorithms/{module_name}",
            "models": f"https://api.github.com/repos/max044/Pyrros/contents/pyrros/models/{module_name}"
        }
        results = {"algorithms": [], "models": []}

        def clone_folder(url, local_dir, file_list):
            resp = get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        if item["type"] == "file":
                            file_url = item["url"]
                            rel_path = item["path"].split(f"pyrros/")[-1]  # e.g. algorithms/module/file.py
                            local_path = Path("pyrros") / rel_path
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            file_resp = get(file_url, headers=headers)
                            if file_resp.status_code == 200:
                                file_data = file_resp.json()
                                if "content" in file_data:
                                    content = base64.b64decode(file_data["content"]).decode("utf-8")
                                    with open(local_path, "w") as f:
                                        f.write(content)
                                    file_list.append(str(local_path))
                                else:
                                    self.console.print(f"[bold yellow]No content in file {item['name']} for {local_dir}.[/]")
                            else:
                                self.console.print(f"[yellow]Failed to fetch file {item['name']} for {local_dir} (status {file_resp.status_code}).[/]")
                        elif item["type"] == "dir":
                            # Recursive call for subfolder
                            clone_folder(item["url"], local_dir / item["name"], file_list)
                else:
                    self.console.print(f"[bold yellow]No files found at {url}.[/]")
            elif resp.status_code == 404:
                pass  # Not found is not an error here
            else:
                self.console.print(f"[bold red]Error fetching content from {url} (status {resp.status_code}).[/]")

        for key, url in base_urls.items():
            local_dir = Path(f"pyrros/{key}/{module_name}")
            clone_folder(url, local_dir, results[key])
        if not results["algorithms"] and not results["models"]:
            self.console.print(f"[bold red]Module {module_name} not found in algorithms or models![/]")
        return results

    def print_help(self) -> None:
        self.console.print("[bold cyan]Usage:[/] [yellow]pyrros --help[/] [dim]or[/] [yellow]pyrros add <name>[/] [dim]to install a precise module or[/] [yellow]pyrros add[/] [dim]to browse available modules.")
        self.console.print("[bold underline]Available commands:[/]")
        self.console.print("  [green]--help[/]: Show this help message")
        self.console.print("  [green]add <name>[/]: Install a specific module by name")
        self.console.print("  [green]add[/]: Browse available things")

    def check_args(self, args: list[str]) -> None:
        if (len(args)) < 2 or (len(args) > 3):
            self.print_help()
            exit(1)
        if "--help" in args:
            self.print_help()
            exit(0)
        if 'add' in args:
            if len(args) == 2:
                self.mode = CLIMode.BROWSE
            elif len(args) == 3:
                self.available = self.fetch_available_modules()
                if args[2] in self.available:
                    self.mode = CLIMode.ADD
                    self.target_module = args[2]
                else:
                    self.print_help()
                    exit(1)
            else:
                self.print_help()
                exit(1)

    def run(self) -> None:
        self.check_args(argv)
        if self.mode == CLIMode.ADD:
            if not self.target_module:
                self.console.print("[bold red]No module specified for installation![/]")
                return
            self.console.print(f"[bold magenta]Installing module: {self.target_module}[/]")
            content = self.fetch_content(self.target_module)
            wrote = False
            if content["algorithms"]:
                self.console.print(f"[green]Cloned files in algorithms/{self.target_module}:\n  - " + "\n  - ".join(content["algorithms"]))
                wrote = True
            if content["models"]:
                self.console.print(f"[green]Cloned files in models/{self.target_module}:\n  - " + "\n  - ".join(content["models"]))
                wrote = True
            if wrote:
                self.console.print(f"[bold green]Module {self.target_module} installed successfully![/]")
            else:
                self.console.print(f"[bold red]Module {self.target_module} not found in algorithms or models![/]")

        elif self.mode == CLIMode.BROWSE:
            self.console.print("[bold magenta]Welcome to Pyrros CLI![/]")
            self.choices = self.fetch_available_modules()
            if not self.choices:
                self.console.print("[bold red]No available modules found![/]")
                return
            selected = questionary.checkbox(
                "Which modules do you want to install?",
                choices=self.choices
            ).ask()
            if selected:
                from pathlib import Path
                Path(f"pyrros/algorithms/").mkdir(parents=True, exist_ok=True)
                Path(f"pyrros/models/").mkdir(parents=True, exist_ok=True)

                for module in selected:
                    self.console.print(f"[bold green]Installing module:[/] [yellow]{module}[/]")
                    content = self.fetch_content(module)
                    wrote = False
                    if content["algorithms"]:
                        self.console.print(f"[green]Cloned files in algorithms/{module}:\n  - " + "\n  - ".join(content["algorithms"]))
                        wrote = True
                    if content["models"]:
                        self.console.print(f"[green]Cloned files in models/{module}:\n  - " + "\n  - ".join(content["models"]))
                        wrote = True
                    if wrote:
                        self.console.print(f"[bold green]Module {module} installed successfully![/]")
                    else:
                        self.console.print(f"[bold red]Module {module} not found in algorithms or models![/]")
            else:
                self.console.print("[bold red]No modules selected for installation![/]")
                return
            
def main() -> None:
    cli = CLI()
    cli.run()

if __name__ == "__main__":
    main()
