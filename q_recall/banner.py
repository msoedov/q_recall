from rich import print
from rich.console import Console

console = Console()

__version__ = "0.0.0"  # replace with your actual version


def banner():
    console.print(
        r"""
[bold cyan]
░▄▀▄░░░░░█▀▄░█▀▀░█▀▀░█▀█░█░░░█░░
░█\█░▄▄▄░█▀▄░█▀▀░█░░░█▀█░█░░░█░░
░░▀\░░░░░▀░▀░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀
[/bold cyan]
"""
    )

    console.print(f"[bright_white]v{__version__}[/bright_white]")

    console.print()
    console.print(
        "[bold blue underline]https://github.com/msoedov/q_recall[/bold blue underline]"
    )
    console.print()
