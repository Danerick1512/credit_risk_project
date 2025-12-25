from pathlib import Path

def get_cli_path() -> str:
    """Devuelve la ruta al entrypoint CLI (`src/main.py`)."""
    return str(Path(__file__).resolve().parents[1] / 'main.py')


def launch_cli_command() -> str:
    """Comando recomendado para ejecutar la CLI desde PowerShell."""
    path = get_cli_path()
    return f"python -m src.main --help"
