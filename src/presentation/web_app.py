from pathlib import Path

def get_web_app_path() -> str:
    """Devuelve la ruta al archivo original de la app Streamlit.

    Esta función no intenta ejecutar la app; solo facilita localizar el archivo fuente
    desde la capa de presentación.
    """
    # src/presentation/web_app.py -> queremos src/web_app.py
    return str(Path(__file__).resolve().parents[1] / 'web_app.py')


def launch_streamlit_command() -> str:
    """Devuelve el comando recomendado para ejecutar la app Streamlit en PowerShell."""
    path = get_web_app_path()
    return f"python -m streamlit run {path}"
