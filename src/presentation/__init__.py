"""Capa de Presentación: wrappers de alto nivel para la interfaz del proyecto."""
from .web_app import get_web_app_path
from .cli import get_cli_path

__all__ = ["get_web_app_path", "get_cli_path"]
