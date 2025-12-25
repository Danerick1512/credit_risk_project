"""Capa de Datos: accesos centralizados a rutas y utilidades del dataset y modelos."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data' / 'german_credit_data.csv'
ALT_DATASET = ROOT / 'german_credit_data.csv'
MODEL_PATH = ROOT / 'output' / 'models' / 'model_logistic.pkl'
EVALS_DIR = ROOT / 'output' / 'evaluations'


def get_dataset_path() -> str:
    if DATASET.exists():
        return str(DATASET)
    if ALT_DATASET.exists():
        return str(ALT_DATASET)
    return ''


def load_dataset(nrows: int = None):
    path = get_dataset_path()
    if not path:
        raise FileNotFoundError('Dataset no encontrado')
    return pd.read_csv(path, nrows=nrows)


def get_model_path() -> str:
    return str(MODEL_PATH)


def ensure_eval_dir():
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    return str(EVALS_DIR)
