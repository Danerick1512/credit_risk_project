import numpy as np
import pandas as pd
from typing import Tuple

def generar_datos_sinteticos(n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    # ingresos en miles
    ingreso = rng.normal(50, 20, size=n).clip(5, 200)
    # ratio de deuda 0-1
    ratio_deuda = rng.beta(2, 5, size=n)
    # puntaje crediticio 0-100
    puntaje_credito = (rng.normal(60, 15, size=n)).clip(0, 100)
    # edad y antigüedad laboral
    edad = rng.normal(40, 12, size=n).clip(18, 90)
    antiguedad = rng.poisson(5, size=n)
    monto_prestamo = rng.normal(20, 10, size=n).clip(1, 150)
    # probabilidad base de riesgo (verdad sintética)
    risk_logit = (
        -0.03 * ingreso
        + 3.5 * ratio_deuda
        -0.02 * puntaje_credito
        + 0.01 * monto_prestamo
        -0.01 * antiguedad
        + rng.normal(0, 1, size=n)
    )
    prob = 1 / (1 + np.exp(-risk_logit))
    objetivo = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        'income': ingreso,
        'debt_ratio': ratio_deuda,
        'credit_score': puntaje_credito,
        'age': edad,
        'employment_length': antiguedad,
        'loan_amount': monto_prestamo,
        'target': objetivo,
    })
    return df


def cargar_csv(path: str):
    df = pd.read_csv(path)
    return df


def mapear_columnas_comunes(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea nombres de columnas comunes (inglés/español) a los nombres usados por la pipeline.

    Objetivo: normalizar columnas a `income`, `debt_ratio`, `credit_score`, `loan_amount`,
    `employment_length`, `target` cuando sea posible.
    """
    cols = {c.lower(): c for c in df.columns}

    # Sinónimos para cada campo
    synonyms = {
        'income': ['income', 'ingreso', 'monthly_income', 'salary', 'ingresos'],
        'debt_ratio': ['debt_ratio', 'dti', 'debttoincome', 'debt_to_income', 'ratio_deuda'],
        'credit_score': ['credit_score', 'score', 'fico', 'puntaje', 'credit_score_value'],
        'loan_amount': ['loan_amount', 'loan', 'monto', 'amount', 'credit amount', 'credit_amount', 'creditamount'],
        'duration': ['duration', 'loan_duration', 'term', 'duracion'],
        'employment_length': ['employment_length', 'emp_len', 'antiguedad', 'tiempo_empleo'],
        'target': ['target', 'default', 'loan_status', 'label', 'y']
    }

    rename_map = {}
    for target_col, possibles in synonyms.items():
        for p in possibles:
            if p in cols:
                rename_map[cols[p]] = target_col
                break

    # Si existe columna 'debt' y 'income' podemos crear debt_ratio
    if 'debt' in cols and 'income' in cols and 'debt_ratio' not in rename_map:
        df['debt_ratio'] = df[cols['debt']] / df[cols['income']].replace({0: 1})
        rename_map['debt_ratio'] = 'debt_ratio'

    if rename_map:
        df = df.rename(columns=rename_map)
    return df
