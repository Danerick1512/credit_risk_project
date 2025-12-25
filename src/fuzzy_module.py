"""
Módulo simple de lógica difusa sin dependencias externas.
Define funciones de membresía y un conjunto de reglas
para calcular un `riesgo_difuso` en [0,1].
"""
from typing import Dict

def memb_triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

def fuzzificar_ingresos(income: float) -> Dict[str, float]:
    # income in thousands
    # Ajuste de fronteras más realistas para ingresos mensuales en Perú
    # income (miles): por ejemplo, 0.5 => S/500
    # - Bajo: hasta ~S/1,200 (1.2 miles)
    # - Medio: entre ~S/800 y S/3,000 (0.8 - 3.0 miles)
    # - Alto: desde ~S/2,500 en adelante
    return {
        'low': memb_triangular(income, 0.0, 0.0, 1.2),
        'medium': memb_triangular(income, 0.8, 1.8, 3.0),
        'high': memb_triangular(income, 2.5, 5.0, 20.0),
    }

def fuzzificar_ratio_deuda(dr: float) -> Dict[str, float]:
    # dr is ratio 0-1
    # Hacemos la función de endeudamiento algo más sensible (p.ej. 0.35 ya comienza a activar 'high')
    return {
        'low': memb_triangular(dr, 0.0, 0.0, 0.15),
        'medium': memb_triangular(dr, 0.1, 0.25, 0.5),
        'high': memb_triangular(dr, 0.35, 0.6, 1.0),
    }

def fuzzificar_puntaje_credito(score: float) -> Dict[str, float]:
    # score 0-100
    return {
        'poor': memb_triangular(score, 0, 0, 50),
        'fair': memb_triangular(score, 30, 55, 75),
        'good': memb_triangular(score, 60, 80, 100),
    }

def puntuacion_riesgo_difuso(income: float, debt_ratio: float, credit_score: float) -> float:
    inc = fuzzificar_ingresos(income)
    dr = fuzzificar_ratio_deuda(debt_ratio)
    cs = fuzzificar_puntaje_credito(credit_score)

    # Valores lingüísticos mapeados a riesgo numérico: low=0, medium=0.5, high=1
    reglas = []

    reglas.append((min(inc['low'], dr['high']), 1.0))
    reglas.append((cs['poor'], 1.0))
    reglas.append((min(inc['medium'], dr['medium']), 0.5))
    reglas.append((min(inc['high'], cs['good']), 0.0))
    reglas.append((dr['low'], 0.0))
    reglas.append((cs['fair'], 0.5))

    num = 0.0
    den = 0.0
    for fuerza, valor_salida in reglas:
        num += fuerza * valor_salida
        den += fuerza

    if den == 0:
        return 0.5
    return float(num / den)

def lote_puntuaciones_difusas(df):
    # espera columnas: 'income', 'debt_ratio', 'credit_score'
    return df.apply(lambda r: puntuacion_riesgo_difuso(r['income'], r['debt_ratio'], r['credit_score']), axis=1)


def fuzzificar_monto_prestamo(amount: float) -> Dict[str, float]:
    return {
        'small': memb_triangular(amount, 0, 0, 2000),
        'medium': memb_triangular(amount, 1500, 4000, 8000),
        'large': memb_triangular(amount, 6000, 15000, 30000),
    }


def fuzzificar_duracion(d: float) -> Dict[str, float]:
    return {
        'short': memb_triangular(d, 0, 0, 12),
        'medium': memb_triangular(d, 6, 24, 48),
        'long': memb_triangular(d, 36, 60, 120),
    }


def fuzzificar_edad(age: float) -> Dict[str, float]:
    return {
        'young': memb_triangular(age, 18, 20, 30),
        'adult': memb_triangular(age, 25, 40, 60),
        'senior': memb_triangular(age, 55, 70, 100),
    }


def riesgo_difuso_desde_prestamo(loan_amount: float, duration: float, age: float) -> float:
    la = fuzzificar_monto_prestamo(loan_amount)
    du = fuzzificar_duracion(duration)
    ag = fuzzificar_edad(age)

    reglas = []
    reglas.append((min(la['large'], du['long']), 1.0))
    reglas.append((min(la['small'], du['short']), 0.0))
    reglas.append((min(ag['young'], la['large']), 1.0))
    reglas.append((min(ag['senior'], la['small']), 0.0))
    reglas.append((min(la['medium'], du['medium']), 0.5))

    num = 0.0
    den = 0.0
    for fuerza, valor_salida in reglas:
        num += fuerza * valor_salida
        den += fuerza
    if den == 0:
        return 0.5
    return float(num / den)


def lote_riesgo_desde_prestamo(df):
    # espera columnas: 'loan_amount', 'Duration' o 'duration' y 'Age' o 'age'
    def _fila(r):
        loan = r.get('loan_amount') if 'loan_amount' in r.index else r.get('Credit amount', 0)
        dur = r.get('duration') if 'duration' in r.index else r.get('Duration', 0)
        age = r.get('age') if 'age' in r.index else r.get('Age', 0)
        return riesgo_difuso_desde_prestamo(float(loan), float(dur), float(age))

    return df.apply(_fila, axis=1)


def explicar_riesgo(income: float, debt_ratio: float, credit_score: float) -> Dict:
    """Devuelve información explicativa: membresías de ingreso y deuda, y reglas activadas.

    income: en miles (como usa el módulo)
    debt_ratio: en 0-1
    credit_score: 0-100
    """
    inc = fuzzificar_ingresos(income)
    dr = fuzzificar_ratio_deuda(debt_ratio)
    cs = fuzzificar_puntaje_credito(credit_score)

    # Reglas (misma lógica que en `puntuacion_riesgo_difuso`)
    reglas = [
        ((min(inc['low'], dr['high'])), 'Si ingreso es BAJO y endeudamiento es ALTO -> riesgo ALTO', 1.0),
        ((cs['poor']), 'Si puntaje es POOR -> riesgo ALTO', 1.0),
        ((min(inc['medium'], dr['medium'])), 'Si ingreso es MEDIO y endeudamiento es MEDIO -> riesgo MEDIO', 0.5),
        ((min(inc['high'], cs['good'])), 'Si ingreso es ALTO y puntaje es GOOD -> riesgo BAJO', 0.0),
        ((dr['low']), 'Si endeudamiento es BAJO -> riesgo BAJO', 0.0),
        ((cs['fair']), 'Si puntaje es FAIR -> riesgo MEDIO', 0.5),
    ]

    # Calcular fuerza total y salida ponderada
    num = 0.0
    den = 0.0
    for fuerza, texto, valor_salida in reglas:
        num += fuerza * valor_salida
        den += fuerza

    score = float(num / den) if den != 0 else 0.5

    # Seleccionar reglas activadas (fuerza > 0)
    activadas = []
    for fuerza, texto, valor_salida in reglas:
        if fuerza > 0:
            activadas.append({'regla': texto, 'fuerza': float(fuerza), 'valor_salida': float(valor_salida)})

    return {
        'income_membership': inc,
        'debt_ratio_membership': dr,
        'credit_score_membership': cs,
        'activated_rules': activadas,
        'fuzzy_score': score,
    }
