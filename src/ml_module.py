import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from typing import Tuple, Dict

def preparar_xy(df: pd.DataFrame, features: list, target_col: str = 'target'):
    X = df[features].copy()
    y = df[target_col].values
    return X, y


def entrenar_y_evaluar(X, y, model_name: str = 'logistic') -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    if model_name == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else preds

    # Cálculo seguro de métricas cuando y_test contiene una sola clase
    from numpy import unique
    if len(unique(y_test)) > 1:
        roc = float(roc_auc_score(y_test, probs))
        precision = float(precision_score(y_test, preds, zero_division=0))
        recall = float(recall_score(y_test, preds, zero_division=0))
    else:
        roc = float('nan')
        precision = float(precision_score(y_test, preds, zero_division=0))
        recall = float(recall_score(y_test, preds, zero_division=0))

    metrics = {
        'model': model_name,
        'accuracy': float(accuracy_score(y_test, preds)),
        'roc_auc': roc,
        'precision': precision,
        'recall': recall,
    }
    return metrics, model


def prob_a_tres_clases(prob: float) -> Dict[str, float]:
    """Convierte una probabilidad binaria (riesgo) en una distribución suave de tres clases.

    Usamos funciones triangulares centradas en 0 (BAJO), 0.5 (MEDIO), 1 (ALTO).
    """
    def tri(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)

    p_low = tri(prob, 0.0, 0.0, 0.5)
    p_med = tri(prob, 0.0, 0.5, 1.0)
    p_high = tri(prob, 0.5, 1.0, 1.0)

    total = p_low + p_med + p_high
    if total == 0:
        # fallback: asignar según cercanía
        return {'low': max(0.0, 1 - prob), 'medium': 0.0 + (1 - abs(prob - 0.5)), 'high': max(0.0, prob)}
    return {'low': float(p_low / total), 'medium': float(p_med / total), 'high': float(p_high / total)}
