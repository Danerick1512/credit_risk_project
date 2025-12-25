import os
from typing import Dict, Optional

from src import utils
import src.fuzzy_module as fuzzy_module
import src.ml_module as ml_module


def hybrid_decision(applicant: Dict, ml_probs: Optional[Dict] = None, model_path: Optional[str] = None) -> Dict:
    """Integra la salida del módulo difuso y del módulo ML.

    - applicant: dict con campos esperados por `explicar_riesgo` (p. ej. 'income','debt_ratio','credit_score')
    - ml_probs: opcional dict {'low':p,'medium':p,'high':p} si ya se dispone
    - model_path: ruta alternativa al modelo persistido (por defecto `output/models/model_logistic.pkl`)

    Devuelve un dict con: fuzzy (membresias y reglas), ml_probs, label_ml, label_fuzzy, final_label, rationale.
    """
    result = {
        'fuzzy': None,
        'ml_probs': None,
        'label_fuzzy': None,
        'label_ml': None,
        'final_label': None,
        'rationale': ''
    }

    # 1) Ejecutar lógica difusa
    try:
        # Asumir que explicar_riesgo acepta kwargs o un tuple
        fuzzy_out = fuzzy_module.explicar_riesgo(**{k: applicant.get(k) for k in ['income','debt_ratio','credit_score']})
        result['fuzzy'] = fuzzy_out
        # fuzzy_out puede incluir 'label' o 'score'
        if isinstance(fuzzy_out, dict):
            result['label_fuzzy'] = fuzzy_out.get('label') or fuzzy_out.get('pred') or None
    except Exception as e:
        result['fuzzy'] = {'error': str(e)}

    # 2) Obtener probabilidades ML
    if ml_probs is None:
        # intentar cargar modelo
        try:
            model_file = model_path or os.path.join('output','models','model_logistic.pkl')
            if os.path.exists(model_file):
                import joblib
                model = joblib.load(model_file)
                # Construir vector de features de forma robusta: intentar columnas esperadas
                # Usaremos income, debt_ratio, credit_score si están
                features = []
                for f in ['income','debt_ratio','credit_score','age']:
                    v = applicant.get(f, 0.0)
                    try:
                        features.append(float(v))
                    except Exception:
                        features.append(0.0)
                import numpy as _np
                X = _np.array(features).reshape(1, -1)
                # Si el modelo no acepta esa dimensión, capturamos la excepción
                probs = None
                try:
                    probs_arr = model.predict_proba(X)
                    # asumiendo binario: probs_arr[0][1]
                    p = float(probs_arr[0][1]) if probs_arr.shape[1] == 2 else float(probs_arr[0].max())
                    probs = ml_module.prob_a_tres_clases(p)
                except Exception:
                    # No se pudo predecir con el vector; fallback a None
                    probs = None
                result['ml_probs'] = probs
        except Exception:
            result['ml_probs'] = None
    else:
        result['ml_probs'] = ml_probs

    # 3) Interpretar labels ML
    if result['ml_probs']:
        lab = max(result['ml_probs'].items(), key=lambda x: x[1])[0]
        result['label_ml'] = lab

    # 4) Decisión final simple
    lf = result.get('label_fuzzy')
    lm = result.get('label_ml')
    if lf and lm:
        if lf == lm:
            final = lf
            rationale = 'Ambos módulos concuerdan.'
        else:
            # priorizar ML si confianza alta
            ml_conf = max(result['ml_probs'].values()) if result['ml_probs'] else 0.0
            if ml_conf >= 0.65:
                final = lm
                rationale = f'Discrepancia; se prioriza ML por confianza {ml_conf:.2f}.'
            else:
                final = lf
                rationale = 'Discrepancia; se prioriza lógica difusa por baja confianza ML.'
    else:
        final = lf or lm or 'unknown'
        rationale = 'Decision basada en el módulo disponible.'

    result['final_label'] = final
    result['rationale'] = rationale
    return result
