"""Script principal para ejecutar la pipeline: carga de datos, cálculo difuso,
entrenamiento de modelos y generación de resultados.
"""
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import generar_datos_sinteticos, cargar_csv, mapear_columnas_comunes
from src.fuzzy_module import lote_puntuaciones_difusas, puntuacion_riesgo_difuso, riesgo_difuso_desde_prestamo, explicar_riesgo
from src.ml_module import preparar_xy, entrenar_y_evaluar, prob_a_tres_clases
from src.reporting import generar_informe_html
import joblib


def asegurar_directorios():
    os.makedirs('output/figures', exist_ok=True)


def ejecutar(data_path: str = None):
    asegurar_directorios()
    # Si no se pasa `--data`, preferimos cualquier CSV existente en la carpeta `data/`
    if data_path:
        df = cargar_csv(data_path)
        df = mapear_columnas_comunes(df)
        if 'target' not in df.columns:
            print('Aviso: no se detectó columna `target`. Se generará `target` sintético más abajo.')
    else:
        # Buscar CSVs en la carpeta `data/`
        csv_path = None
        data_dir = 'data'
        if os.path.isdir(data_dir):
            files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
            if files:
                # Preferir german_credit_data.csv si existe
                preferred = 'german_credit_data.csv'
                if preferred in files:
                    csv_path = os.path.join(data_dir, preferred)
                else:
                    csv_path = os.path.join(data_dir, files[0])

        if csv_path:
            print(f'Usando dataset encontrado: {csv_path}')
            df = cargar_csv(csv_path)
            df = mapear_columnas_comunes(df)
            if 'target' not in df.columns:
                print('Aviso: no se detectó columna `target`. Se generará `target` sintético más abajo.')
        else:
            # Ningún CSV disponible: generar datos sintéticos como último recurso
            print('No se encontró CSV en `data/`. Generando datos sintéticos de respaldo.')
            df = generar_datos_sinteticos(2000)

    # No se requiere que el dataset tenga income/debt_ratio/credit_score.
    # La pipeline usará columnas disponibles (por ejemplo `loan_amount`, `duration`, `Age`) y
    # generará `target` sintético si no existe.

    # Calcular característica difusa: si el dataset contiene las columnas tradicionales
    if all(c in df.columns for c in ['income', 'debt_ratio', 'credit_score']):
        df['fuzzy_risk'] = lote_puntuaciones_difusas(df)
    else:
        # Intentar calcular fuzzy_risk a partir de monto/duración/edad si están disponibles
        from src.fuzzy_module import lote_riesgo_desde_prestamo
        if any(c in df.columns for c in ['loan_amount', 'Credit amount', 'Duration', 'duration', 'Age', 'age']):
            df['fuzzy_risk'] = lote_riesgo_desde_prestamo(df)
        else:
            raise ValueError('No hay columnas adecuadas para calcular la característica difusa')

    # Si no existe columna `target`, crear una sintética con una heurística simple
    if 'target' not in df.columns:
        import numpy as np
        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Usar Age, loan_amount (Credit amount) y Duration si están disponibles
        age_col = 'Age' if 'Age' in df.columns else ('age' if 'age' in df.columns else None)
        loan_col = 'loan_amount' if 'loan_amount' in df.columns else ('Credit amount' if 'Credit amount' in df.columns else None)
        dur_col = 'Duration' if 'Duration' in df.columns else ('duration' if 'duration' in df.columns else None)

        if loan_col is None and dur_col is None and age_col is None:
            # fallback: usar fuzzy_risk
            df['target_prob'] = df['fuzzy_risk']
        else:
            # combinar en una heurística
            coeff_loan = 0.0005 if loan_col else 0.0
            coeff_dur = 0.05 if dur_col else 0.0
            coeff_age = -0.02 if age_col else 0.0

            loan_vals = df[loan_col].astype(float) if loan_col else 0
            dur_vals = df[dur_col].astype(float) if dur_col else 0
            age_vals = df[age_col].astype(float) if age_col else 0

            logit = coeff_loan * loan_vals + coeff_dur * dur_vals + coeff_age * age_vals + 0.5 * df['fuzzy_risk']
            df['target_prob'] = _sigmoid(logit)

        df['target'] = (df['target_prob'] > 0.5).astype(int)

    features_base = ['income', 'debt_ratio', 'credit_score', 'loan_amount', 'employment_length']
    # Filter features that exist
    features_base = [f for f in features_base if f in df.columns]

    results = []

    # Models without fuzzy feature
    X, y = preparar_xy(df, features_base)
    for name in ['logistic', 'random_forest']:
        mname = 'logistic' if name == 'logistic' else 'random_forest'
        metrics, model = entrenar_y_evaluar(X, y, model_name=('logistic' if name == 'logistic' else 'random'))
        metrics['with_fuzzy'] = False
        results.append(metrics)

    # Models with fuzzy feature
    Xf = X.copy()
    Xf['fuzzy_risk'] = df['fuzzy_risk']
    for name in ['logistic', 'random_forest']:
        metrics, model = entrenar_y_evaluar(Xf, y, model_name=('logistic' if name == 'logistic' else 'random'))
        metrics['with_fuzzy'] = True
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv('output/results.csv', index=False)

    # Simple plot: ROC placeholder via roc_auc metric bar
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x='model', y='roc_auc', hue='with_fuzzy')
    plt.title('ROC AUC por modelo (con / sin fuzzy)')
    plt.ylim(0, 1)
    plt.savefig('output/figures/roc_auc_comparison.png')
    print('Resultados guardados en output/results.csv y figuras en output/figures/')


def _map_credit_history_to_score(hist: str) -> float:
    if not hist:
        return 55.0
    hist_low = hist.strip().lower()
    if 'malo' in hist_low or 'poor' in hist_low:
        return 30.0
    if 'regular' in hist_low or 'fair' in hist_low:
        return 55.0
    if 'bueno' in hist_low or 'good' in hist_low:
        return 75.0
    return 55.0


def _risk_label_from_score(score: float) -> str:
    if score < 0.4:
        return 'RIESGO BAJO'
    if score < 0.7:
        return 'RIESGO MEDIO'
    return 'RIESGO ALTO'


def mostrar_evaluacion_solicitante(applicant: dict, df):
    """Muestra por consola una evaluación explicable del solicitante.

    applicant puede contener: age (años), income_monthly (moneda local), debt_ratio (porcentaje 0-1 o 0-100),
    credit_history (texto)
    """
    # Normalizar campos
    age = applicant.get('age', None)
    income_monthly = applicant.get('income_monthly', None)
    debt_ratio = applicant.get('debt_ratio', None)
    credit_history = applicant.get('credit_history', None)

    # Convertir income a miles porque las funciones fuzzy esperan "income" en miles
    income_thousands = None
    if income_monthly is not None:
        # si parece estar en unidades (ej S/ 2,800), convertir a miles
        try:
            income_val = float(str(income_monthly).replace(',', '').replace('S/', '').strip())
            income_thousands = income_val / 1000.0
        except Exception:
            income_thousands = None

    # debt_ratio en 0-1
    dr = None
    if debt_ratio is not None:
        try:
            drv = float(str(debt_ratio).replace('%', '').strip())
            if drv > 1:
                dr = drv / 100.0
            else:
                dr = drv
        except Exception:
            dr = None

    # map historial a puntaje
    credit_score = _map_credit_history_to_score(credit_history)

    # Valores por defecto si faltan
    if income_thousands is None:
        income_thousands = 2.8
    if dr is None:
        dr = 0.45
    if age is None:
        age = 30

    # Calcular riesgo difuso (usar función explicativa)
    expl = explicar_riesgo(income_thousands, dr, credit_score)
    fuzzy_score = expl['fuzzy_score']
    fuzzy_label = _risk_label_from_score(fuzzy_score)

    # Calcular predicción ML: entrenar un modelo simple en el dataframe disponible
    features = [f for f in ['income', 'debt_ratio', 'credit_score'] if f in df.columns]
    ml_label = 'N/A'
    ml_prob = None
    if features:
        X, y = preparar_xy(df, features)
        # entrenar modelo logistic y obtener el returned model y métricas
        metrics, model = entrenar_y_evaluar(X, y, model_name='logistic')
        # construir fila de entrada para el solicitante
        import pandas as _pd
        row = _pd.DataFrame([{ 'income': income_thousands if 'income' in features else 0,
                               'debt_ratio': dr if 'debt_ratio' in features else 0,
                               'credit_score': credit_score if 'credit_score' in features else 0 }])
        try:
            ml_prob = float(model.predict_proba(row[features])[:, 1][0]) if hasattr(model, 'predict_proba') else float(model.predict(row[features])[0])
            probs3 = prob_a_tres_clases(ml_prob)
            # elegir clase con mayor probabilidad
            ml_class = max(probs3.items(), key=lambda x: x[1])[0]
            ml_label = 'RIESGO BAJO' if ml_class == 'low' else ('RIESGO MEDIO' if ml_class == 'medium' else 'RIESGO ALTO')
        except Exception:
            ml_label = 'N/A'

    # Mostrar detalle explicativo de la lógica difusa
    print('\n-- Lógica Difusa (membresías y reglas activadas) --')
    print('Grado de pertenencia (Ingreso medio):', f"{expl['income_membership'].get('medium',0):.2f}")
    print('Grado de pertenencia (Endeudamiento medio):', f"{expl['debt_ratio_membership'].get('medium',0):.2f}")
    if expl['activated_rules']:
        top = max(expl['activated_rules'], key=lambda r: r['fuerza'])
        print('Regla activada:', f"\"{top['regla']}\" (fuerza={top['fuerza']:.2f})")
    else:
        print('Regla activada: ninguna (uso valor por defecto)')

    # Resultado híbrido: decidir y justificar
    final_label = None
    justification = ''
    if ml_label == 'N/A':
        final_label = fuzzy_label
        justification = 'No hay predicción ML disponible; se usa la lógica difusa.'
    else:
        if fuzzy_label == ml_label:
            final_label = fuzzy_label
            justification = 'Ambos módulos coinciden en la misma etiqueta.'
        else:
            # comparar desempeño histórico del modelo ML
            ml_score_for_choice = metrics.get('roc_auc') if not pd.isna(metrics.get('roc_auc')) else metrics.get('accuracy')
            # umbral sencillo: preferir ML si su roc_auc/accuracy > 0.65
            prefer_ml = (ml_score_for_choice is not None) and (ml_score_for_choice >= 0.65)
            if prefer_ml:
                final_label = ml_label
                justification = f'Los módulos discrepan. Se prioriza ML (performance={ml_score_for_choice:.2f}).'
            else:
                final_label = fuzzy_label
                justification = f'Los módulos discrepan. Se prioriza la lógica difusa (ML performance={ml_score_for_choice:.2f} insuficiente).'


    # Imprimir salida formateada
    print('\n🧾 Evaluación de Riesgo Crediticio\n')
    print('Datos del solicitante:')
    print(f'Edad: {int(age)} años')
    print(f'Ingreso mensual: S/ {int(income_thousands*1000):,}'.replace(',', ','))
    print(f'Nivel de endeudamiento: {int(dr*100)}%')
    print(f'Historial crediticio: {credit_history if credit_history else "Regular"}\n')

    print('\n🤖 Resultados del Sistema Inteligente')
    print('Módulo\tResultado')
    print(f'Lógica Difusa\t{fuzzy_label} (score={fuzzy_score:.2f})')
    # Si hay probs3, imprimir detalle
    if ml_prob is not None:
        print(f'Machine Learning\t{ml_label} (prob_riesgo={ml_prob:.2f})')
        print('Probabilidades ML:')
        print(f"  Prob Riesgo Bajo: {probs3['low']:.2f}")
        print(f"  Prob Riesgo Medio: {probs3['medium']:.2f}")
        print(f"  Prob Riesgo Alto: {probs3['high']:.2f}")
    else:
        print(f'Machine Learning\t{ml_label}')

    emoji = '🟢' if 'BAJO' in final_label else ('🔴' if 'ALTO' in final_label else '🟡')
    # Destacar el resultado final con un banner ASCII para mayor visibilidad
    banner_lines = []
    banner_lines.append('\n' + '=' * 70)
    banner_lines.append(f"===  RESULTADO FINAL (HÍBRIDO)  === {emoji}  {final_label}  " )
    banner_lines.append('=' * 70 + '\n')
    print('\n'.join(banner_lines))

    print('Justificación de la decisión:')
    print(justification + '\n')

    # Módulo de retroalimentación: mostrar métricas y posibilidad de reentrenar
    if features and metrics:
        acc_pct = metrics.get('accuracy', 0.0) * 100
        roc = metrics.get('roc_auc')
        if pd.isna(roc):
            roc_text = 'N/A'
        else:
            roc_text = f"{roc:.2f}"
        print('🔄 Retroalimentación / Estado del modelo:')
        print(f"Modelo entrenado (logistic). Precisión actual del modelo: {acc_pct:.1f}% | ROC AUC: {roc_text}")
        print('El sistema permite reentrenar el modelo con nuevos registros para mejorar rendimiento.\n')

    # Resultados globales: distribución de riesgos en el dataset (rápido)
    try:
        # calcular etiquetas fuzzy y ML en el dataset (muestreo si es muy grande)
        sample = df.sample(n=min(len(df), 1000), random_state=42)
        # fuzzy
        sample['fuzzy_score'] = sample.apply(lambda r: puntuacion_riesgo_difuso(r.get('income', 0), r.get('debt_ratio', 0), r.get('credit_score', 0)), axis=1)
        sample['fuzzy_label'] = sample['fuzzy_score'].apply(lambda s: _risk_label_from_score(s))
        # ml probs
        Xs, ys = preparar_xy(sample, features)
        _, model_full = entrenar_y_evaluar(Xs, ys, model_name='logistic')
        probs = model_full.predict_proba(Xs)[:, 1]
        sample['ml_label'] = [ ('RIESGO BAJO' if max(prob_a_tres_clases(p).items(), key=lambda x: x[1])[0]=='low' else ('RIESGO MEDIO' if max(prob_a_tres_clases(p).items(), key=lambda x: x[1])[0]=='medium' else 'RIESGO ALTO')) for p in probs]
        # hybrid label using same rule as above
        def hybrid_row(fr, ml_l, met):
            if ml_l == 'N/A':
                return fr
            if fr == ml_l:
                return fr
            ml_score = met.get('roc_auc') if not pd.isna(met.get('roc_auc')) else met.get('accuracy')
            if (ml_score is not None) and (ml_score >= 0.65):
                return ml_l
            return fr

        # compute distribution
        dist = sample['fuzzy_label'].value_counts(normalize=True).to_dict()
        print('📊 Resultados Globales (estimado sobre muestra):')
        for k in ['RIESGO BAJO','RIESGO MEDIO','RIESGO ALTO']:
            pct = dist.get(k, 0.0) * 100
            print(f'  {k}: {pct:.1f}%')
        # guardar figura simple
        try:
            import matplotlib.pyplot as _plt
            _plt.figure(figsize=(5,5))
            sample['fuzzy_label'].value_counts().plot.pie(autopct='%1.1f%%', ylabel='')
            _plt.title('Distribución de riesgo (difuso)')
            figpath = 'output/figures/distribucion_riesgo_fuzzy.png'
            _plt.savefig(figpath)
            # generar informe HTML básico
            try:
                applicant_for_report = { 'Edad': applicant.get('age'), 'Ingreso mensual': applicant.get('income_monthly'), 'Endeudamiento': applicant.get('debt_ratio'), 'Historial': applicant.get('credit_history') }
                fuzzy_for_report = { 'fuzzy_score': expl.get('fuzzy_score'), 'income_membership': expl.get('income_membership'), 'debt_ratio_membership': expl.get('debt_ratio_membership'), 'activated_rules': expl.get('activated_rules') }
                ml_for_report = probs3 if 'probs3' in locals() else None
                metrics_for_report = metrics if 'metrics' in locals() else None
                from src.reporting import generar_informe_html
                os.makedirs('output/reports', exist_ok=True)
                report_path = os.path.join('output', 'reports', 'report_demo.html')
                # Intentar generar PDF directamente
                pdf_path = report_path.replace('.html', '.pdf')
                from src.reporting import generar_informe_pdf
                ok = generar_informe_pdf(pdf_path, applicant_for_report, fuzzy_for_report, ml_for_report, final_label, metrics_for_report, dist, figpath)
                if ok and os.path.exists(pdf_path):
                    print(f'Informe PDF generado en {pdf_path}')
                else:
                    # fallback HTML
                    generar_informe_html(report_path, applicant_for_report, fuzzy_for_report, ml_for_report, final_label, metrics_for_report, dist, figpath)
                    print(f'Informe generado en {report_path} (PDF no disponible)')
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass


def interfaz_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Ruta a CSV con dataset (opcional)')
    parser.add_argument('--demo', action='store_true', help='Muestra una evaluación de ejemplo explicable')
    parser.add_argument('--retrain', action='store_true', help='Reentrena el modelo con el dataset dado y persiste el modelo')
    args = parser.parse_args()
    if args.demo:
        # Cargar o generar dataset mínimo para poder entrenar el modelo ML
        try:
            df = cargar_csv(args.data) if args.data else None
            if df is None:
                # usar datos sintéticos
                df = generar_datos_sinteticos(500)
            df = mapear_columnas_comunes(df)
        except Exception:
            df = generar_datos_sinteticos(500)

        applicant = {
            'age': 30,
            'income_monthly': 2800,
            'debt_ratio': 0.45,
            'credit_history': 'Regular'
        }
        mostrar_evaluacion_solicitante(applicant, df)
    else:
        if args.retrain:
            # Reentrenar con dataset proporcionado o generado
            print('Reentrenando modelo con datos de:', args.data if args.data else 'datos sintéticos')
            try:
                df_train = cargar_csv(args.data) if args.data else generar_datos_sinteticos(2000)
                df_train = mapear_columnas_comunes(df_train)
                features = [f for f in ['income', 'debt_ratio', 'credit_score'] if f in df_train.columns]
                if not features:
                    print('No se encontraron características válidas para entrenar.')
                else:
                    X, y = preparar_xy(df_train, features)
                    metrics, model = entrenar_y_evaluar(X, y, model_name='logistic')
                    os.makedirs('output/models', exist_ok=True)
                    model_path = os.path.join('output', 'models', 'model_logistic.pkl')
                    joblib.dump({'model': model, 'features': features, 'metrics': metrics}, model_path)
                    print(f'Modelo guardado en {model_path}. Métricas: {metrics}')
            except Exception as e:
                print('Error durante reentrenamiento:', e)
        else:
            ejecutar(args.data)


if __name__ == '__main__':
    interfaz_cli()
