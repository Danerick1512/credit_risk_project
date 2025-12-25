import os
import io
import joblib
import streamlit as st
import pandas as pd
import sys
import pathlib

# Ensure project root is on sys.path so `import src` works when running via Streamlit
_ROOT = pathlib.Path(__file__).resolve().parents[1]
            # DERECHA: gráficas compactas de membresía (Matplotlib primary, Plotly fallback)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import generar_datos_sinteticos, cargar_csv, mapear_columnas_comunes
from src.fuzzy_module import explicar_riesgo, puntuacion_riesgo_difuso
from src.ml_module import preparar_xy, entrenar_y_evaluar, prob_a_tres_clases
from src.reporting import generar_informe_html


MODEL_PATH = os.path.join('output', 'models', 'model_logistic.pkl')

# Estilos configurables para el bloque Resultado Final
FINAL_STYLE = {
    'icon_size': 88,            # tamaño del círculo/icono (px)
    'title_size': 28,           # tamaño del título (px)
    'just_size': 14,            # tamaño de la justificación (px)
    'box_padding': 18,          # padding del contenedor (px)
    'shadow': '0 8px 24px rgba(0,0,0,0.14)'
}


def cargar_o_entrenar_modelo(df=None):
    os.makedirs('output/models', exist_ok=True)
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data['model'], data['features'], data.get('metrics', {})

    # si no existe, entrenar con df o datos sintéticos
    if df is None:
        df = generar_datos_sinteticos(1000)
    df = mapear_columnas_comunes(df)
    features = [f for f in ['income', 'debt_ratio', 'credit_score'] if f in df.columns]
    if not features:
        return None, None, None
    X, y = preparar_xy(df, features)
    metrics, model = entrenar_y_evaluar(X, y, model_name='logistic')
    joblib.dump({'model': model, 'features': features, 'metrics': metrics}, MODEL_PATH)
    return model, features, metrics


def main():
    st.set_page_config(page_title='Evaluación Riesgo Crediticio', layout='wide')
    st.title('Evaluación de Riesgo Crediticio — Sistema Híbrido (Difuso + ML)')
    # Sidebar: entrada guiada con tooltips y rangos sugeridos
    with st.sidebar:
        st.header('Entrada del solicitante')
        age = st.number_input('Edad', min_value=18, max_value=100, value=30, help='Edad del solicitante en años')
        income = st.number_input('Ingreso mensual (S/)', min_value=0.0, value=2800.0, step=100.0, help='Ingreso mensual en soles. Rango típico: 800 – 5000')
        st.caption('Rango típico: 800 – 5000')
        debt_pct = st.slider('Nivel de endeudamiento (%)', 0, 100, 45, help='Porcentaje de ingresos comprometidos en deudas')
        credit_history = st.selectbox('Historial crediticio', ['Regular', 'Bueno', 'Malo', 'Desconocido'], help='Selecciona la evaluación cualitativa del historial')
        uploaded = st.file_uploader('Dataset CSV (opcional) para entrenar / estimar distribución', type=['csv'])
        if uploaded is not None:
            try:
                df_uploaded = pd.read_csv(uploaded)
            except Exception:
                df_uploaded = None
        else:
            df_uploaded = None

        st.markdown('---')
        st.markdown('**Reentrenamiento del modelo**')
        if st.button('Reentrenar modelo con dataset cargado / sintético'):
            try:
                df_train = df_uploaded if df_uploaded is not None else generar_datos_sinteticos(2000)
                df_train = mapear_columnas_comunes(df_train)
                features = [f for f in ['income', 'debt_ratio', 'credit_score'] if f in df_train.columns]
                if not features:
                    st.warning('No hay características válidas en el dataset para entrenar.')
                    st.markdown('**Columnas detectadas en el dataset:**')
                    st.write(list(df_train.columns))
                    st.markdown('Si tu dataset tiene columnas equivalentes con otros nombres, selecciona aquí cómo mapearlas:')
                    income_choice = st.selectbox('Columna para `income` (ingresos mensuales o anuales)', ['(ninguna)'] + list(df_train.columns))
                    debt_choice = st.selectbox('Columna para `debt_ratio` ó `debt` (deuda total)', ['(ninguna)'] + list(df_train.columns))
                    credit_choice = st.selectbox('Columna para `credit_score` (puntaje)', ['(ninguna)'] + list(df_train.columns))
                    if st.button('Aplicar mapeo y reentrenar', key='apply_map'):
                        rename_map = {}
                        if income_choice != '(ninguna)':
                            rename_map[income_choice] = 'income'
                        if credit_choice != '(ninguna)':
                            rename_map[credit_choice] = 'credit_score'
                        # Si el usuario indicó una columna que representa deuda y existe columna de ingreso, crear debt_ratio
                        if debt_choice != '(ninguna)':
                            if debt_choice in df_train.columns and income_choice != '(ninguna)' and income_choice in df_train.columns:
                                try:
                                    df_train['debt_ratio'] = df_train[debt_choice].astype(float) / df_train[income_choice].astype(float).replace({0: 1})
                                except Exception:
                                    # fallback: renombrar si tiene formato ratio
                                    rename_map[debt_choice] = 'debt_ratio'
                            else:
                                rename_map[debt_choice] = 'debt_ratio'

                        if rename_map:
                            df_train = df_train.rename(columns=rename_map)

                        features = [f for f in ['income', 'debt_ratio', 'credit_score'] if f in df_train.columns]
                        if not features:
                            st.error('No se pudieron mapear columnas a las características requeridas. Revisa las selecciones.')
                        else:
                            X, y = preparar_xy(df_train, features)
                            metrics, model = entrenar_y_evaluar(X, y, model_name='logistic')
                            joblib.dump({'model': model, 'features': features, 'metrics': metrics}, MODEL_PATH)
                            st.success(f'Modelo entrenado y guardado. Métricas: {metrics}')
                else:
                    X, y = preparar_xy(df_train, features)
                    metrics, model = entrenar_y_evaluar(X, y, model_name='logistic')
                    joblib.dump({'model': model, 'features': features, 'metrics': metrics}, MODEL_PATH)
                    st.success(f'Modelo entrenado y guardado. Métricas: {metrics}')
            except Exception as e:
                st.error(f'Error al reentrenar: {e}')

    # Cargar modelo (si existe) o entrenar uno rápido
    model, features, metrics = cargar_o_entrenar_modelo(df_uploaded)

    # Selector de vista (Ver solo ML / Difuso / Híbrido)
    view_choice = st.radio('Ver:', ['Sistema Híbrido', 'Solo ML', 'Solo Lógica Difusa'], index=0, horizontal=True)

    st.header('Resultados')
    col1, col2 = st.columns(2)

    # Preparar applicant
    income_thousands = income / 1000.0
    dr = debt_pct / 100.0

    # map historial a puntaje
    credit_score = 55.0
    if credit_history:
        lowhist = credit_history.strip().lower()
        if 'malo' in lowhist or 'poor' in lowhist:
            credit_score = 30.0
        elif 'bueno' in lowhist or 'good' in lowhist:
            credit_score = 75.0
        else:
            credit_score = 55.0

    expl = explicar_riesgo(income_thousands, dr, credit_score)

    # Nivel 2: tarjetas comparativas (Misma estructura visual para ML y Difuso)
    with col1:
        st.subheader('Lógica Difusa')
        # Card visual: texto a la izquierda, gráficas (Ingreso arriba, Endeudamiento abajo) a la derecha
        with st.container():
            col_text, col_plots = st.columns([1.0, 1.4])

            # TEXTO: índice, membresías y reglas activadas
            with col_text:
                st.markdown(f"**Índice de Riesgo Difuso:** {('Bajo' if expl['fuzzy_score']<0.4 else ('Medio' if expl['fuzzy_score']<0.7 else 'Alto'))} ({expl['fuzzy_score']:.2f})")
                st.write('Membresías (ingreso):', expl['income_membership'])
                st.write('Membresías (endeudamiento):', expl['debt_ratio_membership'])
                if expl['activated_rules']:
                    st.markdown('**Reglas activadas (top 3)**')
                    for r in sorted(expl['activated_rules'], key=lambda x: x['fuerza'], reverse=True)[:3]:
                        st.write(f"{r['regla']} (fuerza={r['fuerza']:.2f})")

            # GRÁFICAS: ingreso arriba, endeudamiento debajo (en la misma columna)
            with col_plots:
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    from src.fuzzy_module import fuzzificar_ingresos, fuzzificar_ratio_deuda

                    # Ingreso (rango reducido a 0-10 miles para visualizar bajos ingresos)
                    x_inc = np.linspace(0, 10, 400)
                    inc_low = np.array([fuzzificar_ingresos(x)['low'] for x in x_inc])
                    inc_med = np.array([fuzzificar_ingresos(x)['medium'] for x in x_inc])
                    inc_high = np.array([fuzzificar_ingresos(x)['high'] for x in x_inc])
                    fig_inc, ax_inc = plt.subplots(figsize=(6.4, 2.4))
                    fig_inc.patch.set_facecolor('white')
                    ax_inc.plot(x_inc, inc_low, color='#2ecc71', lw=2.2, label='Bajo')
                    ax_inc.plot(x_inc, inc_med, color='#f1c40f', lw=2.2, label='Medio')
                    ax_inc.plot(x_inc, inc_high, color='#e74c3c', lw=2.2, label='Alto')
                    ax_inc.set_xlim(0, 10)
                    ax_inc.set_ylim(0, 1.05)
                    ax_inc.set_title('Funciones de pertenencia - Ingreso (miles S/)', fontweight='600', fontsize=11)
                    ax_inc.set_xlabel('Ingreso (miles)', fontsize=9)
                    ax_inc.set_ylabel('Pertenencia', fontsize=9)
                    ax_inc.tick_params(labelsize=8)
                    ax_inc.legend(loc='upper right', fontsize=8, frameon=False)
                    ax_inc.axvline(income_thousands, color='#111111', linestyle='--', lw=1)
                    ax_inc.text(income_thousands, 0.03, 'Cliente', fontsize=8, ha='center')
                    plt.tight_layout()
                    st.pyplot(fig_inc)

                    # Endeudamiento (debajo)
                    x_dr = np.linspace(0, 1.0, 300)
                    dr_low = np.array([fuzzificar_ratio_deuda(x)['low'] for x in x_dr])
                    dr_med = np.array([fuzzificar_ratio_deuda(x)['medium'] for x in x_dr])
                    dr_high = np.array([fuzzificar_ratio_deuda(x)['high'] for x in x_dr])
                    fig_dr, ax_dr = plt.subplots(figsize=(6.4, 2.0))
                    fig_dr.patch.set_facecolor('white')
                    ax_dr.plot(x_dr, dr_low, color='#2ecc71', lw=2.2, label='Bajo')
                    ax_dr.plot(x_dr, dr_med, color='#f1c40f', lw=2.2, label='Medio')
                    ax_dr.plot(x_dr, dr_high, color='#e74c3c', lw=2.2, label='Alto')
                    ax_dr.set_xlim(0, 1.0)
                    ax_dr.set_ylim(0, 1.05)
                    ax_dr.set_title('Funciones de pertenencia - Endeudamiento', fontweight='600', fontsize=11)
                    ax_dr.set_xlabel('Endeudamiento (ratio)', fontsize=9)
                    ax_dr.set_ylabel('Pertenencia', fontsize=9)
                    ax_dr.tick_params(labelsize=8)
                    ax_dr.legend(loc='upper right', fontsize=8, frameon=False)
                    ax_dr.axvline(dr, color='#111111', linestyle='--', lw=1)
                    ax_dr.text(dr, 0.03, 'Cliente', fontsize=8, ha='center')
                    plt.tight_layout()
                    st.pyplot(fig_dr)

                except Exception as e:
                    # Fallback: Plotly stacked vertically
                    try:
                        import plotly.graph_objects as go
                        import numpy as _np
                        x_inc = _np.linspace(0, 10, 200)
                        inc_low = [_np.float64(fuzzificar_ingresos(x)['low']) for x in x_inc]
                        inc_med = [_np.float64(fuzzificar_ingresos(x)['medium']) for x in x_inc]
                        inc_high = [_np.float64(fuzzificar_ingresos(x)['high']) for x in x_inc]
                        figp_inc = go.Figure()
                        figp_inc.add_trace(go.Scatter(x=x_inc, y=inc_low, name='Bajo', line=dict(color='#2ecc71')))
                        figp_inc.add_trace(go.Scatter(x=x_inc, y=inc_med, name='Medio', line=dict(color='#f1c40f')))
                        figp_inc.add_trace(go.Scatter(x=x_inc, y=inc_high, name='Alto', line=dict(color='#e74c3c')))
                        figp_inc.add_vline(x=income_thousands, line=dict(color='#333', dash='dash'))
                        figp_inc.update_layout(title='Funciones de pertenencia - Ingreso (miles S/)', template='plotly_white', height=300, margin=dict(t=30,l=10,r=10,b=10))
                        st.plotly_chart(figp_inc, use_container_width='stretch')

                        x_dr = _np.linspace(0,1,150)
                        dr_low = [_np.float64(fuzzificar_ratio_deuda(x)['low']) for x in x_dr]
                        dr_med = [_np.float64(fuzzificar_ratio_deuda(x)['medium']) for x in x_dr]
                        dr_high = [_np.float64(fuzzificar_ratio_deuda(x)['high']) for x in x_dr]
                        figp_dr = go.Figure()
                        figp_dr.add_trace(go.Scatter(x=x_dr, y=dr_low, name='Bajo', line=dict(color='#2ecc71')))
                        figp_dr.add_trace(go.Scatter(x=x_dr, y=dr_med, name='Medio', line=dict(color='#f1c40f')))
                        figp_dr.add_trace(go.Scatter(x=x_dr, y=dr_high, name='Alto', line=dict(color='#e74c3c')))
                        figp_dr.add_vline(x=dr, line=dict(color='#333', dash='dash'))
                        figp_dr.update_layout(title='Funciones de pertenencia - Endeudamiento', template='plotly_white', height=260, margin=dict(t=30,l=10,r=10,b=10))
                        st.plotly_chart(figp_dr, use_container_width='stretch')
                    except Exception:
                        st.warning('No se pudieron renderizar las gráficas de membresía: ' + str(e))

    with col2:
        st.subheader('Machine Learning')
        if model is None:
            st.warning('No hay modelo ML disponible.')
            probs3 = None
            prob = None
        else:
            # predecir probabilidad (fila del solicitante)
            row = pd.DataFrame([{ 'income': income_thousands if 'income' in features else 0, 'debt_ratio': dr if 'debt_ratio' in features else 0, 'credit_score': credit_score if 'credit_score' in features else 0 }])
            try:
                prob = float(model.predict_proba(row[features])[:,1][0]) if hasattr(model, 'predict_proba') else float(model.predict(row[features])[0])
                probs3 = prob_a_tres_clases(prob)
                # Lenguaje ejecutivo
                st.markdown(f"**Probabilidad de Incumplimiento:** {prob*100:.1f}%")
                st.markdown('Distribución por nivel de riesgo (ML):')
                # Mostrar como bloque JSON / código para presentación (Bajo, Medio, Alto)
                try:
                    st.json({'Bajo': probs3['low'], 'Medio': probs3['medium'], 'Alto': probs3['high']})
                except Exception:
                    st.write({'Bajo': probs3.get('low', 0), 'Medio': probs3.get('medium', 0), 'Alto': probs3.get('high', 0)})

                # Visual: gauge (izquierda) + barra de distribución (derecha)
                gcol1, gcol2 = st.columns([1.1, 1])
                with gcol1:
                    # Intentar Plotly gauge primario
                    try:
                        import plotly.graph_objects as go
                        value_pct = prob * 100.0
                        figg = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = value_pct,
                            # number font color set to white for dark Streamlit themes
                            number = {'suffix':'%','font':{'size':22, 'color':'white'}},
                            gauge = {
                                'axis': {
                                    'range':[0,100],
                                    'tickwidth':1,
                                    # use white ticks/text so they're visible on dark background
                                    'tickcolor':'white',
                                    'tickmode':'array',
                                    'tickvals':[20,40,60,80],
                                    'ticktext':['20','40','60','80'],
                                    'tickfont': {'color': 'white'}
                                },
                                'bar': {'color':'#333', 'thickness':0.25},
                                'steps': [
                                    {'range':[0,40], 'color':'#2ecc71'},
                                    {'range':[40,70], 'color':'#f1c40f'},
                                    {'range':[70,100], 'color':'#e74c3c'},
                                ],
                                'threshold': {
                                    'line': {'color':'#000', 'width':4},
                                    'thickness': 0.75,
                                    'value': value_pct
                                }
                            }
                        ))
                        # keep transparent background; use white fonts for visibility in dark mode
                        figg.update_layout(height=320, margin={'t':20,'b':10,'l':10,'r':10}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
                        # add a larger central annotation in white so it's visible over dark background
                        figg.add_annotation(x=0.5, y=0.48, text=f"{value_pct:.1f}%", showarrow=False, font=dict(size=26, color='white'))
                        st.plotly_chart(figg, use_container_width=True)
                        st.caption('El medidor traduce la probabilidad en una señal visual inmediata de riesgo')
                    except Exception:
                        # Fallback Matplotlib semicircular gauge
                        try:
                            import matplotlib.pyplot as plt
                            import numpy as np
                            from matplotlib.patches import Wedge, Circle

                            value_pct = prob * 100.0
                            figg, axg = plt.subplots(figsize=(4,2.6))
                            # Do not force a white background — keep theme transparency —
                            # instead render text/ticks in white for dark themes.
                            # figg.patch.set_facecolor('white')  # intentionally not set
                            # axg.set_facecolor('white')
                            axg.axis('equal')
                            axg.axis('off')

                            def sector(ax, start_pct, end_pct, color, radius=1.0, width=0.28):
                                theta1 = 180 - start_pct * 1.8
                                theta2 = 180 - end_pct * 1.8
                                w = Wedge((0,0), radius, theta2, theta1, width=width, facecolor=color, edgecolor='none')
                                ax.add_patch(w)

                            # Draw sectors: low 0-40, medium 40-70, high 70-100
                            sector(axg, 0, 40, '#2ecc71')
                            sector(axg, 40, 70, '#f1c40f')
                            sector(axg, 70, 100, '#e74c3c')

                            # Needle (use light color so it stands out on dark backgrounds)
                            angle = 180 - (value_pct * 1.8)
                            rad = np.deg2rad(angle)
                            x = 0.85 * np.cos(rad)
                            y = 0.85 * np.sin(rad)
                            axg.plot([0, x], [0, y], color='white', linewidth=2.5)
                            axg.add_patch(Circle((0,0), 0.05, color='white'))

                            # Draw ticks 20,40,60,80 around the semicircle and labels (white)
                            ticks = [20,40,60,80]
                            for t in ticks:
                                ang = 180 - (t * 1.8)
                                rad_tick = np.deg2rad(ang)
                                rx = 0.88 * np.cos(rad_tick)
                                ry = 0.88 * np.sin(rad_tick)
                                tx = 0.98 * np.cos(rad_tick)
                                ty = 0.98 * np.sin(rad_tick)
                                axg.plot([rx, tx], [ry, ty], color='white', lw=1)
                                lx = 1.12 * np.cos(rad_tick)
                                ly = 1.12 * np.sin(rad_tick)
                                axg.text(lx, ly, str(t), ha='center', va='center', fontsize=9, color='white')

                            # Percentage text centered (white)
                            axg.text(0, 0.02, f"{value_pct:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold', color='white')
                            axg.set_xlim(-1.2,1.2)
                            axg.set_ylim(-0.25,1.15)
                            plt.tight_layout()
                            st.pyplot(figg)
                            st.caption('El medidor traduce la probabilidad en una señal visual inmediata de riesgo')
                        except Exception as e:
                            st.warning('No se pudo renderizar el medidor: ' + str(e))

                with gcol2:
                    # Barra de distribución ML (Plotly preferred)
                    try:
                        import plotly.graph_objects as go
                        labels = ['Alto','Medio','Bajo']
                        values = [probs3['high']*100.0, probs3['medium']*100.0, probs3['low']*100.0]
                        colors = ['#e74c3c','#f1c40f','#2ecc71']
                        figb = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
                        figb.update_layout(title='Distribución de probabilidad (ML)', template='plotly_white', yaxis_title='%', margin={'t':30,'b':10,'l':10,'r':10}, height=320)
                        st.plotly_chart(figb, use_container_width='stretch')
                    except Exception:
                        import matplotlib.pyplot as plt
                        figb, axb = plt.subplots(figsize=(3.0,2.4))
                        axb.bar(['Alto','Medio','Bajo'], [probs3['high']*100.0, probs3['medium']*100.0, probs3['low']*100.0], color=['#e74c3c','#f1c40f','#2ecc71'])
                        axb.set_ylabel('%')
                        axb.set_title('Distribución de probabilidad (ML)')
                        plt.tight_layout()
                        st.pyplot(figb)
            except Exception as e:
                st.error(f'Error al predecir: {e}')

    # Nivel 1: Resultado final destacado (visible primero después de las tarjetas)
    # Determinar etiqueta final según elección de vista
    def _map_label_from_scores(fuzzy_score, probs3_local, prob_local):
        # fuzzy_score: 0-1, probs3_local may be None
        if view_choice == 'Solo ML' and probs3_local:
            lab = max(probs3_local.items(), key=lambda x: x[1])[0]
            return 'RIESGO BAJO' if lab=='low' else ('RIESGO MEDIO' if lab=='medium' else 'RIESGO ALTO')
        if view_choice == 'Solo Lógica Difusa':
            return 'RIESGO BAJO' if fuzzy_score < 0.4 else ('RIESGO MEDIO' if fuzzy_score < 0.7 else 'RIESGO ALTO')
        # Híbrido: priorizar ML si confianza alta, else fuzzy
        if probs3_local:
            lab_ml = max(probs3_local.items(), key=lambda x: x[1])[0]
            ml_conf = max(probs3_local.values())
            lab_f = 'RIESGO BAJO' if fuzzy_score < 0.4 else ('RIESGO MEDIO' if fuzzy_score < 0.7 else 'RIESGO ALTO')
            if lab_ml == ('low' if lab_f=='RIESGO BAJO' else ('medium' if lab_f=='RIESGO MEDIO' else 'high')):
                return lab_f
            if ml_conf >= 0.65:
                return 'RIESGO BAJO' if lab_ml=='low' else ('RIESGO MEDIO' if lab_ml=='medium' else 'RIESGO ALTO')
            return lab_f
        else:
            return 'RIESGO BAJO' if expl['fuzzy_score'] < 0.4 else ('RIESGO MEDIO' if expl['fuzzy_score'] < 0.7 else 'RIESGO ALTO')

    final_label = _map_label_from_scores(expl['fuzzy_score'], locals().get('probs3', None), locals().get('prob', None))

    # Level 1 card (destacado)
    risk_color_map = {'RIESGO BAJO':'#2ecc71','RIESGO MEDIO':'#f1c40f','RIESGO ALTO':'#e74c3c'}
    color_card = risk_color_map.get(final_label, '#dddddd')
    st.markdown('---')
    st.subheader('Resultado Final')
    # Presentar en un bloque centrado grande
    colc1, colc2, colc3 = st.columns([1,6,1])
    with colc2:
        st.markdown(f"""
        <div style='border:3px solid {color_card}; padding:18px; border-radius:8px; text-align:center; background:#fffaf0;'>
          <div style='font-size:18px; font-weight:700; color:#333;'>RESULTADO FINAL</div>
          <div style='font-size:34px; font-weight:900; color:{color_card}; margin-top:6px'>{final_label} {'🔴' if 'ALTO' in final_label else ('🟡' if 'MEDIO' in final_label else '🟢')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Nivel 3: Detalles técnicos en acordeones
    with st.expander('Ver detalles técnicos'):
        st.markdown('**¿Por qué este resultado?**')
        reasons = []
        # heurísticas de explicabilidad
        if expl['debt_ratio_membership'].get('high',0) > 0.3:
            reasons.append('🔺 Alto endeudamiento (peso alto)')
        if expl['income_membership'].get('low',0) > 0.3:
            reasons.append('🔻 Ingreso bajo')
        # credit history
        reasons.append(f'⚠ Historial crediticio aproximado (puntaje): {credit_score}')
        for r in reasons:
            st.write(r)

        st.markdown('**Métricas y probabilidades**')
        if locals().get('prob', None) is not None:
            st.write(f'Probabilidad de Incumplimiento (ML): {prob*100:.1f}%')
            st.write('Distribución ML:', locals().get('probs3'))
        st.write(f'Índice de Riesgo Difuso: {expl["fuzzy_score"]:.2f}')
        st.markdown('**Membresías**')
        st.write('Ingreso:', expl['income_membership'])
        st.write('Endeudamiento:', expl['debt_ratio_membership'])

    # Decisión híbrida
    st.header('Decisión Híbrida')
    if model is None:
        final_label = 'RIESGO BAJO' if expl['fuzzy_score'] < 0.4 else ('RIESGO MEDIO' if expl['fuzzy_score'] < 0.7 else 'RIESGO ALTO')
        st.write('Resultado final (solo difuso):', final_label)
    else:
        ml_class = None
        try:
            ml_class = max(probs3.items(), key=lambda x: x[1])[0]
            ml_label = 'RIESGO BAJO' if ml_class == 'low' else ('RIESGO MEDIO' if ml_class == 'medium' else 'RIESGO ALTO')
        except Exception:
            ml_label = 'N/A'

        fuzzy_label = 'RIESGO BAJO' if expl['fuzzy_score'] < 0.4 else ('RIESGO MEDIO' if expl['fuzzy_score'] < 0.7 else 'RIESGO ALTO')
        if ml_label == fuzzy_label:
            final_label = ml_label
            justification = 'Ambos módulos coinciden.'
        else:
            perf = metrics.get('roc_auc') if metrics and not pd.isna(metrics.get('roc_auc')) else metrics.get('accuracy') if metrics else None
            prefer_ml = (perf is not None) and (perf >= 0.65)
            if prefer_ml:
                final_label = ml_label
                justification = f'Se prioriza ML (performance={perf:.2f}).'
            else:
                final_label = fuzzy_label
                justification = f'Se prioriza lógica difusa (performance ML={perf}).'

        # Mostrar resultados (estilo simple como antes)
        st.write('Lógica Difusa:', fuzzy_label, f"(score={expl['fuzzy_score']:.2f})")
        st.write('Machine Learning:', ml_label)
        st.write('Resultado Final (Híbrido):', final_label)
        st.write('Justificación:', justification)

    # Descargar informe (PDF) en un solo paso
    try:
        os.makedirs('output/reports', exist_ok=True)
        figpath = 'output/figures/distribucion_riesgo_fuzzy.png'
        # intentar crear figura si no existe
        try:
            import matplotlib.pyplot as _plt
            if not os.path.exists(figpath):
                sample = (df_uploaded if df_uploaded is not None else generar_datos_sinteticos(500)).copy()
                sample = mapear_columnas_comunes(sample)
                sample['fuzzy_score'] = sample.apply(lambda r: puntuacion_riesgo_difuso(r.get('income',0), r.get('debt_ratio',0), r.get('credit_score',0)), axis=1)
                sample['fuzzy_label'] = sample['fuzzy_score'].apply(lambda s: 'RIESGO BAJO' if s<0.4 else ('RIESGO MEDIO' if s<0.7 else 'RIESGO ALTO'))
                _plt.figure(figsize=(5,5))
                sample['fuzzy_label'].value_counts().plot.pie(autopct='%1.1f%%', ylabel='')
                _plt.title('Distribución de riesgo (difuso)')
                _plt.savefig(figpath)
        except Exception:
            figpath = None

        applicant_for_report = { 'Edad': age, 'Ingreso mensual': income, 'Endeudamiento': debt_pct, 'Historial': credit_history }
        fuzzy_for_report = expl
        ml_for_report = probs3 if 'probs3' in locals() else None
        metrics_for_report = metrics
        dist = None

        # Generar PDF en archivo temporal y leer bytes para el botón de descarga (un único widget)
        pdf_path = os.path.join('output', 'reports', f'report_{int(pd.Timestamp.now().timestamp())}.pdf')
        from src.reporting import generar_informe_pdf, generar_informe_html
        ok = generar_informe_pdf(pdf_path, applicant_for_report, fuzzy_for_report, ml_for_report, final_label, metrics_for_report, dist, figpath)
        if ok and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            st.download_button('Descargar informe (PDF)', data=pdf_bytes, file_name=os.path.basename(pdf_path), mime='application/pdf')
        else:
            # fallback: generar HTML y ofrecer descarga (único botón)
            report_path = os.path.join('output', 'reports', f'report_{int(pd.Timestamp.now().timestamp())}.html')
            generar_informe_html(report_path, applicant_for_report, fuzzy_for_report, ml_for_report, final_label, metrics_for_report, dist, figpath)
            with open(report_path, 'rb') as f:
                html_bytes = f.read()
            st.download_button('Descargar informe (HTML)', data=html_bytes, file_name=os.path.basename(report_path), mime='text/html')
    except Exception as e:
        st.error(f'Error generando informe: {e}')


if __name__ == '__main__':
    main()
