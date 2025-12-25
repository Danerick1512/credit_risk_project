from src.reporting import generar_informe_pdf
import os

applicant = {'Edad':30,'Ingreso mensual':'S/2800','Endeudamiento':'45%','Historial':'Regular'}
fuzzy = {'fuzzy_score':0.57,'income_membership':{'medium':0.0},'debt_ratio_membership':{'medium':0.5},'activated_rules':[{'regla':'Si puntaje es FAIR -> riesgo MEDIO','fuerza':1.0}]}
ml_probs = {'low':0.19,'medium':0.81,'high':0.0}
metrics = {'accuracy':0.888,'roc_auc':0.86}
figpath = 'output/figures/distribucion_riesgo_fuzzy.png'

os.makedirs('output/reports', exist_ok=True)
pdf_path = os.path.join('output','reports','test_report.pdf')
try:
    ok = generar_informe_pdf(pdf_path, applicant, fuzzy, ml_probs, 'RIESGO MEDIO', metrics, None, figpath)
    print('ok=', ok)
    print('exists=', os.path.exists(pdf_path))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('exception', e)
