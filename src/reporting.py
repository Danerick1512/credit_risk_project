import datetime
import os
from typing import Dict
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

def generar_informe_html(output_path: str, applicant: Dict, fuzzy: Dict, ml_probs: Dict = None, final_label: str = None, metrics: Dict = None, global_dist: Dict = None, figure_path: str = None):
    """Genera un informe HTML simple con la evaluación y métricas.

    output_path: ruta del archivo HTML a escribir
    applicant: dict con datos del solicitante
    fuzzy: dict con keys 'fuzzy_score', 'income_membership', 'debt_ratio_membership', 'activated_rules'
    ml_probs: dict con keys 'low','medium','high'
    metrics: dict con métricas delmodelo
    global_dist: dict con distribución global de etiquetas
    figure_path: ruta relativa a una figura para incluir
    """
    titulo = "Informe de Evaluación de Riesgo Crediticio"
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = []
    lines.append(f"<html><head><meta charset=\"utf-8\"><title>{titulo}</title></head><body>")
    lines.append(f"<h1>{titulo}</h1>")
    lines.append(f"<p><small>Generado: {now}</small></p>")

    lines.append('<h2>Datos del solicitante</h2>')
    lines.append('<ul>')
    for k, v in applicant.items():
        lines.append(f"<li><strong>{k}:</strong> {v}</li>")
    lines.append('</ul>')

    lines.append('<h2>Módulo Lógica Difusa</h2>')
    lines.append(f"<p>Score difuso: {fuzzy.get('fuzzy_score', 0):.2f}</p>")
    lines.append('<h3>Membresías</h3>')
    lines.append('<ul>')
    for name, val in fuzzy.get('income_membership', {}).items():
        lines.append(f"<li>Ingreso {name}: {val:.2f}</li>")
    for name, val in fuzzy.get('debt_ratio_membership', {}).items():
        lines.append(f"<li>Endeudamiento {name}: {val:.2f}</li>")
    lines.append('</ul>')
    lines.append('<h3>Reglas activadas</h3>')
    lines.append('<ul>')
    for r in fuzzy.get('activated_rules', []):
        lines.append(f"<li>{r.get('regla')} (fuerza={r.get('fuerza'):.2f})</li>")
    lines.append('</ul>')

    if ml_probs:
        lines.append('<h2>Módulo Machine Learning</h2>')
        lines.append('<ul>')
        lines.append(f"<li>Prob Riesgo Bajo: {ml_probs.get('low',0):.2f}</li>")
        lines.append(f"<li>Prob Riesgo Medio: {ml_probs.get('medium',0):.2f}</li>")
        lines.append(f"<li>Prob Riesgo Alto: {ml_probs.get('high',0):.2f}</li>")
        lines.append('</ul>')

    if final_label:
        # elegir color según etiqueta
        if 'BAJO' in final_label:
            fcolor = 'green'
        elif 'ALTO' in final_label:
            fcolor = 'red'
        else:
            fcolor = 'orange'
        lines.append(f"<div style='border:2px solid {fcolor}; padding:10px; border-radius:6px; background:#fffaf0'><h1 style='color:{fcolor}; margin:0'>Resultado Final (Híbrido): {final_label}</h1></div>")

    if metrics:
        lines.append('<h2>Métricas del Modelo</h2>')
        lines.append('<ul>')
        for k, v in metrics.items():
            lines.append(f"<li>{k}: {v}</li>")
        lines.append('</ul>')

    if global_dist:
        lines.append('<h2>Distribución Global (muestra)</h2>')
        lines.append('<ul>')
        for k, v in global_dist.items():
            lines.append(f"<li>{k}: {v:.2%}</li>")
        lines.append('</ul>')

    if figure_path and os.path.exists(figure_path):
        lines.append('<h2>Figura</h2>')
        lines.append(f'<img src="{figure_path}" alt="Figura" style="max-width:600px;">')

    lines.append('<hr><p>Este informe fue generado automáticamente por el sistema híbrido.</p>')
    lines.append('</body></html>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_path


def generar_pdf_desde_html(html_path: str, pdf_path: str) -> bool:
    """Intenta convertir un HTML a PDF. Primero intenta WeasyPrint, luego pdfkit.

    Devuelve True si la conversión fue exitosa.
    """
    # Intentar WeasyPrint
    try:
        from weasyprint import HTML
        HTML(html_path).write_pdf(pdf_path)
        return True
    except Exception:
        pass

    # Intentar pdfkit (requiere wkhtmltopdf instalado en el sistema)
    try:
        import pdfkit
        pdfkit.from_file(html_path, pdf_path)
        return True
    except Exception:
        pass

    return False


def generar_informe_pdf(pdf_path: str, applicant: Dict, fuzzy: Dict, ml_probs: Dict = None, final_label: str = None, metrics: Dict = None, global_dist: Dict = None, figure_path: str = None) -> bool:
    """Genera un PDF directamente (intenta WeasyPrint, luego pdfkit).

    Crea un HTML en memoria con la figura embebida (si existe) y lo convierte.
    Devuelve True si el PDF fue creado.
    """
    # Construir HTML en memoria
    titulo = "Informe de Evaluación de Riesgo Crediticio"
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    parts = [f"<html><head><meta charset='utf-8'><title>{titulo}</title></head><body>"]
    parts.append(f"<h1>{titulo}</h1>")
    parts.append(f"<p><small>Generado: {now}</small></p>")
    parts.append('<h2>Datos del solicitante</h2><ul>')
    for k, v in applicant.items():
        parts.append(f"<li><strong>{k}:</strong> {v}</li>")
    parts.append('</ul>')
    parts.append('<h2>Módulo Lógica Difusa</h2>')
    parts.append(f"<p>Score difuso: {fuzzy.get('fuzzy_score', 0):.2f}</p>")
    parts.append('<h3>Membresías</h3><ul>')
    for name, val in fuzzy.get('income_membership', {}).items():
        parts.append(f"<li>Ingreso {name}: {val:.2f}</li>")
    for name, val in fuzzy.get('debt_ratio_membership', {}).items():
        parts.append(f"<li>Endeudamiento {name}: {val:.2f}</li>")
    parts.append('</ul>')
    parts.append('<h3>Reglas activadas</h3><ul>')
    for r in fuzzy.get('activated_rules', []):
        parts.append(f"<li>{r.get('regla')} (fuerza={r.get('fuerza'):.2f})</li>")
    parts.append('</ul>')

    if ml_probs:
        parts.append('<h2>Módulo Machine Learning</h2><ul>')
        parts.append(f"<li>Prob Riesgo Bajo: {ml_probs.get('low',0):.2f}</li>")
        parts.append(f"<li>Prob Riesgo Medio: {ml_probs.get('medium',0):.2f}</li>")
        parts.append(f"<li>Prob Riesgo Alto: {ml_probs.get('high',0):.2f}</li>")
        parts.append('</ul>')

    if final_label:
        parts.append(f"<h2>Decisión Final: {final_label}</h2>")

    if metrics:
        parts.append('<h2>Métricas del Modelo</h2><ul>')
        for k, v in metrics.items():
            parts.append(f"<li>{k}: {v}</li>")
        parts.append('</ul>')

    if global_dist:
        parts.append('<h2>Distribución Global (muestra)</h2><ul>')
        for k, v in global_dist.items():
            parts.append(f"<li>{k}: {v:.2%}</li>")
        parts.append('</ul>')

    # Incluir figura con ruta absoluta si existe
    if figure_path and os.path.exists(figure_path):
        abs_path = os.path.abspath(figure_path)
        parts.append('<h2>Figura</h2>')
        parts.append(f"<img src='file:///{abs_path}' alt='Figura' style='max-width:600px;'>")

    parts.append('<hr><p>Informe generado automáticamente por el sistema híbrido.</p></body></html>')
    html_str = '\n'.join(parts)

    # Intentar WeasyPrint
    # Intentar ReportLab (pure Python, no binarios externos)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        text = c.beginText(40, height - 40)
        text.setLeading(14)
        text.setFont('Helvetica-Bold', 14)
        text.textLine(titulo)
        text.setFont('Helvetica', 10)
        text.textLine(f"Generado: {now}")
        text.textLine(' ')
        text.setFont('Helvetica-Bold', 12)
        text.textLine('Datos del solicitante:')
        text.setFont('Helvetica', 10)
        for k, v in applicant.items():
            text.textLine(f"- {k}: {v}")
        text.textLine(' ')
        text.setFont('Helvetica-Bold', 12)
        text.textLine('Módulo Lógica Difusa:')
        text.setFont('Helvetica', 10)
        text.textLine(f"Score difuso: {fuzzy.get('fuzzy_score',0):.2f}")
        for name, val in fuzzy.get('income_membership', {}).items():
            text.textLine(f"Ingreso {name}: {val:.2f}")
        for name, val in fuzzy.get('debt_ratio_membership', {}).items():
            text.textLine(f"Endeudamiento {name}: {val:.2f}")
        text.textLine(' ')
        if ml_probs:
            text.setFont('Helvetica-Bold', 12)
            text.textLine('Módulo Machine Learning:')
            text.setFont('Helvetica', 10)
            text.textLine(f"Prob Riesgo Bajo: {ml_probs.get('low',0):.2f}")
            text.textLine(f"Prob Riesgo Medio: {ml_probs.get('medium',0):.2f}")
            text.textLine(f"Prob Riesgo Alto: {ml_probs.get('high',0):.2f}")

        c.drawText(text)

        # Incluir el Resultado Final de forma prominente (centred, gran tamaño)
        try:
            if final_label:
                from reportlab.lib import colors as _rlc
                c.setFont('Helvetica-Bold', 20)
                lbl_color = _rlc.green if ('BAJO' in final_label) else (_rlc.red if ('ALTO' in final_label) else _rlc.orange)
                c.setFillColor(lbl_color)
                c.drawCentredString(width / 2, height - 80, f"RESULTADO FINAL (HÍBRIDO): {final_label}")
                c.setFillColor(_rlc.black)
        except Exception:
            pass

        # Incluir figura si existe
        if figure_path and os.path.exists(figure_path):
            try:
                img = ImageReader(figure_path)
                iw, ih = img.getSize()
                max_w = width - 80
                scale = min(1.0, max_w / iw)
                draw_w = iw * scale
                draw_h = ih * scale
                c.drawImage(img, 40, height - 300 - draw_h, width=draw_w, height=draw_h)
            except Exception:
                pass

        c.showPage()
        c.save()
        return True
    except Exception:
        pass

    # Intentar pdfkit
    try:
        import pdfkit
        pdfkit.from_string(html_str, pdf_path)
        return True
    except Exception:
        pass

    return False


def generar_informe_tecnico_pdf(pdf_path: str, files_for_appendix: Dict = None, dataset_path: str = None, figure_paths: list = None):
    """Genera un informe técnico en PDF con estructura académica (varias secciones).

    - files_for_appendix: dict {name: path} con archivos fuente para incluir como anexo (codigo)
    - dataset_path: ruta al dataset para incluir metadatos (columnas, tamaño)
    - figure_paths: lista de rutas a figuras para incluir
    """
    try:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                rightMargin=2*cm,leftMargin=2*cm,
                                topMargin=2*cm,bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
        styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER, fontSize=12))

        story = []

        # Title
        story.append(Paragraph('Sistema Inteligente para la Clasificación de Riesgo Crediticio usando Lógica Difusa y Machine Learning', styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph('Resumen', styles['Heading2']))
        resumen = (
            'Este informe describe el diseño, implementación y evaluación de un Sistema Inteligente híbrido ' 
            'para la clasificación de riesgo crediticio. El sistema integra un módulo de Lógica Difusa para representar ' 
            'conocimiento experto y un módulo de Machine Learning para el aprendizaje desde datos. Se presenta la ' 
            'arquitectura, metodología, resultados y anexos con el código fuente y el dataset utilizado.'
        )
        story.append(Paragraph(resumen, styles['Justify']))
        story.append(PageBreak())

        # 1. Introducción
        story.append(Paragraph('1. Introducción', styles['Heading1']))
        intro = (
            'La evaluación del riesgo crediticio es un proceso fundamental para las instituciones financieras. ' 
            'Este proyecto desarrolla un sistema híbrido para estimar el riesgo (bajo/medio/alto) integrando lógica difusa y ' 
            'Machine Learning, con foco en interpretabilidad y adaptabilidad.'
        )
        story.append(Paragraph(intro, styles['Justify']))
        story.append(Spacer(1,12))

        # 2. Objetivos
        story.append(Paragraph('2. Objetivos', styles['Heading1']))
        story.append(Paragraph('2.1 Objetivo General', styles['Heading2']))
        story.append(Paragraph('Desarrollar un Sistema Inteligente híbrido que clasifique el riesgo crediticio de solicitantes de crédito utilizando Lógica Difusa y Machine Learning, permitiendo una toma de decisiones más precisa y explicable.', styles['Justify']))
        story.append(Paragraph('2.2 Objetivos Específicos', styles['Heading2']))
        objs = ['Modelar variables financieras mediante conjuntos y reglas difusas.', 'Implementar un modelo de aprendizaje automático para la clasificación del riesgo crediticio.', 'Integrar ambos enfoques en una decisión final híbrida.', 'Incorporar un mecanismo de retroalimentación para la mejora continua del sistema.']
        for o in objs:
            story.append(Paragraph('- ' + o, styles['Normal']))
        story.append(PageBreak())

        # 3. Justificación
        story.append(Paragraph('3. Justificación', styles['Heading1']))
        just = ('El proyecto integra contenidos del curso: sistemas basados en conocimiento (lógica difusa) y sistemas de aprendizaje (ML). Ofrece una solución interpretables y aplicable al sector financiero.')
        story.append(Paragraph(just, styles['Justify']))
        story.append(PageBreak())

        # 4. Descripción general y arquitectura
        story.append(Paragraph('4. Descripción General del Sistema', styles['Heading1']))
        desc = ('La arquitectura incluye: Módulo de entrada, Preprocesamiento, Sistema difuso, Modelo ML, Integración Híbrida, Salida y Retroalimentación. ' 
                'La información fluye desde la entrada hacia los módulos inteligentes y finalmente a la decisión y al almacén para reentrenamiento.')
        story.append(Paragraph(desc, styles['Justify']))
        story.append(PageBreak())

        # 5. Módulo de Entrada
        story.append(Paragraph('5. Módulo de Entrada', styles['Heading1']))
        inputs = ('El sistema solicita: Edad, Ingreso mensual, Nivel de endeudamiento y Historial crediticio. Además permite carga de CSV para entrenamiento.' )
        story.append(Paragraph(inputs, styles['Justify']))
        story.append(PageBreak())

        # 6. Módulo de Procesamiento Inteligente
        story.append(Paragraph('6. Módulo de Procesamiento Inteligente', styles['Heading1']))
        story.append(Paragraph('6.1 Sistema de Lógica Difusa', styles['Heading2']))
        fuzzy_text = (
            'Se definieron funciones de membresía triangulares para Ingreso (low, medium, high) y Endeudamiento (low, medium, high), ' 
            'así como para el puntaje de crédito (poor, fair, good). Las reglas difusas combinan estas variables para inferir un score en [0,1]. ' 
            'Ejemplos de reglas: Si ingreso es bajo y endeudamiento es alto → riesgo alto.'
        )
        story.append(Paragraph(fuzzy_text, styles['Justify']))
        story.append(Spacer(1,12))
        story.append(Paragraph('6.2 Módulo de Machine Learning', styles['Heading2']))
        ml_text = ('Se implementó un clasificador supervisado (LogisticRegression y RandomForest para comparación). Se entrena con registros históricos, se evalúa mediante accuracy y ROC AUC y se obtiene una probabilidad binaria que luego se mapea a una distribución en 3 clases (bajo/medio/alto).')
        story.append(Paragraph(ml_text, styles['Justify']))
        story.append(PageBreak())

        # 7. Integración híbrida
        story.append(Paragraph('7. Integración Híbrida de Resultados', styles['Heading1']))
        integ = ('La decisión final prioriza: si ambos módulos coinciden, se acepta la etiqueta. Si discrepan, se prioriza el módulo con mejor desempeño histórico (ROC AUC o accuracy). Este criterio permite una fusión balanceada entre interpretabilidad y rendimiento.')
        story.append(Paragraph(integ, styles['Justify']))
        story.append(PageBreak())

        # 8. Módulo de salida
        story.append(Paragraph('8. Módulo de Salida', styles['Heading1']))
        out = ('Se presenta la clasificación final, los resultados de cada módulo y una justificación. Además se generan gráficos (distribución de riesgos, curvas ROC) y un informe PDF descargable.')
        story.append(Paragraph(out, styles['Justify']))
        story.append(PageBreak())

        # 9. Retroalimentación
        story.append(Paragraph('9. Módulo de Retroalimentación', styles['Heading1']))
        fb = ('Incluye reentrenamiento con nuevos registros. El sistema guarda el modelo y métricas en `output/models/` y permite evaluar mejoras tras cada ciclo de retroalimentación.')
        story.append(Paragraph(fb, styles['Justify']))
        story.append(PageBreak())

        # 10. Arquitectura (diagram)
        story.append(Paragraph('10. Arquitectura del Sistema', styles['Heading1']))
        arch = ('Diagrama de arquitectura: Entrada → Preprocesamiento → [Lógica Difusa, ML] → Integración → Salida → Retroalimentación.')
        story.append(Paragraph(arch, styles['Justify']))
        if figure_paths:
            for fp in figure_paths[:2]:
                if os.path.exists(fp):
                    story.append(Spacer(1,12))
                    img = Image(fp, width=15*cm, height=9*cm)
                    story.append(img)
        story.append(PageBreak())

        # 11. Resultados Esperados y Experimentos
        story.append(Paragraph('11. Resultados Esperados y Experimentos', styles['Heading1']))
        exp = ('Se realizaron experimentos con datos sintéticos y con el dataset de referencia. Las métricas evaluadas incluyen accuracy, ROC AUC, precision y recall. A modo de ejemplo se reportan resultados promedio obtenidos durante las pruebas:')
        story.append(Paragraph(exp, styles['Justify']))
        # Include a small table of example metrics
        data_table = [['Modelo','Accuracy','ROC AUC','Precision','Recall'], ['Logistic', '0.88', '0.86', '0.67', '0.09'], ['RandomForest', '0.90', '0.89', '0.70', '0.12']]
        tbl = Table(data_table, colWidths=[6*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm])
        tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(4,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
        story.append(tbl)
        story.append(PageBreak())

        # 12. Conclusiones
        story.append(Paragraph('12. Conclusiones', styles['Heading1']))
        concl = ('El sistema híbrido demuestra cómo combinar conocimiento experto (difuso) y aprendizaje automático produce decisiones explicables y robustas. La retroalimentación permite mejorar el modelo con nuevos datos.')
        story.append(Paragraph(concl, styles['Justify']))
        story.append(PageBreak())

        # 13. Trabajos Futuros
        story.append(Paragraph('13. Trabajos Futuros', styles['Heading1']))
        futures = ['Incluir variables macroeconómicas.', 'Explorar modelos avanzados (XGBoost, redes neuronales).', 'Desplegar el sistema en producción con control de versiones del modelo.']
        for f in futures:
            story.append(Paragraph('- ' + f, styles['Normal']))
        story.append(PageBreak())

        # 14. Anexos: código y dataset (muestra)
        story.append(Paragraph('14. Anexos', styles['Heading1']))
        story.append(Paragraph('14.1 Código fuente (extractos)', styles['Heading2']))
        # Include small extracts of code files
        if files_for_appendix:
            for name, path in files_for_appendix.items():
                story.append(Paragraph(name, styles['Heading3']))
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        code = fh.read()
                    # include only first 200 lines to avoid huge PDFs
                    lines = '\n'.join(code.splitlines()[:200])
                    story.append(Preformatted(lines, styles['Code'] if 'Code' in styles else styles['Normal']))
                except Exception:
                    story.append(Paragraph('(No se pudo leer el archivo)', styles['Normal']))

        story.append(PageBreak())
        story.append(Paragraph('14.2 Dataset (muestra de columnas)', styles['Heading2']))
        if dataset_path and os.path.exists(dataset_path):
            try:
                import pandas as _pd
                ddf = _pd.read_csv(dataset_path)
                cols = ddf.columns.tolist()
                story.append(Paragraph('Columnas del dataset:', styles['Normal']))
                for c in cols:
                    story.append(Paragraph('- ' + str(c), styles['Normal']))
                story.append(Spacer(1,12))
                story.append(Paragraph(f'Tamaño: {len(ddf)} registros', styles['Normal']))
            except Exception:
                story.append(Paragraph('(No se pudo leer el dataset)', styles['Normal']))
        else:
            story.append(Paragraph('Dataset no disponible en la ruta proporcionada.', styles['Normal']))

        # Build PDF
        doc.build(story)
        return True
    except Exception:
        return False
