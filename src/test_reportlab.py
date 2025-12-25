from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

c = canvas.Canvas('output/reports/reportlab_test.pdf', pagesize=A4)
width, height = A4
c.drawString(100, height-100, 'Reportlab test OK')
c.save()
print('wrote reportlab_test.pdf')
