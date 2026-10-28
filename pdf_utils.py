from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

def create_analysis_pdf(session_id: int, filename: str, jump_type: str, 
                        confidence: float, rotations: float, all_probs: dict) -> io.BytesIO:
    """Генерирует PDF-отчёт и возвращает его как байтовый поток."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    elements = []
    styles = getSampleStyleSheet()

    # Заголовок
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=20, textColor=colors.HexColor('#1a5490'),
        spaceAfter=20, alignment=1  # 1 = по центру
    )
    elements.append(Paragraph("Skate Jump Analysis Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Основная информация
    info_data = [
        ["Session ID", str(session_id)],
        ["File", filename],
        ["Jump Type", jump_type],
        ["Confidence", f"{confidence:.2%}"],
        ["Rotations", f"{rotations:.2f}"],
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Таблица вероятностей
    elements.append(Paragraph("Class Probabilities", styles['Heading2']))
    elements.append(Spacer(1, 0.1 * inch))

    prob_data = [["Class", "Probability"]]
    for cls, prob in all_probs.items():
        prob_data.append([cls, f"{prob:.4f}"])

    prob_table = Table(prob_data, colWidths=[2*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a7c9b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(prob_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer