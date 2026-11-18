import re, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
src, dst = (sys.argv + ["RESEARCH_NOTE.md", "RESEARCH_NOTE.pdf"][len(sys.argv) - 1:])[:2] if len(sys.argv) > 2 else ("RESEARCH_NOTE.md", "RESEARCH_NOTE.pdf")
ss = getSampleStyleSheet()
def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t); t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    return re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', t)
story, rows = [], []
def flush():
    global rows
    if rows:
        t = Table([[Paragraph(inline(c), ss["BodyText"]) for c in r] for r in rows], repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        story.extend([t, Spacer(1, 8)]); rows = []
for line in open(src, encoding="utf-8").read().splitlines():
    if line.startswith("|"):
        c = [x.strip() for x in line.strip("|").split("|")]
        if not set("".join(c)) <= set("-: "): rows.append(c)
        continue
    flush()
    if not line.strip(): story.append(Spacer(1, 5))
    elif line.startswith("# "): story.append(Paragraph(inline(line[2:]), ss["Title"]))
    elif line.startswith("## "): story.append(Paragraph(inline(line[3:]), ss["Heading2"]))
    elif re.match(r"^(- |\d+\. )", line): story.append(Paragraph(inline(re.sub(r"^(- |\d+\. )", "• ", line)), ss["BodyText"], bulletText=None))
    else: story.append(Paragraph(inline(line), ss["BodyText"]))
flush()
SimpleDocTemplate(dst, pagesize=A4, leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48).build(story)
print("wrote", dst)
