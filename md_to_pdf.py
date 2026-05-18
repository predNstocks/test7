"""
Convert markdown report to PDF.
Converts LaTeX math to MathML (weasyprint native) and renders via weasyprint.
"""
import sys, re
import markdown
from weasyprint import HTML


def _convert_math_to_mathml(text):
    """Convert \(...\) inline and $$...$$ display LaTeX to MathML."""
    from latex2mathml.converter import convert as latex2mathml_convert

    def _replace_display(m):
        latex = m.group(1).strip()
        try:
            mathml = latex2mathml_convert(latex)
            mathml = mathml.replace('<math', '<math display="block"', 1)
            return mathml
        except Exception:
            return m.group(0)

    def _replace_inline(m):
        latex = m.group(1).strip()
        try:
            return latex2mathml_convert(latex)
        except Exception:
            return m.group(0)

    text = re.sub(r'\$\$(.+?)\$\$', _replace_display, text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', _replace_inline, text)
    return text


def md_to_pdf(md_path, pdf_path):
    with open(md_path) as f:
        md_content = f.read()

    md_content = _convert_math_to_mathml(md_content)

    html_body = markdown.markdown(md_content, extensions=['tables'])

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; line-height: 1.6; color: #1a1a1a; }}
h1 {{ font-size: 22px; border-bottom: 2px solid #2563eb; padding-bottom: 8px; }}
h2 {{ font-size: 18px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; margin-top: 24px; }}
h3 {{ font-size: 15px; color: #4b5563; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12px; }}
th {{ background: #f3f4f6; border: 1px solid #d1d5db; padding: 6px 10px; text-align: left; font-weight: 600; }}
td {{ border: 1px solid #d1d5db; padding: 4px 10px; }}
tr:nth-child(even) {{ background: #f9fafb; }}
strong {{ color: #1e40af; }}
img {{ max-width: 100%; }}
math {{ font-size: 1em; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(pdf_path)
    print(f'PDF created: {pdf_path}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 md_to_pdf.py <input.md> <output.pdf>')
        sys.exit(1)
    md_to_pdf(sys.argv[1], sys.argv[2])
