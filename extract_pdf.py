import pypdf

reader = pypdf.PdfReader('MTechPESOct24AGrp2.pdf')
text = ""
for i, page in enumerate(reader.pages):
    text += f"--- Page {i+1} ---\n"
    text += page.extract_text() + "\n\n"

with open('pdf_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
