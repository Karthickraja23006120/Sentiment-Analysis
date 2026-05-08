import pypdf

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"--- Page {i+1} ---\n"
            text += page.extract_text() + "\n"
    return text

if __name__ == "__main__":
    pdf_path = "NLP_SentimentAnalysis.pdf"
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(extract_text_from_pdf(pdf_path))
