from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup


def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def load_docx(file):
    doc = Document(file)
    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


def load_txt(file):
    return file.read().decode("utf-8")


def load_html(file):
    html = file.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def load_document(uploaded_file):
    extension = uploaded_file.name.split(".")[-1].lower()

    if extension == "pdf":
        return load_pdf(uploaded_file)

    if extension == "docx":
        return load_docx(uploaded_file)

    if extension == "txt":
        return load_txt(uploaded_file)

    if extension == "html":
        return load_html(uploaded_file)

    raise ValueError("Unsupported file format")