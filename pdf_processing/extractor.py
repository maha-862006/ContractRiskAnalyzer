import fitz


def extract_text_from_pdf(uploaded_file):

    uploaded_file.seek(0)

    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    extracted_text = ""

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        extracted_text += page.get_text("text")

    pdf_document.close()

    return extracted_text