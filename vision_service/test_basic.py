from app.document_images import SUPPORTED_DOCUMENT_EXTENSIONS


def test_supported_documents():
    assert ".pdf" in SUPPORTED_DOCUMENT_EXTENSIONS
    assert ".docx" in SUPPORTED_DOCUMENT_EXTENSIONS
    assert ".xlsx" in SUPPORTED_DOCUMENT_EXTENSIONS
