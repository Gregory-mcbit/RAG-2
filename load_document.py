from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


DATA_PATH = "data"


def load_documents():
    document = PyPDFDirectoryLoader(DATA_PATH)
    return document.load()


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )

    return splitter.split_documents(documents=documents)


# doc = load_documents()
# chunks = split_documents(doc)
# print(chunks[100])
