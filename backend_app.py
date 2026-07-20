import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class RAG:

    def __init__(self):

        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.vector_db_path = os.path.join(self.base_dir, "vector_db")

    # -----------------------------
    # Load PDF
    # -----------------------------
    def load_pdf(self, pdf_path):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        return chunks

    # -----------------------------
    # Create Vector Database
    # -----------------------------
    def create_vector_database(self, folder_path):

        if not os.path.exists(folder_path):
            raise FileNotFoundError(
                f"Documents folder not found:\n{folder_path}"
            )

        all_chunks = []

        pdf_files = [
            file for file in os.listdir(folder_path)
            if file.endswith(".pdf")
        ]

        if len(pdf_files) == 0:
            raise Exception("No PDF files found inside documents folder.")

        print(f"\nFound {len(pdf_files)} PDF(s)\n")

        for file in pdf_files:

            pdf_path = os.path.join(folder_path, file)

            print("Loading:", file)

            chunks = self.load_pdf(pdf_path)

            all_chunks.extend(chunks)

        print(f"\nTotal Chunks Created : {len(all_chunks)}")

        db = FAISS.from_documents(
            all_chunks,
            self.embedding
        )

        db.save_local(self.vector_db_path)

        print("\nVector Database Saved Successfully")
        print("Location:", self.vector_db_path)

    # -----------------------------
    # Load Vector Database
    # -----------------------------
    def load_vector_database(self):

        if not os.path.exists(self.vector_db_path):
            raise Exception(
                "Vector database not found. Run rag.py first."
            )

        db = FAISS.load_local(
            self.vector_db_path,
            self.embedding,
            allow_dangerous_deserialization=True
        )

        return db

    # -----------------------------
    # Retrieve Context
    # -----------------------------
    def retrieve(self, question, k=3):

        db = self.load_vector_database()

        retriever = db.as_retriever(
            search_kwargs={"k": k}
        )

        docs = retriever.invoke(question)

        context = ""

        for doc in docs:

            context += doc.page_content
            context += "\n\n"

        return context


# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":

    rag = RAG()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DOCS_PATH = os.path.join(BASE_DIR, "documents")

    print("\nDocuments Folder")
    print(DOCS_PATH)

    rag.create_vector_database(DOCS_PATH)

    print("\nTesting Retrieval...\n")

    question = input("Enter Question : ")

    context = rag.retrieve(question)

    print("\nRetrieved Context\n")
    print("=" * 70)
    print(context)