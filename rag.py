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

    # Load PDF
    def load_pdf(self, pdf_path):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        return chunks

    # Create Vector Database
    def create_vector_database(self, pdf_path):

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found:\n{pdf_path}")

        print("Loading PDF...")
        print(pdf_path)

        chunks = self.load_pdf(pdf_path)

        print(f"Total Chunks: {len(chunks)}")

        db = FAISS.from_documents(
            chunks,
            self.embedding
        )

        db.save_local(self.vector_db_path)

        print("\nVector Database Created Successfully!")
        print("Saved at:", self.vector_db_path)

    # Load Vector Database
    def load_vector_database(self):

        db = FAISS.load_local(
            self.vector_db_path,
            self.embedding,
            allow_dangerous_deserialization=True
        )

        return db

    # Retrieve Context
    def retrieve(self, question, k=3):

        db = self.load_vector_database()

        retriever = db.as_retriever(
            search_kwargs={"k": k}
        )

        docs = retriever.invoke(question)

        context = ""

        for i, doc in enumerate(docs, start=1):

            context += f"\n----- Document {i} -----\n"
            context += doc.page_content
            context += "\n"

        return context


if __name__ == "__main__":

    rag = RAG()

    # CHANGE THIS TO YOUR PDF PATH
    PDF_PATH = r"C:\Users\rushi\Downloads\AI_Notes.pdf"

    rag.create_vector_database(PDF_PATH)

    while True:

        question = input("\nAsk Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        context = rag.retrieve(question)

        print("\nRetrieved Context")
        print("=" * 80)
        print(context)