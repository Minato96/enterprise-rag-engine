import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_and_chunk_pdf  # Importing your function from Step 1

# CONFIGURATION
# We use a small, fast, but high-quality model
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DB_FAISS_PATH = "data/vectors/faiss_index"

def create_vector_db(pdf_path):
    # 1. Get the chunks (Logic from Step 1)
    chunks = load_and_chunk_pdf(pdf_path)
    if not chunks:
        return

    print(f"\n--- Creating Embeddings (Model: {MODEL_NAME}) ---")
    print("This may take a minute on the first run (downloading weights)...")

    # 2. Load the Embedding Model
    # We use 'cpu' (or 'cuda' if you have NVIDIA)
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True} # explicit normalization for cosine similarity
    )

    # 3. Create FAISS Vector Store
    # This actually sends text to the model -> gets numbers -> stores them
    db = FAISS.from_documents(chunks, embeddings)

    # 4. Save to Disk
    db.save_local(DB_FAISS_PATH)
    print(f"\n✅ Vector Database saved to: {DB_FAISS_PATH}")
    print(f"Total Vectors stored: {db.index.ntotal}")

if __name__ == "__main__":
    # Ensure the directory exists
    os.makedirs("data/vectors", exist_ok=True)
    
    # Run on your test PDF
    test_pdf = "data/raw/test.pdf"
    if os.path.exists(test_pdf):
        create_vector_db(test_pdf)
    else:
        print(f"Error: {test_pdf} not found.")