import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(pdf_path):
    print(f"Loading {pdf_path}...")
    
    # "elements" mode breaks the PDF into Title, NarrativeText, Table, etc.
    loader = UnstructuredPDFLoader(
        pdf_path,
        mode="elements", 
        strategy="fast", # Change to 'hi_res' later for complex tables (slower)
    )
    docs = loader.load()
    
    print(f"Initial raw elements found: {len(docs)}")

    # We need to merge these elements into decent sized chunks
    # This splitter respects sentence structure
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    
    return chunks

if __name__ == "__main__":
    # TEST: Put a dummy PDF in data/raw/test.pdf to run this
    test_pdf = "data/raw/test.pdf" 
    
    # Simple check to see if file exists
    if os.path.exists(test_pdf):
        final_chunks = load_and_chunk_pdf(test_pdf)
        
        print(f"\n--- Processing Complete ---")
        print(f"Total Chunks created: {len(final_chunks)}")
        
        # Print first 2 chunks to check quality
        print("\n--- Preview Chunk 1 ---")
        print(final_chunks[0].page_content)
        print(f"Source: {final_chunks[0].metadata}")
    else:
        print(f"Please place a PDF at {test_pdf} to test.")