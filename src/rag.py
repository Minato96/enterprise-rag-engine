from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()


# CONFIGURATION
DB_FAISS_PATH = "data/vectors/faiss_index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
#LLM_MODEL = "tinyllama" # Use "llama3" if you pulled that instead

def get_rag_chain():
    # 1. Load the Vector DB
    # We need the same embedding model to "understand" the stored vectors
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True # Needed for local files
    )
    
    # 2. Setup the Retriever
    # "k=3" means "Find me the top 3 most relevant chunks"
    retriever = db.as_retriever(search_kwargs={'k': 3})

    os.environ["GROQ_API_KEY"] = os.getenv("API_KEY")
    # 3. Setup the LLM (Local Ollama)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1 # Low temp = Factual. High temp = Creative. We want Factual.
    )

    # 4. Define the Prompt Template (Crucial for avoiding hallucinations)
    template = """
    You are a strict cv reviewer to select interns. Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    
    PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    # 5. Create the Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "Stuff" all chunks into the prompt
        retriever=retriever,
        return_source_documents=True, # Critical for showing citations
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

if __name__ == "__main__":
    print("--- Initializing RAG Chain ---")
    chain = get_rag_chain()
    
    # Interactive Loop
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
            
        print("Thinking...")
        
        # Run the query
        response = chain.invoke({"query": query})
        
        print(f"\nAnswer: {response['result']}")
        print("\n--- Sources ---")
        for source in response['source_documents']:
            # Show which file and page the info came from
            print(f"- {source.metadata.get('source', 'Unknown')} (Content preview: {source.page_content[:50]}...)")