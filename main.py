import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.rag import get_rag_chain
from src.vector_store import create_vector_db

# --- Global State ---
# We keep the chain in a global variable so it stays loaded in memory
app_state = {"chain": None}

# --- 1. Lifespan Manager (Startup/Shutdown Logic) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up: Loading RAG Chain...")
    try:
        # Load the model ONCE when server starts
        app_state["chain"] = get_rag_chain()
        print("✅ RAG Chain loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load chain (maybe no DB yet?). Error: {e}")
    
    yield # The app runs here
    
    print("🛑 Shutting down: Cleaning up resources...")
    app_state["chain"] = None

# --- Initialize App ---
app = FastAPI(title="Enterprise RAG Engine", lifespan=lifespan)

# --- 2. Pydantic Models (Validation) ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# --- 3. Endpoints ---

@app.get("/")
async def root():
    return {"message": "RAG Engine is Online. Go to /docs to test."}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Send a question to the RAG model.
    """
    if not app_state["chain"]:
        raise HTTPException(status_code=503, detail="RAG Chain not initialized. Please upload a PDF first.")
    
    # Invoke the chain (same as your terminal script)
    result = app_state["chain"].invoke({"query": request.query})
    
    # Format the sources cleanly
    sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
    
    return {
        "answer": result["result"],
        "sources": list(set(sources)) # Remove duplicates
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and rebuild the vector database.
    """
    # 1. Save the file
    file_location = f"data/raw/{file.filename}"
    os.makedirs("data/raw", exist_ok=True)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Trigger Ingestion (This rebuilds the FAISS index)
    # In a real (Level 2) app, this would be a background task (Celery)
    try:
        create_vector_db(file_location)
        
        # 3. Reload the chain to pick up new data
        app_state["chain"] = get_rag_chain()
        
        return {"message": f"Successfully processed {file.filename}. You can now chat!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# --- Run Server ---
if __name__ == "__main__":

    # Host 0.0.0.0 makes it accessible on your local network
    uvicorn.run(app, host="0.0.0.0", port=8000)