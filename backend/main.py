from fastapi import FastAPI
from pydantic import BaseModel

from backend.core import run_llm
from backend.semantic_search import search_item
from backend.ingestion import ingest_docs
from backend.prediction import predict_rev, PredictionInput
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI()

# Allow all origins (not secure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    message: str

@app.get("/")
def index():
    return {
        "Hello senuth"
    }

@app.post("/vcDatabase")
def update_vector_database():
    res = ingest_docs()
    return {
        "Result":res
    }

# send query to the LLM
@app.post("/chat")
def chat(query: Query):
    user_message = query.message
    res = run_llm(query=user_message)

    return {
        "query": res["query"],
        "response": res["result"]
    }
#
@app.post('/predict')
def predict(data: PredictionInput):
    res =  predict_rev(data)

    return {
        "res": res
    }


@app.post('/semanticSearch')
def get_top_results(query: Query):
    res = search_item(query.message)
    return  res
