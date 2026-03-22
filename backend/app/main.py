from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, search
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="local_storage"), name="static")

# Crucial for your React frontend (port 3000) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(search.router)

@app.get("/")
def read_root():
    return {"message": "DocuMind AI Backend is Running"}