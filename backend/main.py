from fastapi import FastAPI
from backend.routers import chat_router, data_router, news_router

# Initialize FastAPI app
app = FastAPI(
    title="FinSage AI",
    version="1.0",
    description="An intelligent financial chatbot that provides insights on stocks, mutual funds, and personal finance."
)

# Include Routers
app.include_router(chat_router.router, prefix="/chat", tags=["Chatbot"])
app.include_router(data_router.router, prefix="/data", tags=["Data"])
app.include_router(news_router.router, prefix="/news", tags=["News"])

# Root endpoint
@app.get("/")
def root():
    return {"message": "FinSage AI Backend is running successfully!"}
