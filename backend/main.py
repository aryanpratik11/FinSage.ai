from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import chat_router #, data_router, news_router

# Initialize FastAPI app
app = FastAPI(
    title="FinSage AI",
    version="1.0",
    description=(
        "FinSage AI — An intelligent agentic chatbot that provides "
        "data-backed insights on stocks, mutual funds, and personal finance."
    ),
)

# CORS Middleware — Allow frontend (React) to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat_router, prefix="/chat", tags=["Chatbot"])
#app.include_router(data_router.router, prefix="/data", tags=["Market Data"])
#app.include_router(news_router.router, prefix="/news", tags=["Financial News"])

# Root endpoint
@app.get("/")
async def root():
    print("FinSage.ai Backend is running successfully!")
    return {"message": "FinSage.ai Backend is running successfully!"}

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy"}
