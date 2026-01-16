import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # UPDATED: Use the Gemini 2.5 Flash stable model
    # Released June 2025, optimized for speed and agentic tasks
    MODEL_NAME = "gemini-2.5-flash"
    
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    @staticmethod
    def validate():
        if not Config.GOOGLE_API_KEY:
            raise ValueError("Missing GOOGLE_API_KEY in .env file")