import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    @staticmethod
    def validate():
        if not Config.GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in .env file")