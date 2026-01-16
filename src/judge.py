from dataclasses import dataclass, asdict
from typing import Dict
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from .config import Config

@dataclass
class JudgeResult:
    score: float
    is_remedial_needed: bool
    feedback: str
    metrics: Dict[str, float]

class RagasJudge:
    def __init__(self):
        # 1. Judge LLM (Gemini)
        # We use a temperature of 0 for consistent grading
        google_chat = ChatGoogleGenerativeAI(
            temperature=0,
            model=Config.MODEL_NAME,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.llm = LangchainLLMWrapper(google_chat)
        
        # 2. Embeddings (Local - FastEmbed)
        # We keep this local to avoid sending too many requests and to keep it fast.
        fast_embed = FastEmbedEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.embeddings = LangchainEmbeddingsWrapper(fast_embed)
        self.embeddings.model = Config.EMBEDDING_MODEL 

        self.metrics = [Faithfulness(), AnswerRelevancy(), AnswerCorrectness()]

    def evaluate_answer(self, question: str, user_answer: str, context: str, criteria: str) -> JudgeResult:
        data = {
            "question": [question],
            "answer": [user_answer],
            "contexts": [[context]],
            "ground_truth": [criteria]
        }
        dataset = Dataset.from_dict(data)

        # Ragas will now use Gemini to evaluate the answer
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False 
        )

        def safe_get(key):
            try:
                val = results[key]
            except KeyError:
                return 0.0
            
            if isinstance(val, list):
                if len(val) > 0:
                    val = val[0]
                else:
                    return 0.0
            
            try:
                f_val = float(val)
                return f_val if f_val == f_val else 0.0
            except (ValueError, TypeError):
                return 0.0

        f_score = safe_get("faithfulness")
        r_score = safe_get("answer_relevancy")
        c_score = safe_get("answer_correctness")

        final_score = (c_score * 0.5) + (f_score * 0.3) + (r_score * 0.2)
        scaled_score = round(final_score * 10, 1)
        
        is_remedial = (scaled_score < 6.0) or (f_score < 0.4)
        feedback = f"Score: {scaled_score}/10 | F: {f_score:.2f}, R: {r_score:.2f}, C: {c_score:.2f}"

        return JudgeResult(
            score=scaled_score,
            is_remedial_needed=is_remedial,
            feedback=feedback,
            metrics={"faithfulness": f_score, "relevancy": r_score, "correctness": c_score}
        )