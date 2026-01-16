from dataclasses import dataclass, asdict
from typing import Dict
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
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
        # 1. Judge LLM
        groq_chat = ChatGroq(
            temperature=0,
            model_name=Config.MODEL_NAME,
            api_key=Config.GROQ_API_KEY
        )
        self.llm = LangchainLLMWrapper(groq_chat)
        
        # 2. Embeddings
        fast_embed = FastEmbedEmbeddings(model_name=Config.EMBEDDING_MODEL)
        # Fix for Ragas validation
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

        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False 
        )

        def safe_get(key):
            val = results.get(key, 0.0)
            if isinstance(val, list): return val[0] if val else 0.0
            return float(val) if val == val else 0.0

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