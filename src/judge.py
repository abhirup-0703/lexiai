from dataclasses import dataclass
from typing import Dict, List, Optional
from ragas import evaluate, RunConfig
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from .config import Config

@dataclass
class JudgeResult:
    score: float
    is_remedial_needed: bool
    feedback: str
    metrics: Dict[str, float]

class RagasFriendlyEmbeddingsWrapper(Embeddings):
    """
    Wraps an embedding model to explicitly expose 'model' as a string.
    This prevents Ragas from reading the underlying object and crashing.
    """
    def __init__(self, internal_embeddings, model_name: str):
        self.internal = internal_embeddings
        self.model = model_name 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.internal.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.internal.embed_query(text)

class RagasJudge:
    def __init__(self):
        # 1. Judge LLM (Gemini)
        self.google_chat = ChatGoogleGenerativeAI(
            temperature=0,
            model=Config.MODEL_NAME,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.llm = LangchainLLMWrapper(self.google_chat)
        
        # 2. Embeddings (Local - FastEmbed)
        print(f"Loading embeddings: {Config.EMBEDDING_MODEL}...")
        fast_embed = FastEmbedEmbeddings(model_name=Config.EMBEDDING_MODEL)
        safe_embed = RagasFriendlyEmbeddingsWrapper(fast_embed, Config.EMBEDDING_MODEL)
        self.embeddings = LangchainEmbeddingsWrapper(safe_embed)

        self.metrics = [Faithfulness(), AnswerRelevancy(), AnswerCorrectness()]

    def _generate_qualitative_feedback(self, question, answer, context, rubric, exemplar=None) -> str:
        """Generates specific text feedback using the LLM directly."""
        prompt = f"""
        You are a strict professor grading an oral exam.
        
        Question: {question}
        Rubric/Criteria: {rubric}
        Context Info: {context}
        """
        
        if exemplar:
            prompt += f"\nIdeal Exemplar Answer: {exemplar}\n"
            
        prompt += f"\nStudent Answer: {answer}\n"
        
        prompt += """
        Task: Provide a concise (1-2 sentences) critique of the answer. 
        - If correct, acknowledge the specific good points.
        - If incorrect, explain EXACTLY why (e.g., "You missed the concept of X" or "Your reasoning on Y is flawed").
        - Do not give a score here, just the qualitative analysis.
        """
        response = self.google_chat.invoke([HumanMessage(content=prompt)])
        return response.content

    def evaluate_answer(self, question: str, user_answer: str, context: str, criteria: str, exemplar: str = None) -> JudgeResult:
        # 1. Generate Qualitative Feedback
        qualitative_feedback = self._generate_qualitative_feedback(question, user_answer, context, criteria, exemplar)

        # 2. Calculate Ragas Metrics
        # Use exemplar as ground_truth if available, otherwise fallback to criteria
        ground_truth_text = exemplar if exemplar else criteria

        data = {
            "question": [question],
            "answer": [user_answer],
            "contexts": [[context]],
            "ground_truth": [ground_truth_text]
        }
        dataset = Dataset.from_dict(data)

        # Force serial execution to avoid rate limits
        run_config = RunConfig(max_workers=1, timeout=120)
        
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False,
            run_config=run_config
        )

        def safe_get(key):
            try:
                val = results[key]
                if isinstance(val, list) and val: val = val[0]
                f_val = float(val)
                return f_val if f_val == f_val else 0.0
            except:
                return 0.0

        f_score = safe_get("faithfulness")
        r_score = safe_get("answer_relevancy")
        c_score = safe_get("answer_correctness")

        final_score = (c_score * 0.5) + (f_score * 0.3) + (r_score * 0.2)
        scaled_score = round(final_score * 10, 1)
        
        is_remedial = (scaled_score < 6.0) or (f_score < 0.4)
        
        # Combine score and detailed feedback
        full_feedback = f"[{scaled_score}/10] {qualitative_feedback}"

        return JudgeResult(
            score=scaled_score,
            is_remedial_needed=is_remedial,
            feedback=full_feedback,
            metrics={"faithfulness": f_score, "relevancy": r_score, "correctness": c_score}
        )