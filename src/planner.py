import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from .config import Config
from .models import ExamPlan

class CognitivePlanner:
    def __init__(self):
        # Initialize Google Gemini via LangChain
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.2
        )

    def _clean_json_string(self, raw_string: str) -> str:
        # Standard cleanup for LLM markdown code blocks
        clean = re.sub(r"```json\s*", "", raw_string)
        clean = re.sub(r"```\s*$", "", clean)
        return clean.strip()

    def generate_exam_plan(self, context_text: str, enable_refinement: bool = True) -> ExamPlan:
        # Gemini 1.5 Flash has a 1 Million token context window, so we rarely need to truncate.
        # But keeping a safe limit (e.g., 100k chars) is good practice for speed.
        safe_context = context_text[:100000] 
        
        print("Generating Initial Exam Plan...")
        initial_plan = self._generate_initial_pass(safe_context)
        
        if not enable_refinement:
            return initial_plan

        print("Refining Questions...")
        return self._refine_plan(initial_plan, safe_context)

    def _generate_initial_pass(self, context: str) -> ExamPlan:
        json_skeleton = """
        {
          "topic": "Subject",
          "questions": [
            {
              "blooms_level": "Recall",
              "question": "Q",
              "context_snippet": "Quote",
              "grading_rubric": {"concepts": ["c1"], "grading_criteria": "crit"}
            }
          ]
        }
        """
        system_prompt = f"""
        You are an expert academic examiner. Analyze the text and generate a structured exam plan in STRICT JSON.
        Generate exactly 5 questions (Recall -> Evaluate).
        JSON Skeleton: {json_skeleton}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}")
        ]
        
        response = self.llm.invoke(messages)
        return self._parse_and_validate(response.content)

    def _refine_plan(self, initial_plan: ExamPlan, context: str) -> ExamPlan:
        current_json = initial_plan.model_dump_json()
        system_prompt = "You are a Quality Assurance Editor. Fix leakage and ambiguity. Output JSON."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original:\n{context}\n\nCurrent Plan:\n{current_json}")
        ]
        
        # Lower temp for refinement
        self.llm.temperature = 0.1
        response = self.llm.invoke(messages)
        return self._parse_and_validate(response.content)

    def _parse_and_validate(self, raw_json: str) -> ExamPlan:
        try:
            parsed = json.loads(self._clean_json_string(raw_json))
            # Handle nesting if LLM wraps it in a root key
            if "questions" not in parsed:
                for v in parsed.values():
                    if isinstance(v, dict) and "questions" in v:
                        parsed = v
                        break
            return ExamPlan(**parsed)
        except Exception as e:
            print(f"Validation Error: {e}")
            raise