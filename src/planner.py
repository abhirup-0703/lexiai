import json
import re
from groq import Groq
from .config import Config
from .models import ExamPlan

class CognitivePlanner:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.MODEL_NAME

    def _clean_json_string(self, raw_string: str) -> str:
        clean = re.sub(r"```json\s*", "", raw_string)
        clean = re.sub(r"```\s*$", "", clean)
        return clean.strip()

    def generate_exam_plan(self, context_text: str, enable_refinement: bool = True) -> ExamPlan:
        # Truncate to avoid token limits if necessary
        safe_context = context_text[:25000] + ("\n...[TRUNCATED]" if len(context_text) > 25000 else "")
        
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
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2 
        )
        return self._parse_and_validate(response.choices[0].message.content)

    def _refine_plan(self, initial_plan: ExamPlan, context: str) -> ExamPlan:
        current_json = initial_plan.model_dump_json()
        system_prompt = "You are a Quality Assurance Editor. Fix leakage and ambiguity. Output JSON."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original:\n{context}\n\nCurrent Plan:\n{current_json}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return self._parse_and_validate(response.choices[0].message.content)

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