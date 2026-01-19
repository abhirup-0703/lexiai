import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from .config import Config
from .models import ExamPlan

class CognitivePlanner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.3
        )

    def _clean_json_string(self, raw_string: str) -> str:
        # 1. Remove Markdown
        clean = re.sub(r"```json\s*", "", raw_string)
        clean = re.sub(r"```\s*$", "", clean)
        clean = clean.strip()
        
        # 2. Attempt parsing to see if it's already valid
        try:
            json.loads(clean)
            return clean
        except json.JSONDecodeError:
            # 3. Repair: Fix invalid escapes (common with LaTeX/Paths in LLM output)
            # Regex finds backslashes NOT followed by valid JSON escape chars (" \ / b f n r t u)
            # and escapes them (e.g., \alpha -> \\alpha)
            clean = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', clean)
            return clean

    def generate_exam_plan(self, context_text: str, enable_refinement: bool = True) -> ExamPlan:
        # Safe limit for Flash context
        safe_context = context_text[:100000] 
        
        print("Generating Deep-Dive Exam Plan...")
        initial_plan = self._generate_initial_pass(safe_context)
        
        if not enable_refinement:
            return initial_plan

        print("Refining for Leakage and Ambiguity...")
        return self._refine_plan(initial_plan, safe_context)

    def _generate_initial_pass(self, context: str) -> ExamPlan:
        json_skeleton = """
        {
          "topic": "Specific Subject Title",
          "questions": [
            {
              "type": "Recall", 
              "question": "Question text...",
              "context_snippet": "Verbatim quote from text...",
              "grading_rubric": {
                  "grading_criteria": "A specific sentence describing what a correct answer MUST contain.",
                  "concepts": ["keyword1", "keyword2"],
                  "exemplar_answer": "A perfect, concise 1-2 sentence answer to this question."
              }
            }
          ]
        }
        """
        
        system_prompt = f"""
        You are a rigorous Academic Examiner. Your goal is to test deep understanding, not just memory.
        Analyze the text and generate an exam plan in STRICT JSON format.

        ### INSTRUCTIONS:
        1. **Bloom's Taxonomy Progression**: Generate exactly 5 questions in this order:
            - Q1 (Recall): Retrieve specific facts.
            - Q2 (Understand): Explain a concept in own words.
            - Q3 (Apply): Apply a concept to a hypothetical scenario.
            - Q4 (Analyze): Compare/contrast or deconstruct an argument.
            - Q5 (Evaluate): Judge the validity or implications of a conclusion.

        2. **Rubric Quality**:
            - "grading_criteria": MUST be a full sentence explaining the logic of the answer, not just keywords.
            - "exemplar_answer": Write the *perfect* answer. The AI Judge will use this to grade the student.
            - "concepts": List 3-5 technical terms that must appear in the answer.

        3. **Constraints**:
            - Questions must be answerable *only* using the provided context.
            - Do not give away the answer in the question itself (No Leakage).
            - **IMPORTANT**: Escape all backslashes in the output (e.g. use \\\\alpha for LaTeX).

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
        
        system_prompt = """
        You are a Critical Quality Assurance Editor. Your job is to make the exam harder and fairer.
        
        Review the exam plan and output a CORRECTED JSON object.
        
        ### CRITIQUE CHECKLIST:
        1. **Check for Leakage**: Does the question inadvertently contain the answer? If yes, rewrite it.
        2. **Check for Ambiguity**: Is the grading criteria too vague (e.g., "Answers may vary")? If yes, replace it with specific facts from the text.
        3. **Check Progression**: Does the final question actually require evaluation/synthesis? If it's just a lookup, rewrite it to be harder.
        4. **Verify Exemplars**: Ensure the "exemplar_answer" is actually correct based on the text.
        5. **Escape Characters**: Ensure all LaTeX backslashes are double-escaped.
        
        Output ONLY the fixed JSON.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original Text Context (for verification):\n{context}\n\nCurrent Draft Plan:\n{current_json}")
        ]
        
        self.llm.temperature = 0.1
        response = self.llm.invoke(messages)
        return self._parse_and_validate(response.content)

    def _parse_and_validate(self, raw_json: str) -> ExamPlan:
        try:
            clean_str = self._clean_json_string(raw_json)
            parsed = json.loads(clean_str)
            
            # Handle nesting if LLM wraps it in a root key
            if "questions" not in parsed:
                for v in parsed.values():
                    if isinstance(v, dict) and "questions" in v:
                        parsed = v
                        break
            
            return ExamPlan(**parsed)
        except Exception as e:
            print(f"Validation Error: {e}")
            print(f"Failed JSON snippet: {raw_json[:500]}...") 
            raise