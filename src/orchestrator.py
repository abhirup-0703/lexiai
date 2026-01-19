from dataclasses import asdict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .models import InterviewState
from .interfaces import InteractionInterface
from .judge import RagasJudge

class OralExamOrchestrator:
    def __init__(self, interface: InteractionInterface, judge_engine: RagasJudge):
        self.io = interface
        self.judge = judge_engine

    def ask_question(self, state: InterviewState):
        # CHECK: Is there an active follow-up question?
        if state.get("followup_question"):
            text = state["followup_question"]["question"]
            self.io.output(f"[Follow-up]: {text}")
            return {"history": [f"AI (Follow-up): {text}"]}

        # Normal Flow
        idx = state["current_q_index"]
        plan = state["exam_plan"]
        
        if idx >= len(plan["questions"]):
            return {"history": ["System: Exam Finished"]}
            
        q_data = plan["questions"][idx]
        text = q_data["question"]
        self.io.output(text)
        return {"history": [f"AI: {text}"]}

    def listen_answer(self, state: InterviewState):
        answer = self.io.input()
        return {"history": [f"User: {answer}"]}

    def evaluate_response(self, state: InterviewState):
        idx = state["current_q_index"]
        last_answer = state["history"][-1].replace("User: ", "")
        
        # Determine Context: Are we grading a follow-up or a main question?
        followup_data = state.get("followup_question")
        
        if followup_data:
            # GRADE FOLLOW-UP
            q_text = followup_data["question"]
            context = followup_data["context_snippet"]
            rubric = followup_data["rubric"]
            exemplar = rubric.get("exemplar")
        else:
            # GRADE MAIN QUESTION
            q_data = state["exam_plan"]["questions"][idx]
            q_text = q_data["question"]
            context = q_data["context_snippet"]
            rubric = q_data.get("rubric", {})
            exemplar = rubric.get("exemplar") or rubric.get("exemplar_answer")

        # Call Judge
        result = self.judge.evaluate_answer(
            question=q_text,
            user_answer=last_answer,
            context=context,
            criteria=rubric.get("criteria", ""),
            exemplar=exemplar
        )
        
        score = result.score
        print(f"\033[1;30m   >>> [Judge]: {result.feedback} (Score: {score})\033[0m")
        
        # --- 3-TIER LOGIC ---
        
        # 1. PASS (> 7)
        if score > 7.0:
            # If we were in follow-up mode, clear it now
            return {
                "last_judge_result": asdict(result),
                "followup_question": None, # Clear follow-up
                "retry_count": 0,          # Reset retries for next Q
                "current_q_index": idx if followup_data else idx # Logic handled in 'route_logic'
            }

        # 2. PARTIAL (3 - 7) -> Generate Follow-up (Only if not already IN a follow-up)
        if 3.0 <= score <= 7.0:
            if not followup_data:
                # Generate specific follow-up
                print("   >>> [Orchestrator]: Generating specific follow-up...")
                new_q_data = self.judge.generate_followup(
                    original_q=q_text,
                    answer=last_answer,
                    rubric=rubric.get("criteria", ""),
                    context=context
                )
                return {
                    "last_judge_result": asdict(result),
                    "followup_question": new_q_data, # Set the follow-up state
                    "retry_count": state["retry_count"] + 1
                }
            else:
                # Already in follow-up and still mediocre? Treat as fail/hint to avoid infinite loop
                pass 

        # 3. FAIL (< 3) OR Failed Follow-up
        # Pass through to 'remedial_action' via route_logic
        return {
            "last_judge_result": asdict(result),
            # Keep follow-up active if we are in it, so the hint applies to the follow-up
            "followup_question": followup_data, 
            "retry_count": state["retry_count"] # Will be incremented in remedial_action
        }

    def remedial_action(self, state: InterviewState):
        retries = state["retry_count"]
        
        # Hard Stop after max retries
        if retries >= 2:
            self.io.output("We seem to be stuck. Let's move to the next topic.")
            return {"retry_count": 99, "followup_question": None} 

        # Determine which rubric to hint from
        if state.get("followup_question"):
            rubric_hint = state["followup_question"]["rubric"].get("criteria", "Review the concept. Give exactly ONE small hint about the concept.")
        else:
            q_data = state["exam_plan"]["questions"][state["current_q_index"]]
            rubric_hint = q_data.get("rubric", {}).get("criteria", "Review the concept. Give exactly ONE small hint about the concept.")
        
        # Give Hint (Score < 3 behavior)
        hint = f"Not quite. Hint: {rubric_hint}. Try again."
        self.io.output(hint)
        return {"history": [f"AI Hint: {hint}"], "retry_count": retries + 1}

    def advance_step(self, state: InterviewState):
        return {
            "current_q_index": state["current_q_index"] + 1,
            "retry_count": 0,
            "last_judge_result": None,
            "followup_question": None
        }

    def build_workflow(self):
        workflow = StateGraph(InterviewState)
        workflow.add_node("ask", self.ask_question)
        workflow.add_node("listen", self.listen_answer)
        workflow.add_node("grade", self.evaluate_response)
        workflow.add_node("remedial", self.remedial_action)
        workflow.add_node("advance", self.advance_step)
        
        workflow.set_entry_point("ask")
        workflow.add_edge("ask", "listen")
        workflow.add_edge("listen", "grade")
        
        def route_logic(state):
            idx = state["current_q_index"]
            if idx >= len(state["exam_plan"]["questions"]): return END
            
            res = state.get("last_judge_result")
            score = res["score"] if res else 0
            
            # Logic:
            # 1. If Pass (> 7) -> Advance (unless we want to confirm follow-up success, but simplified: pass is pass)
            if score > 6.9:
                return "advance"
            
            # 2. If Follow-up Active (meaning we just set it in grade, OR we failed it)
            if state.get("followup_question"):
                # If we just generated it (score was 3-7), we need to ASK it.
                # If we failed it (score < 3), we need REMEDIAL.
                # How to distinguish? 
                # If 'retry_count' didn't jump to 99, and score was 3-7, 'grade' set followup.
                # However, 'grade' returns. We are here.
                # If score was 3-7, we set 'followup_question'. We want to route to 'ask'.
                if 3.0 <= score <= 7.0:
                    return "ask" # Loop back to ask the follow-up
                
            # 3. If Fail (< 3) or Failed Follow-up (< 3)
            # Check Max Retries handled in remedial node, but here we route there.
            if state["retry_count"] < 3:
                return "remedial"
            
            return "advance"

        workflow.add_conditional_edges("grade", route_logic)
        workflow.add_edge("remedial", "listen") # After hint, listen again
        workflow.add_edge("advance", "ask")
        
        return workflow.compile(checkpointer=MemorySaver())