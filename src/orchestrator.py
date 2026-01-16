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
        q_data = state["exam_plan"]["questions"][idx]
        last_answer = state["history"][-1].replace("User: ", "")
        
        result = self.judge.evaluate_answer(
            question=q_data["question"],
            user_answer=last_answer,
            context=q_data["context_snippet"],
            criteria=q_data["rubric"]["criteria"]
        )
        
        print(f"\033[1;30m   >>> [Judge]: {result.feedback}\033[0m")
        return {
            "last_judge_result": asdict(result),
            "retry_count": 0 if not result.is_remedial_needed else state["retry_count"]
        }

    def remedial_action(self, state: InterviewState):
        retries = state["retry_count"]
        if retries >= 2:
            self.io.output("We seem to be stuck. Let's move to the next topic.")
            return {"retry_count": 99} 

        rubric_hint = state["exam_plan"]["questions"][state["current_q_index"]]["rubric"]["criteria"]
        hint = f"That's not quite what I'm looking for. Hint: {rubric_hint}. Try again."
        self.io.output(hint)
        return {"history": [f"AI Hint: {hint}"], "retry_count": retries + 1}

    def advance_step(self, state: InterviewState):
        return {
            "current_q_index": state["current_q_index"] + 1,
            "retry_count": 0,
            "last_judge_result": None
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
            res = state["last_judge_result"]
            if res and res["is_remedial_needed"] and state["retry_count"] < 99:
                return "remedial"
            return "advance"

        workflow.add_conditional_edges("grade", route_logic)
        workflow.add_edge("remedial", "listen")
        workflow.add_edge("advance", "ask")
        
        return workflow.compile(checkpointer=MemorySaver())