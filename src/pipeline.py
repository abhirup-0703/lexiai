from .ingestion import MarkerIngestion
from .planner import CognitivePlanner
from .judge import RagasJudge
from .orchestrator import OralExamOrchestrator
from .interfaces import TextCLI
from .config import Config

class SpanishInquisitionPipeline:
    def __init__(self):
        Config.validate()
        print("Initializing Pipeline Components...")
        self.ingestor = MarkerIngestion()
        self.planner = CognitivePlanner()
        self.judge = RagasJudge()
        self.io = TextCLI()

    def run(self, pdf_path: str):
        # 1. Ingest
        try:
            raw_text = self.ingestor.process_pdf(pdf_path)
            print(f"Content extracted: {len(raw_text)} chars")
        except Exception as e:
            print(f"Ingestion Failed: {e}")
            return

        # 2. Plan
        try:
            plan = self.planner.generate_exam_plan(raw_text)
            print(f"Exam Topic: {plan.topic}")
        except Exception as e:
            print(f"Planning Failed: {e}")
            return

        # 3. Exam Loop
        orchestrator = OralExamOrchestrator(self.io, self.judge)
        app = orchestrator.build_workflow()
        
        initial_state = {
            "exam_plan": plan.model_dump(),
            "current_q_index": 0,
            "history": [],
            "last_judge_result": None,
            "retry_count": 0
        }
        
        config = {"configurable": {"thread_id": "cli_session"}}
        self.io.output(f"Starting exam on: {plan.topic}")
        
        for event in app.stream(initial_state, config=config):
            pass
        
        print("\nExam Complete.")