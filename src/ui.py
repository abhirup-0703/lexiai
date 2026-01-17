import streamlit as st
import time
import os
from .ingestion import MarkerIngestion
from .planner import CognitivePlanner
from .judge import RagasJudge
from .config import Config

# Page Configuration
st.set_page_config(
    page_title="LexiCognition Oral Examiner",
    page_icon="🎓",
    layout="wide"
)

# --- CSS Styling for "Nice UI" ---
st.markdown("""
<style>
    .stChatMessage { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    .rubric-card { 
        background-color: #e8f4f8; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
    }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "pipeline_ready" not in st.session_state:
    try:
        Config.validate()
        st.session_state.ingestor = MarkerIngestion()
        st.session_state.planner = CognitivePlanner()
        st.session_state.judge = RagasJudge()
        st.session_state.pipeline_ready = True
    except Exception as e:
        st.error(f"System Init Failed: {e}")
        st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "exam_plan" not in st.session_state:
    st.session_state.exam_plan = None
if "current_q_index" not in st.session_state:
    st.session_state.current_q_index = 0
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0
if "exam_complete" not in st.session_state:
    st.session_state.exam_complete = False

# --- Helper Functions ---
def add_message(role, content, metrics=None):
    st.session_state.chat_history.append({
        "role": role, 
        "content": content,
        "metrics": metrics
    })

def process_answer(user_input):
    # 1. Get Context
    plan = st.session_state.exam_plan
    idx = st.session_state.current_q_index
    question_data = plan.questions[idx]
    
    # 2. Evaluate
    with st.spinner("👨‍🏫 Grading answer..."):
        # NEW: Pass exemplar if available
        exemplar = question_data.rubric.exemplar
        result = st.session_state.judge.evaluate_answer(
            question=question_data.question,
            user_answer=user_input,
            context=question_data.context_snippet,
            criteria=question_data.rubric.criteria,
            exemplar=exemplar
        )
    
    # 3. Logic Branching
    if result.is_remedial_needed and st.session_state.retry_count < 2:
        # FAIL: Give Hint
        st.session_state.retry_count += 1
        hint = f"Not quite. Hint: {question_data.rubric.criteria}. Try again."
        add_message("assistant", hint, metrics=result)
    else:
        # PASS (or max retries reached): Move Next
        if st.session_state.retry_count >= 2:
            transition_msg = "Let's move on to the next topic."
            add_message("assistant", transition_msg, metrics=result)
        else:
            good_job_msg = "Correct. Moving on."
            add_message("assistant", good_job_msg, metrics=result)
            
        st.session_state.current_q_index += 1
        st.session_state.retry_count = 0
        
        # Check if Exam Done
        if st.session_state.current_q_index >= len(plan.questions):
            st.session_state.exam_complete = True
            add_message("assistant", f"Exam Complete! You covered: {plan.topic}")
        else:
            # Queue Next Question
            next_q = plan.questions[st.session_state.current_q_index].question
            add_message("assistant", next_q)

# --- MAIN UI LAYOUT ---

# Sidebar: Configuration & Status
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file and not st.session_state.exam_plan:
        if st.button("🚀 Generate Exam"):
            with st.spinner("Ingesting PDF & Generating Questions..."):
                # Save temp file
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Run Pipeline Steps
                raw_text = st.session_state.ingestor.process_pdf("temp.pdf")
                plan = st.session_state.planner.generate_exam_plan(raw_text)
                
                st.session_state.exam_plan = plan
                
                # Start First Question
                first_q = plan.questions[0].question
                add_message("assistant", f"Welcome to your exam on **{plan.topic}**. Let's begin.")
                add_message("assistant", first_q)
                st.rerun()

    # Exam Progress (Only if plan exists)
    if st.session_state.exam_plan:
        st.divider()
        st.subheader("📊 Exam Progress")
        total = len(st.session_state.exam_plan.questions)
        current = st.session_state.current_q_index + 1
        progress = min(current / total, 1.0)
        st.progress(progress)
        st.caption(f"Question {min(current, total)} of {total}")
        
        # Teacher's Cheat Sheet (Debug View)
        with st.expander("👀 Teacher's Rubric (Hidden)"):
            if not st.session_state.exam_complete:
                q = st.session_state.exam_plan.questions[st.session_state.current_q_index]
                st.markdown(f"**Criteria:** {q.rubric.criteria}")
                if q.rubric.exemplar:
                     st.markdown(f"**Exemplar:** {q.rubric.exemplar}")
                st.markdown(f"**Keywords:** {', '.join(q.rubric.key_concepts)}")

# Main Chat Area
st.title("🎓 LexiCognition")
st.caption("AI-Powered Oral Examination Engine")

if not st.session_state.exam_plan:
    st.info("👈 Please upload a PDF in the sidebar to begin.")
else:
    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # If there are grading metrics, show them in a tiny expander
            if msg.get("metrics"):
                m = msg["metrics"]
                status = "❌ Remedial" if m.is_remedial_needed else "✅ Passed"
                with st.expander(f"Grading: {status} ({m.score}/10)"):
                    cols = st.columns(3)
                    cols[0].metric("Faithfulness", f"{m.metrics['faithfulness']:.2f}")
                    cols[1].metric("Relevance", f"{m.metrics['relevancy']:.2f}")
                    cols[2].metric("Correctness", f"{m.metrics['correctness']:.2f}")

    # User Input
    if not st.session_state.exam_complete:
        if user_input := st.chat_input("Type your answer here..."):
            add_message("user", user_input)
            process_answer(user_input)
            st.rerun()
    else:
        st.success("🎉 Examination Concluded.")
        if st.button("Reset Exam"):
            st.session_state.clear()
            st.rerun()