# import argparse
# from src.pipeline import SpanishInquisitionPipeline

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="AI Oral Examiner Pipeline")
#     parser.add_argument("pdf_path", type=str, help="Path to the research paper PDF")
#     args = parser.parse_args()

#     pipeline = SpanishInquisitionPipeline()
#     pipeline.run(args.pdf_path)


import argparse
import sys
import subprocess
from src.pipeline import SpanishInquisitionPipeline

def run_cli(pdf_path):
    if not pdf_path:
        print("Error: PDF path required for CLI mode.")
        return
    pipeline = SpanishInquisitionPipeline()
    pipeline.run(pdf_path)

def run_ui():
    print("Launching Streamlit UI...")
    # Use subprocess to call streamlit run src/ui.py
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui.py"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Oral Examiner Pipeline")
    parser.add_argument("pdf_path", type=str, nargs="?", help="Path to the research paper PDF (Required for CLI)")
    parser.add_argument("--ui", action="store_true", help="Launch the Web Interface")
    
    args = parser.parse_args()

    if args.ui:
        run_ui()
    else:
        run_cli(args.pdf_path)