import argparse
from src.pipeline import SpanishInquisitionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Oral Examiner Pipeline")
    parser.add_argument("pdf_path", type=str, help="Path to the research paper PDF")
    args = parser.parse_args()

    pipeline = SpanishInquisitionPipeline()
    pipeline.run(args.pdf_path)