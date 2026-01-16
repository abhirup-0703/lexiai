import re
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

class MarkerIngestion:
    def __init__(self):
        print("Loading Marker models (this may take time on first run)...")
        self.converter = PdfConverter(artifact_dict=create_model_dict())

    def sanitize_content(self, raw_text: str) -> str:
        pattern = r"(?i)^\s*(references|bibliography|appendices|appendix)\s*$"
        lines = raw_text.split('\n')
        cutoff_index = -1
        
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                cutoff_index = i
                break
        
        if cutoff_index != -1:
            return "\n".join(lines[:cutoff_index])
        return raw_text

    def process_pdf(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Ingesting: {file_path}")
        rendered = self.converter(file_path)
        full_text, _, _ = text_from_rendered(rendered)
        
        # Smart Truncation
        abstract_pattern = r"(?i)^.*?\babstract\b[:.]?\s*"
        match = re.search(abstract_pattern, full_text, re.MULTILINE)
        
        if match:
            print("Smart Truncation: Dropping front matter before Abstract.")
            full_text = full_text[match.start():]
        
        return self.sanitize_content(full_text)