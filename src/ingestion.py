import re
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

class MarkerIngestion:
    def _init_(self):
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

        # --- NEW: Cache Check ---
        # e.g., "data/paper.pdf" -> "data/paper.txt"
        cache_path = os.path.splitext(file_path)[0] + ".txt"
        
        if os.path.exists(cache_path):
            print(f"Found cached text file: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: Could not read cache ({e}). Re-ingesting...")
        # ------------------------

        print(f"Ingesting: {file_path}")
        rendered = self.converter(file_path)
        full_text, _, _ = text_from_rendered(rendered)
        
        # Smart Truncation
        abstract_pattern = r"(?i)^.?\babstract\b[:.]?\s"
        match = re.search(abstract_pattern, full_text, re.MULTILINE)
        
        if match:
            print("Smart Truncation: Dropping front matter before Abstract.")
            full_text = full_text[match.start():]
        
        final_text = self.sanitize_content(full_text)

        # --- NEW: Save Cache ---
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(f"Cached ingested text to: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache file ({e})")
        # -----------------------
        
        return final_text