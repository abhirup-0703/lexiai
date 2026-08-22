# LexiAI

**LexiAI** is an AI-powered oral examiner for research papers. Feed it a PDF and it builds a Retrieval-Augmented Generation (RAG) pipeline over the paper's content, then quizzes and interrogates you on it — like a viva/thesis defense simulator — through either a desktop GUI or the command line.

## Features

- 📄 **PDF ingestion** — parses research paper PDFs using `marker-pdf`
- 🔍 **RAG pipeline** — chunks and embeds paper content with `langchain`, `langgraph`, and `faiss-cpu` / `fastembed` for retrieval
- 🤖 **Google Gemini integration** — powered by `google-generativeai` / `google-genai` via `langchain-google-genai`
- 🖥️ **Desktop GUI** — Tkinter-based interface (`LexiCognitionGUI`)
- ⌨️ **CLI mode** — run the full pipeline (`SpanishInquisitionPipeline`) directly from the terminal
- 📊 **Evaluation** — pipeline quality assessment using `ragas`

## Project Structure

```
lexiai/
├── assets/              # Static assets (images, icons, etc.)
├── src/
│   ├── pipeline.py       # SpanishInquisitionPipeline — core RAG/oral-exam pipeline
│   └── gui.py             # LexiCognitionGUI — Tkinter desktop interface
├── main.py               # Entry point (CLI + GUI launcher)
├── requirements.txt       # Python dependencies
└── .gitignore
```

## Requirements

- Python 3.10+ (recommended)
- A Google AI (Gemini) API key
- Tkinter (usually bundled with Python; on Linux you may need `sudo apt install python3-tk`)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhirup-0703/lexiai.git
   cd lexiai
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root with your Google AI credentials:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

### Desktop GUI (default)

Launch the Tkinter desktop app:

```bash
python main.py
```

### CLI mode

Run the pipeline directly on a research paper PDF from the terminal:

```bash
python main.py path/to/paper.pdf --cli
```

or simply:

```bash
python main.py path/to/paper.pdf
```

## How It Works

1. A research paper PDF is ingested and parsed (`marker-pdf`).
2. The text is chunked and embedded, then indexed in a FAISS vector store.
3. `SpanishInquisitionPipeline` uses LangChain/LangGraph with Gemini to retrieve relevant context and generate oral-exam-style questions and follow-ups based on the paper.
4. Responses can be evaluated for quality using `ragas`.

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.

## License

No license file is currently included in this repository. Consider adding one (e.g., MIT, Apache 2.0) to clarify how others may use this project.
