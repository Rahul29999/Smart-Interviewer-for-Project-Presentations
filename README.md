# AI Driven Automated Interviewer for Project Presentations

A free, local AI prototype that listens to a student project presentation, extracts screen content with OCR, transcribes speech, generates interview questions, asks adaptive follow-ups, scores the answers, and creates a PDF report.

## Why this project
This project was built for a hackathon challenge where the system must understand a student's presentation and conduct a real-time adaptive interview using free tools only.

## Features
- OCR from slide or screen images using EasyOCR
- Speech-to-text using Faster-Whisper
- Context-aware interview question generation using FLAN-T5-small
- Adaptive follow-up questions based on answer quality
- Scoring on:
  - technical depth
  - clarity
  - originality
  - implementation understanding
- Final PDF report generation
- JSON result export

## Tech stack
- Python
- EasyOCR
- Faster-Whisper
- Transformers
- Sentence-Transformers
- KeyBERT
- ReportLab
- Pandas

## Recommended file structure
```text
smart-interviewer/
├── ai_interviewer_final.py
├── requirements.txt
├── README.md
├── slide.png
├── audio.wav
├── interview_report.pdf        # generated after running
├── interview_results.json      # generated after running
└── .gitignore
```

## How to run
### Option 1: Google Colab
1. Upload `ai_interviewer_final.py`, `slide.png`, and `audio.wav`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script.
4. Enter answers in the notebook/console when questions appear.

### Option 2: Local Jupyter / Python
1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Keep `slide.png` and `audio.wav` in the same folder as the script.
4. Run:
   ```bash
   python ai_interviewer_final.py
   ```
5. Uncomment the `run_interview(...)` line in the `__main__` block.

## Input files expected
- `slide.png` or any image file of the presentation screen
- `audio.wav` or any supported audio file containing the presentation speech

## Output files generated
- `interview_results.json`
- `interview_report.pdf`

## Example usage
In the Python file:
```python
from ai_interviewer_final import run_interview

run_interview(
    image_path="slide.png",
    audio_path="audio.wav",
    student_name="Rahul Kumar Sharma",
    project_name="Smart Interviewer for Project Presentations"
)
```

## Notes
- The project is designed to work with free/open-source models only.
- CPU execution is supported.
- For faster OCR and transcription, a GPU is helpful but not required.
- The OCR text and audio transcript should be from the same presentation for best results.

## Project title for submission
**Smart Interviewer for Project Presentations**

## Short description for submission
An AI system that extracts presentation content from screen images and speech, generates adaptive interview questions, scores responses, and produces a final evaluation report.
