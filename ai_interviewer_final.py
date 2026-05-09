"""AI-Driven Automated Interviewer for Project Presentations

Free, local prototype for:
- OCR from screen/slide images
- Speech-to-text from presentation audio
- Context-aware question generation
- Adaptive follow-up questions
- Scoring and PDF report generation

Run in Google Colab or local Jupyter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import easyocr
import pandas as pd
from faster_whisper import WhisperModel
from keybert import KeyBERT
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "google/flan-t5-small"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
WHISPER_MODEL_NAME = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_N_QUESTIONS = 5

# Make reportlab more permissive on some environments
rl_config.warnOnMissingFontGlyphs = 0

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
print("Loading OCR model...")
ocr_reader = easyocr.Reader(["en"], gpu=False)

print("Loading speech-to-text model...")
stt_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

print("Loading question generation model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Loading embedding and keyword models...")
kw_model = KeyBERT(model=EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# -----------------------------------------------------------------------------
# Text utilities
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_image(image_path: str) -> str:
    results = ocr_reader.readtext(image_path, detail=0)
    return clean_text(" ".join(results))


def transcribe_audio(audio_path: str) -> str:
    segments, _info = stt_model.transcribe(audio_path, beam_size=5)
    parts = [seg.text for seg in segments]
    return clean_text(" ".join(parts))


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    if len(text.split()) < 5:
        return []
    kws = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n,
    )
    return [k for k, _ in kws]


def similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    va = embedder.encode(a, convert_to_tensor=True)
    vb = embedder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(va, vb)[0][0])


# -----------------------------------------------------------------------------
# Generation helpers
# -----------------------------------------------------------------------------
def run_flan(prompt: str, max_new_tokens: int = 160) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    outputs = qg_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_questions(context: str, n: int = DEFAULT_N_QUESTIONS) -> List[str]:
    prompt = f"""
Generate exactly {n} concise interview questions from this project context.
Focus on architecture, implementation, model choice, evaluation, originality, and limitations.
Return only questions, one per line.

Context:
{context[:3000]}
"""
    raw = run_flan(prompt)
    lines = [re.sub(r"^\d+[\).\s-]*", "", x).strip() for x in raw.split("\n")]
    questions = [x for x in lines if len(x) > 10]

    fallback = [
        "Can you explain the architecture of your project?",
        "What problem does your project solve?",
        "Why did you choose these models or libraries?",
        "What implementation challenges did you face?",
        "How did you evaluate your system?",
        "What are the limitations of your solution?",
        "How can the project be improved further?",
        "Which part of the project is most original?",
    ]

    for q in fallback:
        if len(questions) >= n:
            break
        if q not in questions:
            questions.append(q)

    return questions[:n]


def generate_followup(question: str, answer: str, context: str) -> str:
    prompt = f"""
Question:
{question}

Student answer:
{answer}

Project context:
{context[:1000]}

Generate 1 short follow-up interview question to probe deeper into technical understanding.
Return only the question.
"""
    return run_flan(prompt).strip()


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def score_answer(answer: str, context: str, question: str) -> Dict[str, Any]:
    answer_l = answer.lower()
    kws = extract_keywords(context, top_n=8)

    keyword_coverage = sum(1 for kw in kws if kw.lower() in answer_l) / max(len(kws), 1)
    sim_context = similarity(answer, context)
    sim_question = similarity(answer, question)

    technical_depth = round(
        min(100, 35 * sim_context + 35 * keyword_coverage * 100 + 30 * sim_question),
        2,
    )

    clarity = round(min(100, 20 + min(len(answer.split()) / 1.2, 80)), 2)
    originality = round(max(0, 100 - 80 * sim_context), 2)
    implementation = round(
        min(
            100,
            40 * keyword_coverage * 100 + 30 * sim_question + 30 * min(len(answer.split()) / 40, 1) * 100,
        ),
        2,
    )

    overall = round(0.35 * technical_depth + 0.20 * clarity + 0.15 * originality + 0.30 * implementation, 2)

    if overall >= 80:
        feedback = "Very strong answer with good technical depth."
    elif overall >= 60:
        feedback = "Good answer but needs more implementation details."
    else:
        feedback = "Weak answer. Add clearer technical explanation."

    return {
        "technical_depth": technical_depth,
        "clarity": clarity,
        "originality": originality,
        "implementation_understanding": implementation,
        "overall": overall,
        "feedback": feedback,
    }


# -----------------------------------------------------------------------------
# Main interview pipeline
# -----------------------------------------------------------------------------
def run_interview(image_path: str, audio_path: str, student_name: str = "Student", project_name: str = "AI-Driven Automated Interviewer") -> List[Dict[str, Any]]:
    image_path = str(image_path)
    audio_path = str(audio_path)

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    print("Extracting OCR text...")
    ocr_text = extract_text_from_image(image_path)

    print("Transcribing audio...")
    stt_text = transcribe_audio(audio_path)

    project_context = clean_text(ocr_text + " " + stt_text)

    print("\nOCR TEXT:\n", ocr_text[:1200])
    print("\nSTT TEXT:\n", stt_text[:1200])
    print("\nMERGED CONTEXT LENGTH:", len(project_context))

    print("\nGenerating questions...")
    questions = generate_questions(project_context, n=DEFAULT_N_QUESTIONS)
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    results: List[Dict[str, Any]] = []
    question_history: List[str] = []

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        answer = input("Your answer: ").strip()

        result = score_answer(answer, project_context, q)

        if result["overall"] < 70 or len(answer.split()) < 25:
            followup = generate_followup(q, answer, project_context)
            print("\nFollow-up Question:")
            print(followup)
            followup_answer = input("Your follow-up answer: ").strip()
            result["followup_question"] = followup
            result["followup_answer"] = followup_answer
        else:
            result["followup_question"] = ""
            result["followup_answer"] = ""

        result["question"] = q
        result["answer"] = answer
        results.append(result)
        question_history.append(q)

    print("\nInterview completed successfully.")
    print("Average Score:", round(sum(r["overall"] for r in results) / max(len(results), 1), 2))

    save_results(results, student_name=student_name, project_name=project_name)
    generate_report(results, student_name=student_name, project_name=project_name)
    return results


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], student_name: str = "Student", project_name: str = "AI-Driven Automated Interviewer") -> None:
    payload = {
        "student_name": student_name,
        "project_name": project_name,
        "results": results,
    }
    with open("interview_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("Saved: interview_results.json")


def generate_report(results: List[Dict[str, Any]], student_name: str = "Student", project_name: str = "AI-Driven Automated Interviewer") -> None:
    doc = SimpleDocTemplate("interview_report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(project_name, styles["Title"]))
    story.append(Paragraph(f"Student: {student_name}", styles["Normal"]))
    story.append(Spacer(1, 12))

    avg_score = sum(r["overall"] for r in results) / max(len(results), 1)
    story.append(Paragraph(f"Overall Score: {avg_score:.2f}/100", styles["Heading2"]))
    story.append(Spacer(1, 10))

    summary_data = [["Metric", "Average"]]
    for key in ["technical_depth", "clarity", "originality", "implementation_understanding"]:
        avg_val = sum(r[key] for r in results) / max(len(results), 1)
        summary_data.append([key.replace("_", " ").title(), f"{avg_val:.2f}"])

    table = Table(summary_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    for idx, r in enumerate(results, 1):
        story.append(Paragraph(f"Q{idx}: {r['question']}", styles["Heading3"]))
        story.append(Paragraph(f"Answer: {r['answer']}", styles["Normal"]))
        if r.get("followup_question"):
            story.append(Paragraph(f"Follow-up: {r['followup_question']}", styles["Normal"]))
            story.append(Paragraph(f"Follow-up Answer: {r['followup_answer']}", styles["Normal"]))
        story.append(Paragraph(f"Score: {r['overall']}/100", styles["Normal"]))
        story.append(Paragraph(f"Feedback: {r['feedback']}", styles["Normal"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    print("Saved: interview_report.pdf")


# -----------------------------------------------------------------------------
# Optional batch mode
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example:
    # run_interview("slide.png", "audio.wav", student_name="Rahul Kumar Sharma")
    print("Module loaded. Uncomment run_interview(...) in __main__ to execute the full flow.")
