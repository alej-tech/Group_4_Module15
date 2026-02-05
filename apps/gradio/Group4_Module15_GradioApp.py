"""
Group4 - Trustworthy AI Explainer (Gradio Prototype)
Module 15 Team Project

Features implemented to satisfy Module 15 requirements:
- LLM chatbot interface with conversational memory (OpenAI API; fallback mock)
- Explainability: LIME (local token importance) + SHAP (global/tabular feature importance)
- User feedback collection (thumbs up/down + comment) persisted to CSV
- Latency management (status + optional streaming)
- Clear configuration and safe API key handling (env var OPENAI_API_KEY)

Run:
  python Group4_Module15_GradioApp.py
"""

from __future__ import annotations

from email.mime import message
import os
import time
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Tuple, Dict, Generator, Optional

import gradio as gr

# Explainability
from matplotlib.pyplot import hist
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a Trustworthy AI tutor. Explain concepts clearly and step-by-step. "
    "When unsure, say so. Avoid making up facts. Provide actionable guidance."
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FEEDBACK_CSV = os.getenv("MODULE15_FEEDBACK_CSV", "feedback_gradio.csv")


# =============================================================================
# LLM BACKEND
# =============================================================================

class MockLLM:
    """Fallback LLM for demo environments without API keys."""
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        time.sleep(0.5)
        return (
            "Mock response (no OPENAI_API_KEY detected).\n\n"
            f"Prompt: {prompt}\n\n"
            "Tip: set OPENAI_API_KEY as an environment variable to use the real LLM."
        )


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def build_messages(system_prompt: str, history: List[Tuple[str, str]], user_message: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for u, a in history[-12:]:  # keep last N turns to bound context size
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_message})
    return msgs


def generate_response_text(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> str:
    """Non-streaming response generation."""
    client = _get_openai_client()
    if client is None:
        return MockLLM().generate(message, temperature=temperature, max_tokens=max_tokens)

    msgs = build_messages(system_prompt, history, message)
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=msgs,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return resp.choices[0].message.content


def stream_response_text(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> Generator[str, None, None]:
    """Streaming response generator. Falls back to non-streaming if streaming unavailable."""
    client = _get_openai_client()
    if client is None:
        # simulate streaming for UX
        full = MockLLM().generate(message, temperature=temperature, max_tokens=max_tokens)
        chunk = ""
        for tok in full.split():
            chunk += tok + " "
            time.sleep(0.02)
            yield chunk
        return

    msgs = build_messages(system_prompt, history, message)
    try:
        stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=msgs,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            stream=True,
        )
        acc = ""
        for evt in stream:
            delta = evt.choices[0].delta.content or ""
            if delta:
                acc += delta
                yield acc
    except Exception:
        yield generate_response_text(message, history, temperature, max_tokens, system_prompt)


# =============================================================================
# TRUST / EXPLAINABILITY LAYER (Lightweight + robust)
# =============================================================================

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def quality_features(input_text: str, output_text: str, temperature: float, turns: int) -> Dict[str, float]:
    """Simple tabular features for a lightweight quality/risk scorer."""
    itok = len(input_text.split())
    otok = len(output_text.split())
    has_steps = 1.0 if any(s in output_text.lower() for s in ["1)", "1.", "- ", "step", "paso"]) else 0.0
    uncertainty = 1.0 if any(s in output_text.lower() for s in ["i'm not sure", "no estoy seguro", "uncertain", "can't verify"]) else 0.0
    too_long = 1.0 if otok > 450 else 0.0
    return {
        "input_tokens": float(itok),
        "output_tokens": float(otok),
        "has_steps": has_steps,
        "uncertainty": uncertainty,
        "too_long": too_long,
        "temperature": float(temperature),
        "turns": float(turns),
    }


def quality_score_from_features(feat: Dict[str, float]) -> float:
    """
    Heuristic quality scorer mapped to [0, 1].
    Higher = better. Designed to be stable and explainable.
    """
    # weights chosen for interpretability (not trained)
    z = 0.0
    z += 1.2 * feat["has_steps"]
    z += -1.1 * feat["uncertainty"]
    z += -0.9 * feat["too_long"]
    z += -0.15 * max(0.0, feat["temperature"] - 0.7)
    z += 0.05 * min(feat["input_tokens"], 200.0) / 200.0
    z += -0.02 * max(0.0, feat["turns"] - 8.0)
    return float(sigmoid(z))


# ---- LIME (local text explanation) ----
_LIME_EXPLAINER = LimeTextExplainer(class_names=["low_quality", "high_quality"])


def _lime_predict_proba(texts: List[str]) -> np.ndarray:
    """
    LIME needs a predict_proba on raw texts.
    We'll score based on text-only heuristics (input side).
    """
    probs = []
    for t in texts:
        t = t or ""
        itok = len(t.split())
        has_question = 1.0 if "?" in t else 0.0
        has_request_for_steps = 1.0 if any(k in t.lower() for k in ["paso", "step", "how", "como", "guía", "guide"]) else 0.0
        z = -0.2 + 0.6 * has_question + 0.4 * has_request_for_steps + 0.002 * min(itok, 300)
        p_hi = float(sigmoid(z))
        probs.append([1.0 - p_hi, p_hi])
    return np.array(probs, dtype=float)


def compute_lime_explanation(input_text: str, num_features: int = 10) -> str:
    exp = _LIME_EXPLAINER.explain_instance(
        input_text,
        _lime_predict_proba,
        num_features=int(num_features),
        labels=(1,),
    )
    # Convert to markdown-like list
    weights = exp.as_list(label=1)
    lines = ["**LIME (local) — influential tokens in the *user input***"]
    for tok, w in weights:
        sign = "↑" if w >= 0 else "↓"
        lines.append(f"- `{tok}`: {sign} {abs(w):.3f}")
    return "\n".join(lines)


# ---- SHAP (global/tabular explanation on quality features) ----
def compute_shap_explanation_tabular(feat: Dict[str, float]) -> str:
    """
    SHAP KernelExplainer for a small, deterministic scorer.
    Returns a compact markdown summary (top contributions).
    """
    feature_names = list(feat.keys())
    x = np.array([[feat[k] for k in feature_names]], dtype=float)

    # background: simple defaults (small set to keep it fast)
    bg = np.array([
        [50, 150, 1, 0, 0, 0.7, 2],
        [20, 80, 0, 0, 0, 0.7, 1],
        [120, 500, 1, 1, 1, 1.2, 6],
    ], dtype=float)

    def f(X: np.ndarray) -> np.ndarray:
        out = []
        for row in X:
            d = {feature_names[i]: float(row[i]) for i in range(len(feature_names))}
            out.append(quality_score_from_features(d))
        return np.array(out, dtype=float)

    explainer = shap.KernelExplainer(f, bg)
    shap_vals = explainer.shap_values(x, nsamples=50)  # keep small for speed
    sv = np.array(shap_vals).reshape(-1)

    # Top absolute contributions
    pairs = sorted(zip(feature_names, sv), key=lambda p: abs(p[1]), reverse=True)[:6]
    lines = ["**SHAP (tabular) — top feature contributions to *quality score***"]
    for name, val in pairs:
        sign = "↑" if val >= 0 else "↓"
        lines.append(f"- `{name}`: {sign} {abs(float(val)):.4f}")
    return "\n".join(lines)


def compute_explanation_bundle(input_text: str, output_text: str, temperature: float, turns: int) -> str:
    feat = quality_features(input_text, output_text, temperature=temperature, turns=turns)
    q = quality_score_from_features(feat)
    lime_md = compute_lime_explanation(input_text)
    shap_md = compute_shap_explanation_tabular(feat)
    header = (
        "### Explainability Summary\n"
        f"- **Quality score (0–1):** `{q:.3f}` (higher = better)\n"
        f"- **Signals:** steps={int(feat['has_steps'])}, uncertainty={int(feat['uncertainty'])}, too_long={int(feat['too_long'])}\n"
    )
    return header + "\n\n" + shap_md + "\n\n" + lime_md


# =============================================================================
# FEEDBACK
# =============================================================================

def append_feedback_csv(row: Dict[str, object], path: str = FEEDBACK_CSV) -> None:
    import csv
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def handle_feedback(message: str, response: str, rating: str, comment: str) -> str:
    if not rating:
        return "Please select a rating (Thumbs Up / Thumbs Down) before submitting."

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        "response": response,
        "rating": rating,
        "comment": (comment or "").strip(),
        "model": DEFAULT_MODEL,
    }
    append_feedback_csv(row)
    return f"✅ Feedback saved. Rating: {rating}"


# =============================================================================
# GRADIO APP
# =============================================================================

def chatbot_with_explanation(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    streaming: bool,
) -> Tuple[List[Tuple[str, str]], str, str]:
    # Generate response (stream or not)
    if streaming:
        # consume stream to final string for history
        final = ""
        for partial in stream_response_text(message, history, temperature, max_tokens, system_prompt):
            final = partial
        response = final
    else:
        response = generate_response_text(message, history, temperature, max_tokens, system_prompt)

    turns = len(history) + 1
    explanation = compute_explanation_bundle(message, response, temperature=temperature, turns=turns)
    history.append((message, response))
    return history, response, explanation

# =============================================================================
# fn helper
# =============================================================================
def messages_to_tuples(history: Any) -> List[Tuple[str, str]]:
    """
    Convert Gradio 'messages' history -> list of (user, assistant) tuples.
    Accepts None, tuples, or messages.
    """
    if not history:
        return []

    # If it already looks like tuples: [(u,a), ...]
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], (tuple, list)) and len(history[0]) == 2:
        # ensure strings
        out = []
        for u, a in history:
            out.append((str(u) if u is not None else "", str(a) if a is not None else ""))
        return out

    # Otherwise assume "messages" format: [{"role":..., "content":...}, ...]
    tuples: List[Tuple[str, str]] = []
    pending_user: str = ""
    has_pending = False

    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        content = "" if content is None else str(content)

        if role == "user":
            pending_user = content
            has_pending = True
        elif role == "assistant":
            if not has_pending:
                pending_user = ""
            tuples.append((pending_user, content))
            pending_user = ""
            has_pending = False

    # if last user has no assistant yet, keep it as a pair with empty assistant
    if has_pending:
        tuples.append((pending_user, ""))

    return tuples

def tuples_to_messages(history_tuples: Any) -> List[Dict[str, str]]:
    """
    Convert [(user, assistant), ...] -> [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]
    Works even if values aren't strings.
    """
    if not history_tuples:
        return []

    msgs: List[Dict[str, str]] = []
    for pair in history_tuples:
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            continue
        u, a = pair
        u = "" if u is None else str(u)
        a = "" if a is None else str(a)
        if u.strip():
            msgs.append({"role": "user", "content": u})
        if a.strip():
            msgs.append({"role": "assistant", "content": a})
    return msgs


def get_last_exchange_from_messages(hist: Any) -> Tuple[str, str]:
    """Return (last_user, last_assistant) from messages history."""
    if not hist:
        return "", ""
    # scan backwards for assistant, then nearest user before it
    for i in range(len(hist) - 1, -1, -1):
        m = hist[i]
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_assistant = str(m.get("content", "") or "")
            last_user = ""
            for j in range(i - 1, -1, -1):
                mj = hist[j]
                if isinstance(mj, dict) and mj.get("role") == "user":
                    last_user = str(mj.get("content", "") or "")
                    return last_user, last_assistant
            return "", last_assistant
    return "", ""


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Trustworthy AI Explainer (Gradio)", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Trustworthy AI Explainer (Gradio Prototype)")
        gr.Markdown(
            "Chat with memory + **LIME/SHAP explainability** + feedback logging.\n\n"
            "If `OPENAI_API_KEY` is missing, the app runs in **Mock** mode."
        )

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation", height=420)

                    message_input = gr.Textbox(
                        label="Your message",
                        placeholder="Ask about Trustworthy AI, explainability, bias, evaluation, etc.",
                        lines=2,
                    )
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=1):
                    gr.Markdown("### Explainability")
                    explanation_output = gr.Markdown(value="Send a message to see SHAP/LIME explanations.")

                    gr.Markdown("### Response (latest)")
                    latest_response = gr.Textbox(lines=10, interactive=False)

            with gr.Accordion("Advanced Settings", open=False):
                system_prompt = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3)
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                    max_tokens = gr.Slider(50, 2000, value=500, step=50, label="Max tokens")
                streaming = gr.Checkbox(value=True, label="Enable streaming (if supported)")

            gr.Markdown("### Feedback")
            with gr.Row():
                feedback_rating = gr.Radio(choices=["Thumbs Up", "Thumbs Down"], label="Rate the last response")
                feedback_comment = gr.Textbox(label="Optional comment", lines=2, placeholder="What was good/bad?")
            feedback_btn = gr.Button("Submit Feedback")
            feedback_status = gr.Textbox(label="Status", interactive=False)

            last_message = gr.State("")
            last_response = gr.State("")

            def _submit(message, history, temp, tokens, sys_prompt, streaming_enabled):
                if not (message or "").strip():
                    return history, "", "Please type a message.", ""

                # Convert Gradio messages history -> tuples (backend format)
                history_tuples = messages_to_tuples(history)

                # Call your existing backend (tuples in / tuples out)
                history_tuples, resp, exp = chatbot_with_explanation(
                    message, history_tuples, temp, tokens, sys_prompt, streaming_enabled
                )

                # Convert tuples back -> messages (for Gradio Chatbot display)
                history_msgs = tuples_to_messages(history_tuples)

                return history_msgs, resp, exp


            def _clear():
                return [], "", "Cleared.", ("", "")

            send_btn.click(
                fn=_submit,
                inputs=[message_input, chatbot, temperature, max_tokens, system_prompt, streaming],
                outputs=[chatbot, latest_response, explanation_output],
            )

            # Use .then to reset textbox and state
            send_btn.click(fn=lambda: "", outputs=message_input)

            # Properly update last_message/last_response after sending
            def _set_last_from_chat(hist):
                return get_last_exchange_from_messages(hist)


            send_btn.click(fn=_set_last_from_chat, inputs=chatbot, outputs=[last_message, last_response])

            clear_btn.click(fn=_clear, outputs=[chatbot, latest_response, explanation_output, gr.State()]).then(
                fn=lambda: "",
                outputs=[message_input]
            ).then(
                fn=lambda: ("", ""),
                outputs=[last_message, last_response]
            )

            feedback_btn.click(
                fn=handle_feedback,
                inputs=[last_message, last_response, feedback_rating, feedback_comment],
                outputs=feedback_status,
            )

        with gr.Tab("About"):
            gr.Markdown(
                "## Module 15 Notes\n"
                "- **Goal**: UI for LLM apps with trust mechanisms: memory, explainability, feedback.\n"
                "- **Explainability here**: LIME (token influence on a lightweight text scorer) + "
                "SHAP (feature contributions to a tabular quality score).\n"
                "- **Security**: API keys via env var `OPENAI_API_KEY`.\n"
            )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
