"""
Group4 - Trustworthy AI Explainer (Streamlit Dashboard)
Module 15 Team Project

Implements Module 15 technical requirements:
- Multi-page dashboard (Chat / Explainability / Feedback / Monitoring / Documentation)
- LLM chatbot with conversational memory (OpenAI API; fallback mock)
- Explainability displayed alongside outputs (LIME local + SHAP tabular)
- Feedback collection + analytics (stored in-session + persisted to CSV)
- State management (st.session_state) + caching (@st.cache_resource, @st.cache_data)
- Latency management (streaming if available + spinners)
- User-friendly error handling

Run:
  streamlit run Group4_Module15_StreamlitApp.py
"""

from __future__ import annotations

import os
import time
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import shap
from lime.lime_text import LimeTextExplainer

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PAGE CONFIG (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="Trustworthy AI Explainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FEEDBACK_CSV = os.getenv("MODULE15_FEEDBACK_CSV", "feedback_streamlit.csv")


# =============================================================================
# LLM BACKEND
# =============================================================================

class MockLLM:
    def generate(self, prompt: str, history: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500) -> str:
        time.sleep(0.4)
        return (
            "Mock response (no OPENAI_API_KEY detected).\n\n"
            f"Prompt: {prompt}\n\n"
            "Tip: define OPENAI_API_KEY in your environment or Streamlit secrets to enable real responses."
        )


@st.cache_resource
def load_llm():
    """
    Cached "resource": OpenAI client if API key exists, else MockLLM.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    # Streamlit Cloud secrets (optional)
    if not api_key and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = str(st.secrets["OPENAI_API_KEY"]).strip()

    if not api_key:
        return MockLLM()

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return MockLLM()


def build_messages(system_prompt: str, chat_history: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": system_prompt}]
    # keep last N messages to bound context
    trimmed = chat_history[-24:]
    msgs.extend(trimmed)
    msgs.append({"role": "user", "content": user_message})
    return msgs


def openai_chat_completion(client, messages: List[Dict[str, str]], temperature: float, max_tokens: int, stream: bool):
    """
    Returns either a string (non-stream) or an iterator of string chunks (stream).
    """
    if isinstance(client, MockLLM):
        full = client.generate(messages[-1]["content"], history=messages[:-1], temperature=temperature, max_tokens=max_tokens)
        if not stream:
            return full

        def gen():
            acc = ""
            for tok in full.split():
                acc += tok + " "
                time.sleep(0.02)
                yield acc
        return gen()

    if not stream:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content

    # stream=True
    stream_resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        stream=True,
    )

    def gen():
        acc = ""
        for evt in stream_resp:
            delta = evt.choices[0].delta.content or ""
            if delta:
                acc += delta
                yield acc
    return gen()


# =============================================================================
# TRUST / EXPLAINABILITY LAYER
# =============================================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def quality_features(input_text: str, output_text: str, temperature: float, turns: int) -> Dict[str, float]:
    itok = len((input_text or "").split())
    otok = len((output_text or "").split())
    has_steps = 1.0 if any(s in (output_text or "").lower() for s in ["1)", "1.", "- ", "step", "paso"]) else 0.0
    uncertainty = 1.0 if any(s in (output_text or "").lower() for s in ["i'm not sure", "no estoy seguro", "uncertain", "can't verify"]) else 0.0
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
    z = 0.0
    z += 1.2 * feat["has_steps"]
    z += -1.1 * feat["uncertainty"]
    z += -0.9 * feat["too_long"]
    z += -0.15 * max(0.0, feat["temperature"] - 0.7)
    z += 0.05 * min(feat["input_tokens"], 200.0) / 200.0
    z += -0.02 * max(0.0, feat["turns"] - 8.0)
    return float(sigmoid(z))


# ---- LIME: local token influence (text-only scorer) ----
_LIME_EXPLAINER = LimeTextExplainer(class_names=["low_quality", "high_quality"])


def _lime_predict_proba(texts: List[str]) -> np.ndarray:
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


@st.cache_data(show_spinner=False)
def lime_explain(input_text: str, num_features: int = 10) -> List[Tuple[str, float]]:
    exp = _LIME_EXPLAINER.explain_instance(
        input_text or "",
        _lime_predict_proba,
        num_features=int(num_features),
        labels=(1,),
    )
    return [(tok, float(w)) for tok, w in exp.as_list(label=1)]


# ---- SHAP: tabular feature contributions to quality score ----
@st.cache_data(show_spinner=False)
def shap_explain_tabular(feat: Dict[str, float]) -> List[Tuple[str, float]]:
    feature_names = list(feat.keys())
    x = np.array([[feat[k] for k in feature_names]], dtype=float)

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
    shap_vals = explainer.shap_values(x, nsamples=50)
    sv = np.array(shap_vals).reshape(-1)
    pairs = list(zip(feature_names, [float(v) for v in sv]))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs


def explanation_bundle(input_text: str, output_text: str, temperature: float, turns: int) -> Dict:
    feat = quality_features(input_text, output_text, temperature=temperature, turns=turns)
    q = quality_score_from_features(feat)
    lime_pairs = lime_explain(input_text)
    shap_pairs = shap_explain_tabular(feat)
    return {
        "quality_score": q,
        "features": feat,
        "lime": lime_pairs,
        "shap": shap_pairs,
    }


# =============================================================================
# FEEDBACK STORAGE
# =============================================================================

def append_feedback_csv(row: Dict[str, object], path: str = FEEDBACK_CSV) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


@st.cache_data(ttl=3600, show_spinner=False)
def load_feedback_df() -> pd.DataFrame:
    if "feedback_db" in st.session_state and st.session_state.feedback_db:
        return pd.DataFrame(st.session_state.feedback_db)
    return pd.DataFrame(columns=["timestamp", "message", "response", "rating", "comment", "model", "quality_score"])


def save_feedback(message: str, response: str, rating: str, comment: str, quality_score: float):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        "response": response,
        "rating": rating,
        "comment": (comment or "").strip(),
        "model": DEFAULT_MODEL,
        "quality_score": float(quality_score),
    }
    st.session_state.feedback_db.append(row)
    st.session_state.metrics["total_feedback"] += 1
    append_feedback_csv(row)
    load_feedback_df.clear()


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []  # list of {"role": "...", "content": "..."}
    if "feedback_db" not in st.session_state:
        st.session_state.feedback_db = []
    if "current_explanation" not in st.session_state:
        st.session_state.current_explanation = None
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {
            "temperature": 0.7,
            "max_tokens": 500,
            "system_prompt": (
                "You are a Trustworthy AI tutor. Explain concepts clearly and step-by-step. "
                "When unsure, say so. Avoid making up facts."
            ),
            "streaming": True,
        }
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_messages": 0,
            "avg_response_time": 0.0,
            "total_feedback": 0,
        }


# =============================================================================
# CORE CHAT FLOW
# =============================================================================

def generate_and_explain(user_text: str) -> Tuple[str, Dict, float]:
    client = load_llm()
    prefs = st.session_state.user_preferences
    messages = build_messages(
        prefs["system_prompt"],
        st.session_state.conversation_history,
        user_text,
    )

    start = time.time()
    stream_enabled = bool(prefs.get("streaming", True))

    if stream_enabled:
        gen = openai_chat_completion(client, messages, prefs["temperature"], prefs["max_tokens"], stream=True)
        # We'll render streaming in UI and also capture final text
        final = ""
        for partial in gen:
            final = partial
        response_text = final
    else:
        response_text = openai_chat_completion(client, messages, prefs["temperature"], prefs["max_tokens"], stream=False)

    elapsed = time.time() - start

    # update metrics
    st.session_state.metrics["total_messages"] += 1
    n = st.session_state.metrics["total_messages"]
    prev = st.session_state.metrics["avg_response_time"]
    st.session_state.metrics["avg_response_time"] = (prev * (n - 1) + elapsed) / n

    turns = max(1, sum(1 for m in st.session_state.conversation_history if m["role"] == "user") + 1)
    expl = explanation_bundle(user_text, response_text, temperature=prefs["temperature"], turns=turns)
    return response_text, expl, elapsed


# =============================================================================
# UI PAGES
# =============================================================================
# ============================================================================
# PAGE: CHAT INTERFACE
# ============================================================================

def page_chat():
    """Main chat interface page."""
    st.title("💬 Chat (LLM + Memory + Explainability)")
    st.markdown("Interact with the Trustworthy AI tutor. You will see the LIME/SHAP explanation and be able to give feedback.")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        prefs = st.session_state.user_preferences

        prefs["temperature"] = st.slider("Temperature", 0.0, 2.0, float(prefs["temperature"]), 0.1)
        prefs["max_tokens"] = st.slider("Max tokens", 50, 2000, int(prefs["max_tokens"]), 50)
        prefs["streaming"] = st.checkbox("Enable streaming", value=bool(prefs["streaming"]))
        with st.expander("System prompt"):
            prefs["system_prompt"] = st.text_area("System prompt", value=prefs["system_prompt"], height=120)

        st.divider()
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.current_explanation = None
            st.rerun()

    col_chat, col_exp = st.columns([2, 1], gap="large")

    with col_chat:
        st.subheader("Conversation")

        # Display history
        for msg in st.session_state.conversation_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_text := st.chat_input("Type your question..."):
            # Add user message
            st.session_state.conversation_history.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)

            # Generate assistant message
            try:
                if st.session_state.user_preferences.get("streaming", True):
                    # streaming render
                    client = load_llm()
                    prefs = st.session_state.user_preferences
                    messages = build_messages(prefs["system_prompt"], st.session_state.conversation_history[:-1], user_text)

                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        acc = ""

                        start = time.time()
                        gen = openai_chat_completion(client, messages, prefs["temperature"], prefs["max_tokens"], stream=True)
                        for partial in gen:
                            acc = partial
                            placeholder.markdown(acc)
                        elapsed = time.time() - start

                    response_text = acc

                    # update metrics
                    st.session_state.metrics["total_messages"] += 1
                    n = st.session_state.metrics["total_messages"]
                    prev = st.session_state.metrics["avg_response_time"]
                    st.session_state.metrics["avg_response_time"] = (prev * (n - 1) + elapsed) / n

                    turns = max(1, sum(1 for m in st.session_state.conversation_history if m["role"] == "user"))
                    expl = explanation_bundle(user_text, response_text, temperature=prefs["temperature"], turns=turns)
                else:
                    with st.spinner("Thinking..."):
                        response_text, expl, elapsed = generate_and_explain(user_text)
                    with st.chat_message("assistant"):
                        st.markdown(response_text)

                st.session_state.conversation_history.append({"role": "assistant", "content": response_text})
                st.session_state.current_explanation = {
                    "input": user_text,
                    "output": response_text,
                    "details": expl,
                }
                st.rerun()
            except Exception as e:
                st.error(f"Something went wrong while generating the response: {e}")

    with col_exp:
        st.subheader("Explainability + Feedback")
        if not st.session_state.current_explanation:
            st.info("Send a message to view explanations and provide feedback.")
            return

        exp = st.session_state.current_explanation["details"]
        q = exp["quality_score"]

        st.metric("Quality score (0–1)", f"{q:.3f}")
        f = exp["features"]
        st.caption(f"Signals: steps={int(f['has_steps'])}, uncertainty={int(f['uncertainty'])}, too_long={int(f['too_long'])}")

        st.divider()

        st.markdown("**SHAP (tabular) — top contributions**")
        shap_pairs = exp["shap"][:8]
        df_shap = pd.DataFrame(shap_pairs, columns=["feature", "contribution"])
        fig = px.bar(df_shap, x="contribution", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**LIME (local) — influential tokens (input)**")
        lime_pairs = exp["lime"][:10]
        for tok, w in lime_pairs:
            sign = "↑" if w >= 0 else "↓"
            st.write(f"- `{tok}`: {sign} {abs(w):.3f}")

        st.divider()

        st.markdown("**Feedback**")
        rating = st.radio("Rate the last response:", ["👍 Helpful", "👎 Not Helpful"], horizontal=True)
        comment = st.text_area("Optional comment", placeholder="What was good or could be improved?")

        if st.button("Submit feedback", use_container_width=True):
            try:
                save_feedback(
                    st.session_state.current_explanation["input"],
                    st.session_state.current_explanation["output"],
                    rating,
                    comment,
                    q,
                )
                st.success("✅ Feedback saved.")
            except Exception as e:
                st.error(f"Could not save feedback: {e}")

# ============================================================================
# PAGE: EXPLAINABILITY ANALYSIS
# ============================================================================

def page_explainability():
    """Detailed explainability analysis page."""
    st.title("🔍 Explainability (Deep Dive)")
    st.markdown("Select an interaction to analyze explanations and quality signals.")

    # Build pairs of (user, assistant)
    hist = st.session_state.conversation_history
    pairs = []
    i = 0
    while i < len(hist) - 1:
        if hist[i]["role"] == "user" and hist[i + 1]["role"] == "assistant":
            pairs.append((hist[i]["content"], hist[i + 1]["content"]))
            i += 2
        else:
            i += 1

    if not pairs:
        st.info("No complete user/assistant pairs yet. Use the Chat page first.")
        return

    idx = st.selectbox(
        "Conversation turn",
        options=list(range(len(pairs))),
        format_func=lambda k: f"Turn {k+1}: {pairs[k][0][:60]}...",
    )
    user_text, assistant_text = pairs[idx]

    exp = explanation_bundle(user_text, assistant_text, temperature=st.session_state.user_preferences["temperature"], turns=idx + 1)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("User input")
        st.code(user_text, language="text")
        st.subheader("Assistant output")
        st.code(assistant_text, language="text")

    with col2:
        st.subheader("Quality scoring")
        st.metric("Quality score", f"{exp['quality_score']:.3f}")
        st.json(exp["features"])

        st.subheader("SHAP contributions")
        df = pd.DataFrame(exp["shap"][:10], columns=["feature", "contribution"])
        fig = px.bar(df, x="contribution", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("LIME token influence (input)")
        for tok, w in exp["lime"][:12]:
            sign = "↑" if w >= 0 else "↓"
            st.write(f"- `{tok}`: {sign} {abs(w):.3f}")

# ============================================================================
# PAGE: FEEDBACK DASHBOARD
# ============================================================================

def page_feedback():
    """User feedback and quality monitoring page."""
    st.title("📊 Feedback Dashboard")
    df = load_feedback_df()

    if df.empty:
        st.info("No feedback yet. Provide feedback from the Chat page.")
        return

    # Summary
    col1, col2, col3 = st.columns(3)
    pos = int(df["rating"].astype(str).str.contains("👍").sum())
    neg = int(df["rating"].astype(str).str.contains("👎").sum())
    with col1:
        st.metric("Total feedback", len(df))
    with col2:
        st.metric("Positive", pos)
    with col3:
        st.metric("Negative", neg)

    st.divider()

    # Satisfaction
    sat = (pos / len(df)) * 100 if len(df) else 0
    st.metric("Satisfaction %", f"{sat:.1f}%")

    # Quality score distribution
    if "quality_score" in df.columns and df["quality_score"].notna().any():
        fig = px.histogram(df, x="quality_score")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # Recent table
    show = df.copy()
    show["timestamp"] = pd.to_datetime(show["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    show["message"] = show["message"].astype(str).str.slice(0, 80)
    show["response"] = show["response"].astype(str).str.slice(0, 80)
    st.dataframe(show[["timestamp", "rating", "quality_score", "message", "response", "comment"]], use_container_width=True, hide_index=True)

    # Export feedback
    st.divider()
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download feedback CSV", data=csv_data, file_name="feedback_export.csv", mime="text/csv")

# ============================================================================
# PAGE: MONITORING
# ============================================================================

def page_monitoring():
    """System monitoring and performance metrics page."""
    st.title("📈 Monitoring")
    m = st.session_state.metrics

    # Metrics overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total messages", m["total_messages"])
    with col2:
        st.metric("Avg response time (s)", f"{m['avg_response_time']:.2f}")
    with col3:
        st.metric("Total feedback", m["total_feedback"])

    st.divider()
    st.subheader("Cache controls")
    c1, c2 = st.columns(2)
    with c1:
        st.success("✅ Model client cached (@st.cache_resource)")
        st.info("ℹ️ Feedback cached (@st.cache_data, TTL=1h)")
    with c2:
        if st.button("Clear caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared.")
            st.rerun()

    st.divider()
    with st.expander("Session state (debug)"):
        st.json({
            "conversation_messages": len(st.session_state.conversation_history),
            "feedback_entries": len(st.session_state.feedback_db),
            "preferences": st.session_state.user_preferences,
            "metrics": st.session_state.metrics,
        })

# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================

def page_documentation():
    """Documentation page."""
    st.title("📚 Documentation")
    st.markdown(
        """
### What this app demonstrates (Module 15)
- **Streamlit multi-page dashboard** for an LLM app
- **Conversational memory** via `st.session_state.conversation_history`
- **Explainability** displayed alongside outputs:
  - **LIME** local token influence (user input)
  - **SHAP** tabular feature contributions (quality scorer)
- **Feedback** logging + analytics
- **Caching** and **latency management**
- **Security**: API keys via env vars / Streamlit secrets

### Environment variables
- `OPENAI_API_KEY` (required for real responses)
- `OPENAI_MODEL` (optional, default: gpt-4o-mini)
- `MODULE15_FEEDBACK_CSV` (optional)

### Deployment (recommended)
- Gradio → **Hugging Face Spaces**
- Streamlit → **Streamlit Community Cloud**
        """
    )

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main applicationentry point."""

    # Initialize session state
    init_state()

    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 Trustworthy AI")
        page = st.radio(
            "Navigation",
            ["💬 Chat", "🔍 Explainability", "📊 Feedback", "📈 Monitoring", "📚 Documentation"],
            label_visibility="collapsed",
        )
        st.caption(f"Session messages: {len(st.session_state.conversation_history)}")

    # Route to selected page
    if page == "💬 Chat":
        page_chat()
    elif page == "🔍 Explainability":
        page_explainability()
    elif page == "📊 Feedback":
        page_feedback()
    elif page == "📈 Monitoring":
        page_monitoring()
    else:
        page_documentation()


if __name__ == "__main__":
    main()
