# Module 15 – Group_4  
## Trustworthy AI Explainer Dashboard  
**Interactive LLM Tutor with Explainability (Gradio + Streamlit)**

---

## Institution
**CITEDI**

## Team
**Group_4**

## Date
**05 February 2026**

---

## Project Overview

This project implements a **Trustworthy AI Explainer Dashboard**, an interactive LLM-based tutor designed to explain concepts related to **Trustworthy / Responsible AI** while simultaneously exposing its own decision-making behavior through explainability techniques.

The system combines:
- A conversational LLM with memory
- Global and local explainability (SHAP + LIME)
- User feedback logging
- Two complementary user interfaces:
  - **Gradio** (rapid prototyping)
  - **Streamlit** (structured dashboard)

The application is intentionally implemented **without RAG**, focusing on conversational reasoning and transparency rather than document retrieval.

---

## Use Case

**Tutor / Explainer chatbot for Trustworthy AI**, supporting learning across previous course modules (prompt engineering, evaluation, bias, transparency, and monitoring).

Target users:
- University students
- Academic staff
- AI practitioners interested in trustworthy systems

---

## Project Structure


```
Group_4_Module15/
├── notebook/
│ └── Group_4_Module15_Project.ipynb
├── apps/
│ ├── gradio/
│ │ └── Group4_Module15_GradioApp.py
│ └── streamlit/
│ └── Group4_Module15_StreamlitApp.py
├── requirements.txt
├── deployment_urls.txt
└── README_Group4.md
```


---

## Features

### Core Capabilities
- LLM-based conversational chatbot with multi-turn memory
- Trustworthy AI tutoring and explanations
- Global explainability with **SHAP**
- Local input-level explainability with **LIME**
- User feedback collection (stored as CSV)
- Secure API key handling (no hardcoded secrets)

---

### Gradio Prototype (Hugging Face Spaces)
- Lightweight conversational interface
- Memory-aware chat
- Integrated explainability panel
- Designed for rapid experimentation and demos

---

### Streamlit Dashboard (Community Cloud)
- Multi-page structured dashboard
- Session state management
- Explainability visualization
- Feedback logging and monitoring
- Cached resources for performance

---

## Deployment

### Gradio – Hugging Face Spaces
Public URL:  
- https://huggingface.co/spaces/alej-hugg/group-4-module15-gradio

Secrets are managed via **Hugging Face Space Settings → Secrets**.

---

### Streamlit – Streamlit Community Cloud
Public URL:  
- https://group-4-module15-strmlit.streamlit.app/

Secrets are managed via **App Settings → Secrets** using TOML format:
```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4o-mini"
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Gradio Prototype

```bash
python apps/gradio/Group4_Module15_GradioApp.py
```

Opens at: http://localhost:7860

### 3. Run Streamlit Dashboard

```bash
streamlit run apps/streamlit/Group4_Module15_StreamlitApp.py
```

Opens at: http://localhost:8501

## Explainability Design

## SHAP (global):
Explains contributions of high-level features (steps, uncertainty, length, turns) to a quality proxy score.

## LIME (local):
Highlights influential tokens in the user input for a given response.

This dual approach supports both system-level transparency and per-interaction interpretability.


## Team Roles and Responsibilities

## Member A — System Architect & LLM Integration
- Overall system design
- LLM integration and memory handling

## Member B — Explainability & Evaluation Lead
- SHAP and LIME implementation
- Evaluation signals and testing

## Member C — Gradio Prototype Developer
- Gradio interface design
- Hugging Face deployment

## Member D — Streamlit Dashboard Architect
- Multi-page dashboard implementation
- State management and caching
- Streamlit Cloud deployment

## Member E — Deployment & Documentation
- GitHub repository management
- Secret configuration
- Documentation and final submission

---

## Team Deliverables

Submit the following files:

1. `Group4_Module15_Project.ipynb` - Completed notebook
2. `Group4_Module15_GradioApp.py` - Customized Gradio app
3. `Group4_Module15_StreamlitApp.py` - Customized Streamlit app
4. `requirements.txt` - All dependencies
5. `README_Group4.md` - Updated with your customizations
6. `deployment_urls.txt` - Links to deployed applications

## Conclusion
This project demonstrates how **Trustworthy AI principles** can be operationalized in LLM-based systems through explainability, feedback, and transparent deployment. The dual-interface approach highlights trade-offs between rapid prototyping and structured dashboards while maintaining consistent backend logic.
