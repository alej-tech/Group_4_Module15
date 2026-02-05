# Module 15 - Group 4: Trustworthy AI Explainer Dashboard

## Team Project - App Prototyping with Streamlit

This project demonstrates building user interfaces for LLM applications using Gradio and Streamlit, with integrated explainability and feedback mechanisms.

## Project Structure

```
module-15/project/
├── module-15-project.md              # Project specification
├── Group4_Module15_Project.ipynb     # Learning notebook with examples
├── Group4_Module15_GradioApp.py      # Gradio prototype (fully functional)
├── Group4_Module15_StreamlitApp.py   # Streamlit dashboard (multi-page)
├── requirements.txt                  # Python dependencies
├── README_Group4.md                  # This file
└── deployment_urls.txt               # links to deployed applications
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Gradio Prototype

```bash
python Group4_Module15_GradioApp.py
```

Opens at: http://localhost:7860

### 3. Run Streamlit Dashboard

```bash
streamlit run Group4_Module15_StreamlitApp.py
```

Opens at: http://localhost:8501

## Features

## Group4 Customization (No RAG)

- **Use case**: Tutor/Explainer chatbot for Trustworthy AI concepts.
- **No RAG**: Chat-only with conversational memory.
- **Explainability**: LIME (local token influence on input) + SHAP (tabular feature contributions to a lightweight quality score).
- **Feedback**: Stored in-session and persisted to CSV (`feedback_gradio.csv`, `feedback_streamlit.csv`).


### Gradio Prototype
- Simple chatbot interface
- Conversation memory
- Explainability display
- User feedback (thumbs up/down)
- Ready for Hugging Face Spaces deployment

### Streamlit Dashboard
- **Multi-page application:**
  - Chat: Interactive LLM conversation
  - Explainability: Detailed analysis
  - Feedback: User feedback analytics
  - Monitoring: Performance metrics
  - Documentation: App information
- State management for conversation history
- Caching for expensive operations
- Real-time visualizations
- Feedback collection and analytics

## Configuration

### LLM Integration

Both templates use a mock LLM by default. To integrate a real LLM:

**Option 1: OpenAI**
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**Option 2: AWS Bedrock**
```python
import boto3
from langchain_aws import ChatBedrock
```

**Option 3: Local (Ollama)**
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1")
```

**Option 4: Hugging Face Transformers**
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="model-name")
```

### API Keys

**Never hardcode API keys!**

**Method 1: Environment Variables**
```bash
export OPENAI_API_KEY="your-key-here"
```

**Method 2: .env File**
```
OPENAI_API_KEY=your-key-here
```

**Method 3: Streamlit Secrets** (for deployment)
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your-key-here"
```

## Customization

### 1. Customize Gradio App

Edit `Group4_Module15_GradioApp.py`:
- Replace `MockLLM()` with your LLM
- Implement actual explainability in `compute_explanation()`
- Add feedback storage in `handle_feedback()`
- Customize UI theme and layout

### 2. Customize Streamlit App

Edit `Group4_Module15_StreamlitApp.py`:
- Replace `MockLLM()` in `load_llm()`
- Implement SHAP/LIME in `compute_explanation()`
- Connect feedback to database
- Add new pages as needed
- Customize visualizations

## Deployment

### Deploy Gradio to Hugging Face Spaces

```bash
# 1. Create new Space on huggingface.co
# 2. Choose "Gradio" as SDK
# 3. Push your code

git init
git add Group4_Module15_GradioApp.py requirements.txt
git commit -m "Initial commit"
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
git push hf main
```

**Remember:** Add API keys in Space Settings → Repository secrets

### Deploy Streamlit to Streamlit Cloud

```bash
# 1. Push to GitHub
git init
git add Group4_Module15_StreamlitApp.py requirements.txt
git commit -m "Initial commit"
git push origin main

# 2. Go to share.streamlit.io
# 3. Connect your GitHub repo
# 4. Deploy!
```

**Remember:** Add API keys in App Settings → Secrets

### Deploy with Docker

**Dockerfile for Gradio:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY Group4_Module15_GradioApp.py .
EXPOSE 7860
CMD ["python", "Group4_Module15_GradioApp.py"]
```

**Dockerfile for Streamlit:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY Group4_Module15_StreamlitApp.py .
EXPOSE 8501
CMD ["streamlit", "run", "Group4_Module15_StreamlitApp.py"]
```

**Build and run:**
```bash
docker build -t ai-dashboard .
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY ai-dashboard
```

## Explainability Integration

### SHAP Implementation

```python
import shap

@st.cache_data
def compute_shap_explanation(text, model):
    explainer = shap.KernelExplainer(model, background_data)
    shap_values = explainer.shap_values(text)
    return shap_values
```

### LIME Implementation

```python
from lime.lime_text import LimeTextExplainer

def compute_lime_explanation(text, model):
    explainer = LimeTextExplainer()
    exp = explainer.explain_instance(text, model.predict)
    return exp
```

## Troubleshooting

### Gradio Issues

**Port already in use:**
```python
demo.launch(server_port=7861)  # Use different port
```

**Sharing issues:**
```python
demo.launch(share=True)  # Creates public link
```

### Streamlit Issues

**Port already in use:**
```bash
streamlit run Group4_Module15_StreamlitApp.py --server.port 8502
```

**Caching not working:**
```python
# Clear caches
st.cache_data.clear()
st.cache_resource.clear()
```

**State not persisting:**
```python
# Check initialization
if 'key' not in st.session_state:
    st.session_state.key = default_value
```

## Team Deliverables

Submit the following files:

1. `TeamName_Module15_Project.ipynb` - Completed notebook
2. `TeamName_Module15_GradioApp.py` - Customized Gradio app
3. `TeamName_Module15_StreamlitApp.py` - Customized Streamlit app
4. `requirements.txt` - All dependencies
5. `README.md` - Updated with your customizations
6. `deployment_urls.txt` - Links to deployed applications

## Resources

**Documentation:**
- [Gradio Docs](https://www.gradio.app/docs)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Streamlit Cloud](https://streamlit.io/cloud)

**Explainability:**
- [SHAP](https://shap.readthedocs.io/)
- [LIME](https://lime-ml.readthedocs.io/)

**LLM Integration:**
- [OpenAI API](https://platform.openai.com/docs)
- [LangChain](https://python.langchain.com/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## Support

For questions or issues:
1. Check the documentation in the notebook
2. Review the comments in template files
3. Consult the module scripts
4. Ask your team members
5. Reach out to instructors

---

**Module 15 - App Prototyping with Streamlit**

**Good luck with your Trustworthy AI Explainer Dashboard!**

