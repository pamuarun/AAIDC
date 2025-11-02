# 🩺 **MEDIBOT — Multi-Agent AI Medical Assistant**

**MEDIBOT** is an advanced **multi-agent AI chatbot** built for the **medical domain**, combining **Retrieval-Augmented Generation (RAG)**, **Google Gemini LLM**, and multiple **domain-specific APIs**.  
It delivers clinically relevant, context-aware responses across drug data, diagnosis, wellness, and research insights.

---

## 🚀 **Project Overview**
MEDIBOT integrates **LangChain**, **Gemini**, **FAISS**, and **HuggingFace embeddings** to provide intelligent, medically accurate conversations.  
It uses an agentic workflow — each agent performs a specialized medical task, ensuring precision, safety, and adaptability.

---

### 🧩 **Core Features**
- 💊 **Drug Info Agent** — Fetches FDA-verified drug details  
- ⚖️ **BMI Agent** — Calculates BMI with personalized health guidance  
- 🩺 **Diagnosis Agent** — Identifies possible diseases via PubMed + RAG  
- 🧘 **Lifestyle Agent** — Generates fitness & diet plans (WGER + Gemini)  
- 🧬 **Research Agent** — Retrieves & summarizes latest EuropePMC studies  
- 🖼️ **Image Agent** — Creates educational medical diagrams via Gemini / HF  

---

### 🧠 **Architecture**
![Architecture Diagram](docs/architecture.png)

**Flow:**  
User Input → Intent Detection → Specialized Agent → LLM (Gemini) → Semantic Evaluation → Output + CSV Logging  

---

### ⚙️ **Tech Stack**
**LLM:** Google Gemini 
**Framework:** LangChain, Langgraph  
**Embeddings:** HuggingFace MiniLM 
**Vector DB:** FAISS  
**APIs:** OpenFDA, PubMed, WGER, EuropePMC  
**Visualization:** Matplotlib, Pillow, Rich CLI  

---

### 📊 **Highlights**
- Multi-agent orchestration with memory & semantic evaluation  
- API-driven RAG design for accuracy & transparency  
- Auto-logging and performance tracking (MSE, semantic similarity)  
- Lightweight, extensible, and ready for deployment  

---

### 🧾 **Performance & Metrics**
- ⚡ Avg. Response Time: 1–3 seconds  
- 📊 Semantic Similarity: ≥ 0.85 (typical)  
- 🧠 Memory Trim & Summary: 5-turn rolling window  

---

### 🪪 **License**
MIT License © 2025 **Arun Teja**

---

### 🙌 **Acknowledgements**
Google Gemini • LangChain • Hugging Face • OpenFDA • PubMed • EuropePMC • WGER API
