# 🩺 **MEDIBOT — Multi-Agent AI Medical Assistant**

**MEDIBOT** is a next-generation **multi-agent AI system** built for the **medical domain**, designed to act as an intelligent, domain-aware virtual health consultant.  
It combines **Retrieval-Augmented Generation (RAG)** with advanced **Large Language Models (Google Gemini)** and **LangChain orchestration** to ensure reliable, factual, and safe medical communication.  

Unlike generic chatbots, MEDIBOT operates through an **agentic workflow**, where each specialized agent performs a defined clinical or analytical role — such as diagnosis, lifestyle guidance, or drug information retrieval.  
This distributed design ensures **precision**, **transparency**, and **scalability**, making MEDIBOT a highly adaptable platform for both **educational and professional healthcare applications**.  

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
![Architecture Diagram](docs/Architecture.png)

**Flow Summary:**  
User Input → Intent Detection → Specialized Agent → LLM (Gemini) → Semantic Evaluation → Output + Logging  

Each agent interacts independently with external APIs or internal retrievers and sends results through the **Gemini reasoning layer**, ensuring accuracy, factual grounding, and clarity in responses.

---

### ⚙️ **Tech Stack**
**LLM:** Google Gemini  
**Framework:** LangChain, LangGraph  
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

## 🪪 **License**

This project is **copyright © 2025 Arun Teja**.  
All rights reserved. Unauthorized copying, modification, or distribution of this software without prior permission is strictly prohibited.  

This project was developed as part of the **AAIDC Module 2 Certification Program**.

---

### 🙌 **Acknowledgements**
- **LangChain / LangGraph** — For providing the orchestration framework that enables smooth multi-agent workflows.  
- **AAIDC Program** — For defining the project requirements, guidance, and structured approach to development.  
- **Google Gemini** — For powering the advanced LLM reasoning and contextual understanding.  
- **Hugging Face** — For embeddings and vector representation support.  
- **OpenFDA API** — For accurate drug information retrieval.  
- **PubMed & EuropePMC** — For evidence-based research and clinical reference data.  
- **WGER API** — For lifestyle, fitness, and diet planning integration.
