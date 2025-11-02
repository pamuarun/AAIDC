# -*- coding: utf-8 -*-
"""
MEDIBOT - Medical Domain RAG Chatbot
With Fixed + Summary Memory + Strict Medical Q-Type + Semantic Evaluation
Includes:
    - Drug Info Agent (via OpenFDA API)
    - Body Metrics (BMI) Agent (Manual Calculation Only)
    - Diagnosis Agent (via PubMed + RAG reasoning + smart symptom trigger)
    - Lifestyle & Prevention Agent (WGER + LLM hybrid)
    - Image Agent (Gemini primary + Hugging Face Stable Diffusion XL backup)
LLM fallback enabled for questions with no FAISS context
Created: 2025-10-28
@author: Arun
"""

# ============================ #
# Step 0: Imports
# ============================ #
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from sklearn.metrics.pairwise import cosine_similarity
from langchain.docstore.document import Document
import csv, re, requests, os
from dotenv import load_dotenv

# Image agent imports
import google.generativeai as genai
from huggingface_hub import InferenceClient
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# ============================ #
# Step 1: Load Environment Keys
# ============================ #
# Use your project .env path (same as earlier)
ENV_PATH = r"D:\AAIDC\Project 2\.env"
load_dotenv(ENV_PATH)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WGER_API_KEY = os.getenv("WGER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")   # ensure this is in your .env

# ============================ #
# Step 2: Load FAISS DB + Embeddings
# ============================ #
DB_FAISS_PATH = r"D:\AAIDC\Project 2\vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 10})

# ============================ #
# Step 3: Gemini LLM
# ============================ #
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_output_tokens=1200
)

# ============================ #
# Step 4: Prompt Template
# ============================ #
FULL_PROMPT_TEMPLATE = """
You are MEDIBOT, a strict AI medical tutor and consultant.
Answer only medical-related questions based on the provided context if available.
If context is empty, fallback to general medical knowledge:
- Include recent studies, clinical trials, or research findings (up to your knowledge cutoff)
- Include clinical relevance, examples, and treatment implications
- Give approximate references or citations in readable format
Do NOT provide answers outside the medical domain.

---

### Memory & Context Rules:
- Use chat history to interpret vague follow-ups
- Continue the flow instead of repeating explanations
- Only answer medical questions; refuse irrelevant queries

---

### Depth Control:
- Expand answers into at least 4–5 lines
- Include key context, clinical reasoning, and examples
- Avoid any non-medical content

---

Chat History:
{chat_history}

Context from medical material (if any):
{context}

Patient / User Question:
{question}

Answer:
"""

FULL_PROMPT = PromptTemplate(
    template=FULL_PROMPT_TEMPLATE,
    input_variables=["chat_history", "context", "question"]
)

# ============================ #
# Step 5: Memory
# ============================ #
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

summary_memory = ""

def update_summary_memory(chat_history, max_recent_turns=5):
    global summary_memory
    if len(chat_history) <= max_recent_turns:
        return ""
    old_msgs = chat_history[:-max_recent_turns]
    summary_memory = " ".join([f"{m.type}: {m.content}" for m in old_msgs])
    return summary_memory

def trim_chat_history(chat_history, max_recent_turns=5):
    return chat_history[-max_recent_turns:]

def get_combined_history(chat_history, max_recent_turns=5):
    trimmed = trim_chat_history(chat_history, max_recent_turns)
    combined = ""
    if summary_memory:
        combined += f"[Summary of older conversation]: {summary_memory}\n"
    for msg in trimmed:
        combined += f"{msg.type}: {msg.content}\n"
    return combined

# ============================ #
# Step 6: Conversational Retrieval Chain
# ============================ #
medical_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": FULL_PROMPT},
    return_source_documents=False,
    output_key="answer"
)

# ============================ #
# Step 7: Semantic Evaluation
# ============================ #
def semantic_similarity_score(reference, generated, embed_model=embeddings):
    if not reference or not reference.strip() or not generated or not generated.strip():
        return None
    ref_vec = embed_model.embed_query(reference)
    gen_vec = embed_model.embed_query(generated)
    score = cosine_similarity([ref_vec], [gen_vec])[0][0]
    return round(score, 4)

def evaluate_response(reference, generated):
    return {"SemanticSim": semantic_similarity_score(reference, generated)}

# ============================ #
# Step 8: Medical Question Heuristic
# ============================ #
def is_medical_question(question):
    followups = r"\b(ok|okay|yes|continue|go with this|that one|steps in it|the 3rd one)\b"
    if re.search(followups, question.lower()):
        return True
    non_medical_patterns = [
        r"\b(joke|funny|politics|movie|celebrity|personal)\b",
        r"\b(who|where|when) is .* president\b"
    ]
    for pat in non_medical_patterns:
        if re.search(pat, question.lower()):
            return False
    return True

# ============================ #
# Step 9: Agents
# ============================ #

# 9.1 Drug Info Agent
OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"

def get_drug_info(drug_name):
    try:
        params = {"search": f"openfda.generic_name:{drug_name}", "limit": 1}
        response = requests.get(OPENFDA_ENDPOINT, params=params, timeout=10)
        if response.status_code != 200:
            return f"⚠️ Unable to fetch data from OpenFDA (Status: {response.status_code})"
        data = response.json()
        if "results" not in data or not data["results"]:
            return "⚠️ No official drug information found in OpenFDA."
        result = data["results"][0]
        openfda = result.get("openfda", {})
        brand = ", ".join(openfda.get("brand_name", ["N/A"]))
        generic = ", ".join(openfda.get("generic_name", ["N/A"]))
        indications = result.get("indications_and_usage", ["Not available"])[0]
        dosage = result.get("dosage_and_administration", ["Not available"])[0]
        warnings = result.get("warnings", ["Not available"])[0]
        contraindications = result.get("contraindications", ["Not available"])[0]
        return f"""💊 **Drug Information (via OpenFDA)**

**Brand & Generic Names:** {brand} | {generic}

**Indications (What it’s used for):** {indications}

**Dosage & Administration:** {dosage}

**Warnings:** {warnings}

**Contraindications:** {contraindications}

📚 *Source: FDA Drug Label Database (openFDA)*"""
    except Exception as e:
        return f"❌ Error fetching drug info: {e}"

def is_drug_query(query):
    match = re.search(r"tell me about\s+([A-Za-z0-9\-]+)", query.lower())
    if match:
        return match.group(1)
    return None

# 9.2 BMI Agent (Manual Only)
def is_bmi_query(query):
    patterns = [
        r"\bbmi\b",
        r"\bbody\s*mass\b",
        r"\b(height|tall).*(weight|weigh)\b",
        r"\b(weight|weigh).*(height|tall)\b",
        r"\b\d+\s*(cm|m).*\d+\s*kg\b"
    ]
    return any(re.search(p, query.lower()) for p in patterns)

def bmi_agent(query):
    query = query.lower()
    h_match = re.search(r"(\d+(\.\d+)?)\s*(cm|m)\b", query)
    w_match = re.search(r"(\d+(\.\d+)?)\s*kg\b", query)
    height = float(h_match.group(1)) / 100 if h_match and h_match.group(3) == "cm" else (float(h_match.group(1)) if h_match else None)
    weight = float(w_match.group(1)) if w_match else None

    if height and weight:
        bmi = round(weight / (height ** 2), 2)
        if bmi < 18.5:
            category, advice = "Underweight", "Increase calorie intake with nutrient-rich foods."
        elif 18.5 <= bmi < 24.9:
            category, advice = "Normal weight", "Maintain your balanced diet and activity."
        elif 25 <= bmi < 29.9:
            category, advice = "Overweight", "Exercise regularly and avoid processed foods."
        else:
            category, advice = "Obese", "Consult a healthcare provider for a structured plan."
        return f"""🧍 **Body Metrics Agent (BMI Report)**

**Height:** {height:.2f} m  
**Weight:** {weight:.1f} kg  
**BMI:** {bmi}  
**Category:** {category}  

💡 **Advice:** {advice}"""

    if re.search(r"what|how|mean|range|ideal|purpose", query):
        return """📘 **General Information on BMI**
BMI (Body Mass Index) estimates body fat based on height and weight.
Categories:
- Underweight: <18.5
- Normal: 18.5–24.9
- Overweight: 25–29.9
- Obese: ≥30"""
    return "🧍 Please provide height and weight clearly (e.g., 'My height is 170 cm and weight is 65 kg')."

# 9.3 Diagnosis Agent (PubMed + RAG + Smart Trigger)
def fetch_pubmed_articles(symptoms, max_results=5):
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base}esearch.fcgi"
        params = {"db": "pubmed", "term": symptoms, "retmax": max_results, "retmode": "json"}
        res = requests.get(search_url, params=params, timeout=10)
        data = res.json()
        ids = data["esearchresult"].get("idlist", [])
        if not ids:
            return []
        fetch_url = f"{base}efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "text", "rettype": "abstract"}
        fetch_res = requests.get(fetch_url, params=fetch_params, timeout=10)
        return fetch_res.text.split("\n\n")[:max_results]
    except Exception as e:
        return [f"⚠️ Error fetching PubMed results: {e}"]

def build_pubmed_vectorstore(abstracts):
    docs = [Document(page_content=a) for a in abstracts]
    db_temp = FAISS.from_documents(docs, embeddings)
    return db_temp.as_retriever(search_kwargs={"k": 5})

DIAGNOSIS_PROMPT = PromptTemplate(
    template="""
You are MEDIBOT's Diagnosis Agent.
Given the patient's symptoms and PubMed research, identify possible diagnoses.
Provide:
- Differential diagnoses (with reasoning)
- Related investigations or tests
- Mention relevant literature clues

Symptoms: {symptoms}

PubMed Research Context:
{context}

Diagnostic Summary:
""",
    input_variables=["symptoms", "context"]
)

def diagnosis_agent(symptoms):
    print("\n🩺 [Diagnosis Agent Activated]")
    abstracts = fetch_pubmed_articles(symptoms)
    if not abstracts:
        return "⚠️ No PubMed results found for these symptoms."
    retriever_pubmed = build_pubmed_vectorstore(abstracts)
    docs = retriever_pubmed.get_relevant_documents(symptoms)
    context = " ".join([d.page_content for d in docs])
    try:
        prompt = DIAGNOSIS_PROMPT.format(symptoms=symptoms, context=context)
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"❌ Error generating diagnosis: {e}"

# ✅ Enhanced Trigger Logic
def is_symptom_query(text):
    text = text.lower().strip()

    symptom_keywords = [
        "fever", "pain", "cough", "headache", "nausea", "vomit", "vomiting",
        "dizziness", "sore", "infection", "swelling", "fatigue", "cramps",
        "rash", "itch", "chills", "throat", "burning", "bleeding", "breath",
        "tightness", "diarrhea", "loss of taste", "loss of smell", "palpitation",
        "tremor", "sensitivity", "inflammation", "tingling", "sweating", "faint",
        "weakness", "pressure", "appetite", "stiffness", "cramp", "ache",
        "painful", "dizzy", "sick", "tired", "hurts"
    ]

    symptom_phrases = [
        r"\bi have\b", r"\bi’m having\b", r"\bim having\b", r"\bi am having\b",
        r"\bi feel\b", r"\bi’m feeling\b", r"\bim feeling\b", r"\bi am feeling\b",
        r"\bi got\b", r"\bi’ve got\b", r"\bi am suffering\b", r"\bi’m suffering\b",
        r"\bim suffering\b", r"\bi’ve been having\b", r"\bfeeling\b", r"\bhaving\b",
        r"\bsuffering from\b", r"\bmy\b"
    ]

    has_symptom = any(word in text for word in symptom_keywords)
    has_context = any(re.search(pattern, text) for pattern in symptom_phrases)
    return has_symptom and has_context

# ============================ #
# 9.4 Lifestyle & Prevention Agent (Exercise + Diet + Wellness) - SMART HYBRID FINAL
# ============================ #
import random

def lifestyle_agent(query, llm):
    """
    Smart Lifestyle & Wellness Agent (Dynamic + WGER Hybrid + Fallback)
    """
    intent_prompt = f"""
    You are an intent classifier for a wellness chatbot.
    Determine if the user's query is mainly about:
    1. Exercise / Fitness
    2. Diet / Nutrition
    3. General Lifestyle or Health Habit
    4. Combination of above

    Respond in one word only:
    - "exercise"
    - "diet"
    - "lifestyle"
    - "mixed"

    User query: {query}
    """
    intent_resp = llm.invoke(intent_prompt)
    intent = intent_resp.content.strip().lower() if hasattr(intent_resp, "content") else str(intent_resp).lower()

    system_prompt = """
    You are LIFEGEN, an expert AI Lifestyle & Wellness Coach.
    Provide structured, motivating answers on exercise, diet, and healthy living.
    Use friendly tone, emojis, and clear formatting.
    Output format:
    🧩 **Category:** (Exercise / Diet / Lifestyle / Combination)
    🧠 **Understanding:** (Short summary of user intent)
    ✅ **Personalized Plan:**
    - Tip 1
    - Tip 2
    - Tip 3
    💡 **Bonus Tip:** (Motivational or practical advice)
    """

    llm_response = llm.invoke(f"{system_prompt}\n\nUser Query: {query}")
    base_answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

    wger_text = ""
    headers = {"Accept": "application/json"}

    try:
        if "exercise" in intent or "mixed" in intent:
            res = requests.get("https://wger.de/api/v2/exercise/?language=2&limit=30", headers=headers, timeout=10)
            exercises = []
            if res.status_code == 200:
                data = res.json().get("results", [])
                exercises = [ex["name"] for ex in data if ex.get("name") and not ex["name"].startswith("UUID")]

            if not exercises:
                exercises = ["Jumping Jacks", "Plank Hold", "Squats", "Push-ups", "Lunges"]

            wger_text += "\n🏋️ **Example Exercises:**\n" + "\n".join(f"- {ex}" for ex in random.sample(exercises, min(5, len(exercises))))

        if "diet" in intent or "mixed" in intent:
            res = requests.get("https://wger.de/api/v2/ingredient/?limit=30", headers=headers, timeout=10)
            foods = []
            if res.status_code == 200:
                data = res.json().get("results", [])
                foods = [f["name"] for f in data if f.get("name") and not f["name"].startswith("UUID")]

            if not foods:
                foods = ["Oatmeal", "Greek Yogurt", "Brown Rice", "Broccoli", "Chicken Breast"]

            wger_text += "\n🥗 **Example Foods:**\n" + "\n".join(f"- {f}" for f in random.sample(foods, min(5, len(foods))))

    except Exception:
        fallback_exercises = ["Push-ups", "Plank", "Lunges", "Crunches", "Burpees"]
        fallback_foods = ["Quinoa", "Avocado", "Tofu", "Sweet Potato", "Salmon"]
        wger_text = "\n🏋️ **Example Exercises:**\n" + "\n".join(f"- {x}" for x in fallback_exercises)
        wger_text += "\n🥗 **Example Foods:**\n" + "\n".join(f"- {x}" for x in fallback_foods)

    final_output = f"{base_answer}\n{wger_text}"
    return final_output

# ============================ #
# 9.5 Medical Research Agent (Europe PMC + LLM)
# ============================ #
def research_agent(query, llm=None, memory=None):
    import requests

    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "pageSize": 5  # Fetch top 5
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return "❌ Failed to fetch research data. Please try again later."

    data = response.json()
    papers = data.get("resultList", {}).get("result", [])
    if not papers:
        return "⚕️ No research papers found for your query."

    research_summary = "🔬 **Top Medical Research Results:**\n\n"
    for i, paper in enumerate(papers, start=1):
        title = paper.get("title", "No title")
        authors = paper.get("authorString", "Unknown authors")
        year = paper.get("pubYear", "N/A")
        journal = paper.get("journalTitle", "N/A")
        link = f"https://europepmc.org/article/{paper.get('source', 'MED')}/{paper.get('id', '')}"

        research_summary += f"**{i}. {title}**\n"
        research_summary += f"👨‍⚕️ *{authors}* | 🏛️ {journal} ({year})\n"
        research_summary += f"🔗 [Read Paper]({link})\n\n"

    if llm:
        llm_prompt = f"""
        You are a medical research assistant. Summarize the following 5 papers concisely,
        highlighting key findings, innovations, and relevance to the query: "{query}".

        {research_summary}
        """
        try:
            llm_summary = llm.invoke(llm_prompt)
            final_output = (
                f"🧬 [Medical Research Agent Activated]\n\n"
                f"{research_summary}\n"
                f"🩺 **LLM Summary:**\n{llm_summary}"
            )
        except Exception as e:
            final_output = (
                f"🧬 [Medical Research Agent Activated]\n\n"
                f"{research_summary}\n"
                f"⚠️ Summary unavailable: {e}"
            )
    else:
        final_output = f"🧬 [Medical Research Agent Activated]\n\n{research_summary}"

    if memory is not None:
        memory["last_research"] = query
        memory["last_results"] = research_summary

    return final_output

# ============================ #
# Step 10: Debug Permissions
# ============================ #
def ask_permission(prompt_text):
    while True:
        resp = input(prompt_text + " (yes/no): ").strip().lower()
        if resp in ["yes", "y"]:
            return True
        elif resp in ["no", "n"]:
            return False
        else:
            print("Please answer with 'yes' or 'no'.")

DEBUG_MODE = ask_permission("Do you want to see retrieved context debug info?")
DEBUG_HISTORY = ask_permission("Do you want to see full chat history debug info?")

# ============================ #
# Step 12: Image Agent (Gemini + Hugging Face Backup) - Integration
# ============================ #
# Configure clients (Gemini + HF) — uses keys from .env
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"⚠️ Warning: genai.configure failed: {e}")

hf_client = None
if HF_TOKEN:
    try:
        hf_client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN)
    except Exception as e:
        print(f"⚠️ Warning: HF InferenceClient init failed: {e}")

# folder for saving images (timestamped)
IMAGES_ROOT = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", "images")

def image_agent(prompt):
    """
    Generate an educational image using:
    1. Google Gemini (primary)
    2. Hugging Face Stable Diffusion XL (backup if Gemini quota fails)
    Saves images in images/YYYYMMDD_HHMMSS/
    Returns a short textual result describing the saved file or an error.
    """
    # prepare folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(IMAGES_ROOT, ts)
    os.makedirs(folder, exist_ok=True)

    full_prompt = f"High-quality educational diagram for K-12 students: {prompt}, clear, neat, labelled where appropriate, simple color palette, high resolution"

    # ---------- Try Gemini First ----------
    try:
        print("🟢 Trying Gemini Image API...")
        model = genai.GenerativeModel("gemini-2.5-flash-preview-image")
        response = model.generate_content(full_prompt)

        saved_files = []
        for idx, candidate in enumerate(response.candidates):
            for part_idx, part in enumerate(candidate.content.parts):
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    img_data = inline.data
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                    fname = f"gemini_{idx}_{part_idx}.png"
                    path = os.path.join(folder, fname)
                    img.save(path)
                    saved_files.append(path)
        if saved_files:
            # display first image inline (matplotlib) for CLI users who have display
            try:
                im = Image.open(saved_files[0])
                plt.imshow(im)
                plt.axis("off")
                plt.show()
            except Exception:
                pass
            return f"[Image Agent] ✅ Gemini image(s) saved: {', '.join(saved_files)}"

    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")

    # ---------- Hugging Face Backup ----------
    try:
        if hf_client is None:
            raise RuntimeError("Hugging Face client not configured (HF_TOKEN missing or init failed).")
        print("🟡 Falling back to Hugging Face Stable Diffusion...")
        # Use HF Inference client text_to_image
        image = hf_client.text_to_image(prompt=full_prompt)
        # Some HF SDKs return PIL Image, some return bytes — try both
        if isinstance(image, Image.Image):
            img = image
        else:
            # If bytes-like, try to load
            img = Image.open(BytesIO(image)).convert("RGBA")

        fname = "hf_generated.png"
        path = os.path.join(folder, fname)
        img.save(path)
        try:
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        except Exception:
            pass
        return f"[Image Agent] ✅ Hugging Face image saved: {path}"
    except Exception as e:
        return f"[Image Agent] ❌ Both Gemini & Hugging Face failed. Error: {e}"

# ============================ #
# Step 11: Chat Loop (with Research Agent + Colors + Safe Logging)
# ============================ #
from rich import print  # For color-coded output
from rich.console import Console
console = Console()

print("[bold green]✅ MEDIBOT ready![/bold green] Type 'exit' or 'quit' to stop.\n")
memory_context = {}

with open("medibot_logs.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Query", "Agent", "Answer", "SemanticSim"])

    while True:
        query = input("\n🟢 Your question: ").strip()
        if query.lower() in ["exit", "quit", "bye"]:
            print("[bold yellow]👋 Goodbye! Stay healthy![/bold yellow]")
            break

        q_lower = query.lower()
        agent_name = "Unknown"

        # ---------------- Skip Diagnosis Agent for Lifestyle/Diet Questions ---------------- #
        skip_diagnosis = any(word in q_lower for word in [
            "diet", "eat", "food", "meal", "nutrition", "lifestyle",
            "exercise", "stretch", "workout", "fitness", "wellness",
            "habit", "prevention"
        ])

        # ---------------- Medical Research Agent ---------------- #
        if any(word in q_lower for word in [
            "research", "study", "journal", "paper", "publication",
            "findings", "article", "studies"
        ]):
            agent_name = "Research Agent"
            console.print("\n🧬 [bold cyan][Medical Research Agent Activated][/bold cyan]")
            answer = research_agent(query, llm=llm, memory=memory_context)
            console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")
            writer.writerow([datetime.now().isoformat(), query, agent_name, answer, None])
            continue

        # ---------------- Follow-Up for Research Agent ---------------- #
        if "last_research" in memory_context and any(word in q_lower for word in ["go with",
             "more", "related", "summarize", "details",
            "follow-up", "expand"
        ]):
            agent_name = "Research Follow-up"
            console.print("\n🔄 [bold cyan][Research Follow-Up Detected][/bold cyan]")
            followup_prompt = f"""
            The user previously searched for: "{memory_context['last_research']}".
            The following were the top research papers:
            {memory_context['last_results']}
            Now the user asks: "{query}".
            Provide a relevant follow-up answer with clinical reasoning.
            """
            try:
                answer = llm.invoke(followup_prompt)
            except Exception as e:
                answer = f"⚠️ LLM follow-up failed: {e}"

            console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")
            writer.writerow([datetime.now().isoformat(), query, agent_name, answer, None])
            continue

        # ---------------- Diagnosis Agent ---------------- #
        if not skip_diagnosis and is_symptom_query(query):
            agent_name = "Diagnosis Agent"
            console.print("\n🩺 [bold magenta][Diagnosis Agent Activated][/bold magenta]")
            answer = diagnosis_agent(query)
            console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")
            writer.writerow([datetime.now().isoformat(), query, agent_name, answer, None])
            continue

        # ---------------- Drug Agent ---------------- #
        drug_name = is_drug_query(query)
        if drug_name:
            agent_name = "Drug Info Agent"
            console.print("\n💊 [bold green][Drug Info Agent Activated][/bold green]")
            answer = get_drug_info(drug_name)
            console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")
            writer.writerow([datetime.now().isoformat(), query, agent_name, answer, None])
            continue

        # ---------------- BMI Agent ---------------- #
        if is_bmi_query(query):
            agent_name = "BMI Agent"
            console.print("\n⚖️ [bold yellow][BMI Agent Activated][/bold yellow]")
            answer = bmi_agent(query)
            console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")
            writer.writerow([datetime.now().isoformat(), query, agent_name, answer, None])
            continue

        # ---------------- Lifestyle & Prevention Agent ---------------- #
        if any(word in q_lower for word in [
            "exercise", "diet", "meal", "nutrition", "lifestyle", "fitness",
            "wellness", "habit", "prevention", "health", "food plan",
            "workout", "stretch"
        ]):
            agent_name = "Lifestyle Agent"
            console.print("\n🧘 [bold green][Lifestyle & Prevention Agent Activated][/bold green]")
            answer = lifestyle_agent(query, llm=llm)
            console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")
            writer.writerow([datetime.now().isoformat(), query, agent_name, answer, None])
            continue

        # ---------------- Image Agent ---------------- #
        if any(word in q_lower for word in ["sketch", "diagram", "draw", "generate", "image", "illustration", "figure"]):
            agent_name = "Image Agent"
            console.print("\n🖼️ [bold blue][Image Agent Activated][/bold blue]")
            try:
                img_resp = image_agent(query)
                answer = img_resp
            except Exception as e:
                answer = f"[Image Agent] ❌ Error: {e}"
            console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")
            writer.writerow([datetime.now().isoformat(), query, agent_name, answer, None])
            continue

        # ---------------- Non-Medical Filter ---------------- #
        if not is_medical_question(query):
            agent_name = "Non-Medical Filter"
            answer = "⚠️ This question is outside the medical domain. MEDIBOT cannot answer it."
            console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")
            writer.writerow([datetime.now().isoformat(), query, agent_name, answer, None])
            continue

        # ---------------- RAG Retrieval ---------------- #
        docs = retriever.get_relevant_documents(query)
        context_text = " ".join([doc.page_content for doc in docs]) if docs else ""
        full_history = memory.load_memory_variables({}).get("chat_history", [])
        update_summary_memory(full_history)
        chat_history_str = get_combined_history(full_history)

        # ---------------- LLM Fallback ---------------- #
        try:
            result = medical_chain({
                "question": query,
                "chat_history": chat_history_str
            })
            answer = result["answer"].strip()
            agent_name = "RAG Chain"
        except Exception as e:
            console.print(f"[bold red]❌ Error generating answer:[/bold red] {e}")
            continue

        # ---------------- Debug Outputs ---------------- #
        console.print(f"\n🟢 [bold cyan]Question:[/bold cyan]\n{query}")
        if DEBUG_MODE:
            console.print("\n🔎 [bold yellow][DEBUG] Retrieved Context (first 300 chars):[/bold yellow]")
            console.print(context_text[:300] + "..." if context_text else "⚠️ No context retrieved")

        console.print(f"\n💬 [bold white]Answer:[/bold white]\n{answer}")

        if DEBUG_HISTORY:
            console.print("\n📚 [bold magenta][DEBUG] Full Chat History:[/bold magenta]")
            for i, msg in enumerate(full_history, 1):
                console.print(f"{i}. {msg.type}: {msg.content}")

        # ---------------- Evaluation ---------------- #
        scores = evaluate_response(context_text, answer)
        console.print(f"\n📊 [bold green]Semantic Similarity:[/bold green] {scores['SemanticSim']}")
        writer.writerow([datetime.now().isoformat(), query, agent_name, answer, scores.get("SemanticSim")])
