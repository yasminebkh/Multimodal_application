import tkinter as tk
import tkinter.font as tkFont
import sounddevice as sd
import whisper
import requests
import threading
import numpy as np
import soundfile as sf
import noisereduce as nr
import re
import sys
from gtts import gTTS
import pygame
import os
import tempfile
import time # Ajout pour la mesure de la latence

# --- CONFIG ---
RATE = 16000
MAX_QUESTIONS = 6

# Initialisation
pygame.mixer.init()
# Chargez votre modèle Whisper (attention à la latence du modèle "medium")
model = whisper.load_model("medium")

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a clinical interviewer strictly following the PQRST framework:
- **P**rovoked: What triggers or worsens the symptom?
- **Q**uality: How does the symptom feel? (e.g., sharp, dull, burning)
- **R**egion & Radiation: Where is it located? Does it spread?
- **S**everity: Rate from 0 (none) to 10 (worst imaginable)
- **T**iming: When did it start? How long does it last? Is it constant?

RULES:
1. Ask **ONE** PQRST-related question per turn.
2. **NEVER** give advice, diagnosis, or explanations.
3. **DO NOT** repeat questions already answered.
4. If all PQRST elements are covered, respond **exactly**: [SUMMARY]
5. Use **simple, clear English** suitable for elderly or illiterate patients.

Begin the medical interview now."""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
patient_data = {k: None for k in ["name", "age", "marital_status", "children", "children_ages",
                                 "operations", "operation_details", "chronic_diseases", "chronic_disease_details"]}
conversation_phase = "personal_info"
question_counter = 0
medical_questions_asked = set()
is_speaking = False

# --- NOUVELLE SECTION: METRIQUES D'ÉVALUATION AVEC ASR ET LLM ---
evaluation_metrics = {
    "latencies": [],
    "asr_latencies": [],      # ⬅️ Nouvelle métrique pour Whisper
    "llm_latencies": [],      # ⬅️ Nouvelle métrique pour LLM
    "llm_questions_valid": 0,
    "llm_questions_fallback": 0
}
# ------------------------------------------------------------------


# --- TTS ---
def speak_text(text):
    global is_speaking
    clean_text = re.sub(r'[🎤📄✅🛑🩺●]', '', text).strip()
    if not clean_text or "[SUMMARY_REQUESTED]" in text:
        return
    
    try:
        is_speaking = True
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
        
        gTTS(text=clean_text, lang='en', slow=False).save(temp_file)
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.unload()
        try: os.remove(temp_file)
        except: pass
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        is_speaking = False

# --- AUDIO ---
def record_audio(duration=8):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * RATE), samplerate=RATE, channels=1)
    sd.wait()
    audio = nr.reduce_noise(y=audio.flatten().astype(np.float32), sr=RATE)
    filename = "patient.wav"
    sf.write(filename, audio, RATE)
    print("✅ Audio saved")
    return filename

# --- EXTRACTION (non modifié) ---
def extract_numbers(text):
    return re.findall(r'\d+', text)

def extract_age(text):
    nums = extract_numbers(text)
    return nums[0] if nums else text.strip(".").strip()

def extract_children(text):
    nums = extract_numbers(text)
    if nums:
        return nums[0]
    
    words = {"one":1,"two":2,"three":3,"four":4,"five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10}
    for word, num in words.items():
        if word in text.lower().strip('.'): 
            return str(num)
    return None

# --- PHASE 1: PERSONAL INFO (non modifié) ---
PERSONAL_QUESTIONS = {
    "name": "What is your current age?",
    "age": "Are you currently single or married?",
    "marital_single": "Have you ever had any operations?",
    "marital_married": "Do you have children? How many?",
    "children_yes": "How old are they?",
    "children_no": "Have you ever had any operations?",
    "children_ages": "Have you ever had any operations?",
    "operations_yes": "What kind of operation(s) did you have?",
    "operations_no": "Do you have any chronic diseases?",
    "operation_details": "Do you have any chronic diseases?",
    "chronic_yes": "What chronic disease(s) do you have?",
    "chronic_no": "Now, let's talk about your current health. How are you feeling today?"
}

def get_personal_info_question(patient_text):
    global patient_data, conversation_phase, question_counter
    
    if not patient_data["name"]:
        name_match = re.search(r'(?:my name is|i am)\s+([a-zA-Z\s]+)', patient_text.lower())
        patient_data["name"] = name_match.group(1).strip() if name_match else patient_text.strip(".")
        return PERSONAL_QUESTIONS["name"]
    
    if not patient_data["age"]:
        patient_data["age"] = extract_age(patient_text)
        return PERSONAL_QUESTIONS["age"]
    
    if not patient_data["marital_status"]:
        status = "Married" if "married" in patient_text.lower() else "Single"
        patient_data["marital_status"] = status
        return PERSONAL_QUESTIONS["marital_married"] if "married" in status.lower() else PERSONAL_QUESTIONS["marital_single"]
    
    if "married" in patient_data["marital_status"].lower() and not patient_data["children"]:
        children = extract_children(patient_text)
        if children is not None or "yes" in patient_text.lower():
            patient_data["children"] = children or "Yes"
            return PERSONAL_QUESTIONS["children_yes"]
        
        patient_data["children"] = "No"
        return PERSONAL_QUESTIONS["children_no"]
        
    
    if patient_data["children"] and patient_data["children"] != "No" and not patient_data["children_ages"]:
        patient_data["children_ages"] = patient_text.rstrip('.?!').strip()
        return PERSONAL_QUESTIONS["children_ages"]
    
    if patient_data["operations"] is None:
        if "yes" in patient_text.lower() or "operation" in patient_text.lower():
            patient_data["operations"] = "yes"
            return PERSONAL_QUESTIONS["operations_yes"]
        patient_data["operations"] = "no"
        return PERSONAL_QUESTIONS["operations_no"]
    
    if patient_data["operations"] == "yes" and not patient_data["operation_details"]:
        patient_data["operation_details"] = re.sub(r'i (had|was)', '', patient_text, flags=re.I).strip(".")
        return PERSONAL_QUESTIONS["operation_details"]
    
    if patient_data["chronic_diseases"] is None:
        text = patient_text.lower()

        if "yes" in text:
            patient_data["chronic_diseases"] = "yes"
            return PERSONAL_QUESTIONS["chronic_yes"]

        if "no" in text or "don't" in text or "do not" in text or "none" in text:
            patient_data["chronic_diseases"] = "no"
            conversation_phase = "medical_consultation"
            question_counter = 0
            return PERSONAL_QUESTIONS["chronic_no"]

        # Si la réponse n'est pas claire
        return "Please answer yes or no. Do you have any chronic diseases?"

    # Si on est ici, tout est rempli → on passe en consultation
    conversation_phase = "medical_consultation"
    return "How are you feeling today?"

# --- PHASE 2: MEDICAL CONSULTATION (non modifié) ---
PQRST_FALLBACK = [
    "Can you describe what the pain feels like?",
    "Where exactly do you feel this discomfort?",
    "How would you rate the intensity?",
    "When did you first notice these symptoms?",
    "What seems to trigger or worsen it?",
    "Does anything help relieve the symptoms?"
]

def validate_question(response):
    """Valide si la question est complète et bien formée"""
    words = response.split()
    if len(words) < 4:
        return False
    
    bad_patterns = [
        r'^(and|or|but|so)\s+\w+\?$',
        r'^\w{1,3}\s*\?$',
        r'^(caused by|spreads to|how long|when|where)\s*\?$'
    ]
    
    return not any(re.search(p, response.lower()) for p in bad_patterns)

def generate_medical_question(messages):
    global medical_questions_asked, question_counter

    if messages and messages[-1]["role"] == "assistant":
        content = messages[-1]["content"].strip()
        if content and content != "[SUMMARY_REQUESTED]":
            medical_questions_asked.add(content)

    # Filtrer les messages utiles
    filtered = [m for m in messages if m["role"] in ("system", "user", "assistant")]

    # Ajouter un rappel des questions déjà posées (optionnel mais utile)
    if medical_questions_asked:
        history_note = "Already asked: " + "; ".join(medical_questions_asked) + "\n"
        # Injecter dans le dernier message système ou en créer un
        if filtered and filtered[0]["role"] == "system":
            filtered[0]["content"] = SYSTEM_PROMPT + "\n\n" + history_note
        else:
            filtered.insert(0, {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + history_note})
    else:
        if not (filtered and filtered[0]["role"] == "system"):
            filtered.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    try:
        # Appel à llama.cpp (API OpenAI-compatible)
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "mistral-7b-instruct-v0.2",  # peut être ignoré par llama.cpp
                "messages": filtered,
                "temperature": 0.3,        # Réduit l'aléatoire → moins d'hallucinations
                "max_tokens": 80,          # Limite la réponse à une question courte
                "stop": ["\n", "."]        # Force l'arrêt après une phrase
            },
            timeout=20  # Important pour éviter les freezes
        )
        response.raise_for_status()
        llm_response = response.json()["choices"][0]["message"]["content"].strip()

        # Nettoyage
        llm_response = re.sub(r"^(Doctor:|Interviewer:)\s*", "", llm_response, flags=re.I)
        llm_response = llm_response.split("\n")[0].strip()
        if not llm_response.endswith("?"):
            llm_response = llm_response.rstrip(" .") + "?"

        if "[SUMMARY]" in llm_response:
            return "[SUMMARY_REQUESTED]"

        if validate_question(llm_response):
            medical_questions_asked.add(llm_response)
            evaluation_metrics["llm_questions_valid"] += 1
            return llm_response
        else:
            evaluation_metrics["llm_questions_fallback"] += 1
            return PQRST_FALLBACK[question_counter % len(PQRST_FALLBACK)]

    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        evaluation_metrics["llm_questions_fallback"] += 1
        return PQRST_FALLBACK[question_counter % len(PQRST_FALLBACK)]

# --- MAIN PROCESS AVEC CHRONOMÉTRAGE DÉTAILLÉ ---
def process_audio_thread():
    global conversation_phase, question_counter
    
    while is_speaking:
        root.after(100)
    
    record_button.config(text="● Recording...", bg="#d9534f", state="disabled")
    report_button.config(state="disabled")
    root.update()
    
    # ----------------------------------------------------
    # DÉBUT DU CHRONOMÉTRAGE TOTAL
    start_time_total = time.time() 
    # ----------------------------------------------------
    
    filename = record_audio()
    
    # ----------------------------------------------------
    # 1. DÉBUT DU CHRONOMÉTRAGE ASR (Whisper)
    start_time_asr = time.time()
    # ----------------------------------------------------
    
    patient_text = model.transcribe(filename, language="en")["text"].strip()
    
    # ----------------------------------------------------
    # 1. FIN DU CHRONOMÉTRAGE ASR
    end_time_asr = time.time()
    evaluation_metrics["asr_latencies"].append(end_time_asr - start_time_asr)
    print(f"ASR Latency: {end_time_asr - start_time_asr:.2f} s")
    # ----------------------------------------------------
    
    print(f"\nPatient: {patient_text}")
    messages.append({"role": "user", "content": patient_text})
    
    doctor_text = ""
    
    if conversation_phase == "personal_info":
        # Traitement rapide de la logique if/else (pas de LLM, pas de chronométrage spécifique)
        doctor_text = get_personal_info_question(patient_text)
    else:
        if question_counter >= MAX_QUESTIONS:
            doctor_text = "[SUMMARY_REQUESTED]"
        else:
            # ----------------------------------------------------
            # 2. DÉBUT DU CHRONOMÉTRAGE LLM (Seulement pour les requêtes à Mistral)
            start_time_llm = time.time()
            # ----------------------------------------------------
            
            doctor_text = generate_medical_question(messages)
            
            # ----------------------------------------------------
            # 2. FIN DU CHRONOMÉTRAGE LLM
            end_time_llm = time.time()
            evaluation_metrics["llm_latencies"].append(end_time_llm - start_time_llm)
            print(f"LLM Latency: {end_time_llm - start_time_llm:.2f} s")
            # ----------------------------------------------------

            if doctor_text != "[SUMMARY_REQUESTED]":
                question_counter += 1
    
    # ----------------------------------------------------
    # FIN DU CHRONOMÉTRAGE TOTAL
    end_time_total = time.time()
    evaluation_metrics["latencies"].append(end_time_total - start_time_total) 
    # ----------------------------------------------------

    if doctor_text == "[SUMMARY_REQUESTED]":
        print("\n🛑 Consultation concluded")
        generate_summary()  # Génère 'patient_medical_report.txt'
        final_msg = display_evaluation()  # Génère 'system_evaluation.txt'
        # ✅ Message clair pour l'utilisateur
        response_label.config(
            text="✅ Consultation complete!\n"
        )
        threading.Thread(target=speak_text, args=(
            "Consultation complete! Your medical report has been saved.",
        ), daemon=True).start()
        record_button.config(text="✅ FINISHED", bg="#28a745", state="disabled")
        report_button.config(state="disabled")
        return
    
    print(f"Doctor: {doctor_text}")
    response_label.config(text=doctor_text)
    messages.append({"role": "assistant", "content": doctor_text})
    threading.Thread(target=speak_text, args=(doctor_text,), daemon=True).start()
    
    record_button.config(text="🎤 Speak", bg="#0275d8", state="normal")
    report_button.config(state="normal")

# --- SUMMARY (non modifié) ---
def generate_summary():
    # Extraire les messages médicaux
    medical_msgs = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and "feeling today" in m["content"]:
            medical_msgs = messages[i+1:]
            break
    if not medical_msgs:
        medical_msgs = [m for m in messages if m["role"] != "system"]

    conversation = "\n".join([f'{m["role"]}: {m["content"]}' for m in medical_msgs])

    # Générer le résumé clinique
    try:
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "messages": [
                    {"role": "system", "content": "Generate a structured PQRST medical summary in English using exactly 3 numbered points:\n1) Chief complaint and Quality\n2) Region/Radiation and Severity\n3) Timing and Modifying factors\nBe concise. No extra text."},
                    {"role": "user", "content": conversation}
                ],
                "temperature": 0.2,
                "max_tokens": 200
            },
            timeout=20
        )
        clinical_summary = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Summary generation error: {e}")
        clinical_summary = "Error generating clinical summary."

    # ✅ ÉCRIRE LE RAPPORT COMPLET DANS LE FICHIER
    with open("patient_medical_report.txt", "w", encoding="utf-8") as f:
        f.write("📄 MEDICAL CONSULTATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write("### I. PERSONAL INFORMATION ###\n")
        for key, value in patient_data.items():
            label = key.replace('_', ' ').title()
            f.write(f"• {label}: {value if value is not None else 'Not provided'}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("### II. CLINICAL SUMMARY (PQRST) ###\n")
        f.write(clinical_summary + "\n\n")
        f.write("="*50 + "\n")
        f.write("Generated automatically. For physician review only.\n")

    print("\n📝 Medical report saved to 'patient_medical_report.txt'")

# --- AFFICHAGE DE L'ÉVALUATION AVEC LES NOUVELLES MÉTRIQUES ---
def display_evaluation():
    
    # Calcul des moyennes totales (Ancienne métrique)
    num_latencies = len(evaluation_metrics["latencies"])
    avg_latency = sum(evaluation_metrics["latencies"]) / num_latencies if num_latencies > 0 else 0
    
    # ⬅️ NOUVEAUX CALCULS DE MOYENNE
    num_asr = len(evaluation_metrics["asr_latencies"])
    avg_asr_latency = sum(evaluation_metrics["asr_latencies"]) / num_asr if num_asr > 0 else 0
    
    num_llm = len(evaluation_metrics["llm_latencies"])
    avg_llm_latency = sum(evaluation_metrics["llm_latencies"]) / num_llm if num_llm > 0 else 0
    # -------------------
    
    valid_count = evaluation_metrics["llm_questions_valid"]
    fallback_count = evaluation_metrics["llm_questions_fallback"]
    total_medical_questions = valid_count + fallback_count
    
    valid_rate = (valid_count / total_medical_questions) * 100 if total_medical_questions > 0 else 0
    
    personal_fields = ["name", "age", "marital_status", "children", "operations", "chronic_diseases"]
    filled_fields = sum(1 for k in personal_fields if patient_data.get(k) is not None and patient_data.get(k) != 'N/A')
    
    # Écriture dans un fichier d'évaluation
    evaluation_file = "system_evaluation.txt"
    with open(evaluation_file, "w", encoding="utf-8") as f:
        f.write("📈 SYSTEM EVALUATION REPORT\n" + "="*50 + "\n\n")
        
        f.write("### 1. PERFORMANCE TECHNIQUE ###\n")
        f.write(f"• Latence Moyenne (temps de réponse complet) : **{avg_latency:.2f} s**\n")
        # ⬅️ NOUVEAUX AFFICHAGES
        f.write(f"• Latence Moyenne ASR (Whisper) : **{avg_asr_latency:.2f} s**\n")
        f.write(f"• Latence Moyenne LLM (Mistral) : **{avg_llm_latency:.2f} s**\n")
        # -------------------
        f.write(f"• Nombre total d'interactions chronométrées : {num_latencies}\n")
        f.write("\n" + "-"*50 + "\n")

        f.write("### 2. EFFICACITÉ DU DIALOGUE ###\n")
        f.write(f"• Taux de Complétion du Profil Personnel : **{filled_fields}/{len(personal_fields)}**\n")
        f.write(f"• Nombre total de questions médicales posées : {total_medical_questions}\n")
        f.write(f"• Conformité LLM (Questions validées) : {valid_count} ({valid_rate:.1f}%)\n")
        f.write(f"• Fallback utilisé (Questions corrigées) : {fallback_count}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("Note: D'autres métriques (WER, Score Clinique) nécessitent une évaluation hors ligne.\n")

    print(f"\n📈 System evaluation saved to '{evaluation_file}'")
    
    return f"Evaluation metrics saved to {evaluation_file}"

def generate_report_manually():
    report_button.config(state="disabled", text="Generating...")
    record_button.config(state="disabled")
    root.update()
    generate_summary()
    final_msg = display_evaluation() 
    response_label.config(text=f"Report saved to 'patient_medical_report.txt'. {final_msg}")
    record_button.config(text="✅ FINISHED", bg="#28a745", state="disabled")
    report_button.config(text="Report Saved", bg="#28a745")

# --- UI (non modifié) ---
root = tk.Tk()
root.title("Medical Voice Assistant")
root.geometry("540x480")
root.configure(bg="#f5f7fa")

tk.Label(root, text="Intelligent Medical Assistant", bg="#f5f7fa", fg="#2a3f54",
             font=tkFont.Font(family="Arial", size=18, weight="bold")).pack(pady=(20, 5))

button_frame = tk.Frame(root, bg="#f5f7fa")
button_frame.pack(pady=10)

record_button = tk.Button(button_frame, text="🎤 Speak",
                             command=lambda: threading.Thread(target=process_audio_thread, daemon=True).start(),
                             bg="#0275d8", fg="white", font=("Arial", 14), width=15, height=1)
record_button.pack(side="left", padx=10)

report_button = tk.Button(button_frame, text="📄 Generate Report", command=generate_report_manually,
                             bg="#ffc107", fg="#333", font=("Arial", 12), width=15, height=1)
report_button.pack(side="left", padx=10)

frame = tk.Frame(root, bg="white", bd=2, relief="groove")
frame.pack(padx=20, pady=10, fill="both", expand=True)

response_label = tk.Label(frame, text="", bg="white", fg="#333", wraplength=480,
                             justify="left", font=("Arial", 12), anchor="nw")
response_label.pack(padx=10, pady=10, fill="both", expand=True)

initial_msg = "Hello! Could you please tell me your full name for my records?"
response_label.config(text=initial_msg)
threading.Thread(target=speak_text, args=(initial_msg,), daemon=True).start()

root.mainloop()