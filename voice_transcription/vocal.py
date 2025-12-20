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

pygame.mixer.init()
model = whisper.load_model("medium")

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a doctor using PQRST method (Provocation, Quality, Region, Severity, Timing).

RULES:
1. Ask ONE complete question per message (8-15 words)
2. Questions must have subject + verb + complement
3. Never repeat questions
4. After gathering PQRST info, respond with [SUMMARY]

GOOD: 'Can you describe how the pain feels?'
BAD: 'Caused by?' (incomplete)

Be professional and systematic."""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
# --- LIGNE CORRIGÉE : Indentation standard ---
patient_data = {k: None for k in ["name", "age", "marital_status", "children", "children_ages",
                                 "operations", "operation_details", "chronic_diseases", "chronic_disease_details"]}
conversation_phase = "personal_info"
question_counter = 0
medical_questions_asked = set()
is_speaking = False

# --- NOUVELLE SECTION: METRIQUES D'ÉVALUATION ---
evaluation_metrics = {
    "latencies": [],
    "llm_questions_valid": 0,
    "llm_questions_fallback": 0
}
# -------------------------------------------------


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

# --- EXTRACTION ---
def extract_numbers(text):
    return re.findall(r'\d+', text)

def extract_age(text):
    nums = extract_numbers(text)
    return nums[0] if nums else text.strip(".").strip()

# --- MODIFICATION POUR INCLURE 'SIX' ET PLUS DANS LES MOTS ---
def extract_children(text):
    nums = extract_numbers(text)
    if nums:
        return nums[0]
    
    # Dictionnaire étendu
    words = {"one":1,"two":2,"three":3,"four":4,"five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10}
    for word, num in words.items():
        if word in text.lower().strip('.'): 
            return str(num)
    return None
# -------------------------------------------------------------

# --- PHASE 1: PERSONAL INFO ---
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
        # --- MODIFICATION DE LA LOGIQUE DES ENFANTS ---
        # Vérifie si un nombre a été extrait (children n'est pas None) OU si 'yes' est présent
        if children is not None or "yes" in patient_text.lower():
            patient_data["children"] = children or "Yes" # Si children est None mais 'yes' est là, enregistre 'Yes'
            return PERSONAL_QUESTIONS["children_yes"]
        
        # Si aucun nombre n'est extrait ET 'yes' n'est pas là, on suppose 'No'
        patient_data["children"] = "No"
        return PERSONAL_QUESTIONS["children_no"]
        # -----------------------------------------------
    
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
        if "yes" in patient_text.lower() or "have" in patient_text.lower():
            patient_data["chronic_diseases"] = "yes"
            return PERSONAL_QUESTIONS["chronic_yes"]
        patient_data["chronic_diseases"] = "no"
        conversation_phase = "medical_consultation"
        question_counter = 0
        return PERSONAL_QUESTIONS["chronic_no"]
    
    if patient_data["chronic_diseases"] == "yes" and not patient_data["chronic_disease_details"]:
        patient_data["chronic_disease_details"] = re.sub(r'i (have|live with)', '', patient_text, flags=re.I).strip(".")
        conversation_phase = "medical_consultation"
        question_counter = 0
        return PERSONAL_QUESTIONS["chronic_no"]
    
    conversation_phase = "medical_consultation"
    return "How are you feeling today?"

# --- PHASE 2: MEDICAL CONSULTATION ---
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
        medical_questions_asked.add(messages[-1]["content"].strip())
    
    filtered = [m for m in messages if m["role"] == "system" or m["role"] == "user" or 
                (m["role"] == "assistant" and ('?' in m["content"] or '[SUMMARY]' in m["content"]))]
    
    temp_messages = filtered + [{
        "role": "system",
        "content": f"Questions asked: {list(medical_questions_asked)}\n\n"
                     "Ask a NEW complete question about PQRST. "
                     "If you have enough info, respond with [SUMMARY]."
    }]
    
    try:
        r = requests.post("http://localhost:11434/api/chat", 
                          json={"model": "mistral", "messages": temp_messages, "stream": False})
        response = r.json()["message"]["content"].strip().replace('-', '')
        
        if response.startswith("[SUMMARY]"):
            return "[SUMMARY_REQUESTED]"
        
        # Extraire première phrase
        match = re.search(r'[.?\n]', response)
        response = response[:match.end()].strip() if match else response.strip()
        response = response.rstrip('.') + '?' if not response.endswith('?') else response
        
        # Valider et utiliser fallback si nécessaire
        if validate_question(response):
            medical_questions_asked.add(response)
            evaluation_metrics["llm_questions_valid"] += 1 # ⬅️ Métrique : Bonne question
            return response
        else:
            evaluation_metrics["llm_questions_fallback"] += 1 # ⬅️ Métrique : Question corrigée
            return PQRST_FALLBACK[question_counter % 6]
            
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN PROCESS ---
def process_audio_thread():
    global conversation_phase, question_counter
    
    while is_speaking:
        root.after(100)
    
    record_button.config(text="● Recording...", bg="#d9534f", state="disabled")
    report_button.config(state="disabled")
    root.update()
    
    start_time = time.time() # ⬅️ Début du chronométrage pour la latence
    filename = record_audio()
    patient_text = model.transcribe(filename, language="en")["text"].strip()
    print(f"\nPatient: {patient_text}")
    
    messages.append({"role": "user", "content": patient_text})
    
    if conversation_phase == "personal_info":
        doctor_text = get_personal_info_question(patient_text)
    else:
        if question_counter >= MAX_QUESTIONS:
            doctor_text = "[SUMMARY_REQUESTED]"
        else:
            doctor_text = generate_medical_question(messages)
            if doctor_text != "[SUMMARY_REQUESTED]":
                question_counter += 1
    
    end_time = time.time()
    evaluation_metrics["latencies"].append(end_time - start_time) # ⬅️ Enregistrement de la latence

    if doctor_text == "[SUMMARY_REQUESTED]":
        print("\n🛑 Consultation concluded")
        generate_summary()
        final_msg = display_evaluation() # ⬅️ Génère et affiche les métriques d'évaluation
        response_label.config(text=f"Consultation complète! {final_msg}")
        threading.Thread(target=speak_text, args=("Consultation complete! Your report is being sent to your doctor. Wishing you a speedy recovery!",), daemon=True).start()
        record_button.config(text="✅ FINISHED", bg="#28a745", state="disabled")
        report_button.config(state="disabled")
        return
    
    print(f"Doctor: {doctor_text}")
    response_label.config(text=doctor_text)
    messages.append({"role": "assistant", "content": doctor_text})
    threading.Thread(target=speak_text, args=(doctor_text,), daemon=True).start()
    
    record_button.config(text="🎤 Speak", bg="#0275d8", state="normal")
    report_button.config(state="normal")

# --- SUMMARY ---
def generate_summary():
    medical_msgs = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and "feeling today" in m["content"]:
            medical_msgs = messages[i+1:]
            break
    
    if not medical_msgs:
        medical_msgs = [m for m in messages if m["role"] != "system"]
    
    conversation = "\n".join([f'{m["role"]}: {m["content"]}' for m in medical_msgs])
    
    summary_prompt = {
        "role": "system",
        "content": "Generate a structured PQRST medical summary in English using numbered points:\n"
                     "1) Chief complaint and Quality\n2) Region/Radiation and Severity\n3) Timing and Modifying factors"
    }
    
    try:
        r = requests.post("http://localhost:11434/api/chat",
                          json={"model": "mistral", "messages": [summary_prompt, {"role": "user", "content": conversation}], "stream": False})
        summary = r.json()["message"]["content"].strip()
    except:
        summary = "Error generating summary"
    
    with open("patient_medical_report.txt", "w", encoding="utf-8") as f:
        f.write("📄 MEDICAL REPORT\n" + "="*50 + "\n\n")
        f.write("### I. PERSONAL INFORMATION ###\n")
        for k, v in patient_data.items():
            f.write(f"• {k.replace('_', ' ').title()}: {v or 'N/A'}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("### II. CLINICAL SUMMARY ###\n" + summary + "\n")
    
    print("\n📝 Report saved to 'patient_medical_report.txt'")

# --- NOUVELLE FONCTION: AFFICHAGE DE L'ÉVALUATION ---
def display_evaluation():
    
    # Calcul des moyennes
    num_latencies = len(evaluation_metrics["latencies"])
    avg_latency = sum(evaluation_metrics["latencies"]) / num_latencies if num_latencies > 0 else 0
    
    valid_count = evaluation_metrics["llm_questions_valid"]
    fallback_count = evaluation_metrics["llm_questions_fallback"]
    total_medical_questions = valid_count + fallback_count
    
    valid_rate = (valid_count / total_medical_questions) * 100 if total_medical_questions > 0 else 0
    
    # Déterminer la complétion des données personnelles
    personal_fields = ["name", "age", "marital_status", "children", "operations", "chronic_diseases"]
    filled_fields = sum(1 for k in personal_fields if patient_data.get(k) is not None and patient_data.get(k) != 'N/A')
    
    # Écriture dans un fichier d'évaluation
    evaluation_file = "system_evaluation.txt"
    with open(evaluation_file, "w", encoding="utf-8") as f:
        f.write("📈 SYSTEM EVALUATION REPORT\n" + "="*50 + "\n\n")
        
        f.write("### 1. PERFORMANCE TECHNIQUE ###\n")
        f.write(f"• Latence Moyenne (temps de réponse complet) : **{avg_latency:.2f} s**\n")
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
# -------------------------------------------------------------

def generate_report_manually():
    report_button.config(state="disabled", text="Generating...")
    record_button.config(state="disabled")
    root.update()
    generate_summary()
    final_msg = display_evaluation() # ⬅️ Appel pour l'évaluation
    response_label.config(text=f"Report saved to 'patient_medical_report.txt'. {final_msg}")
    record_button.config(text="✅ FINISHED", bg="#28a745", state="disabled")
    report_button.config(text="Report Saved", bg="#28a745")

# --- UI ---
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