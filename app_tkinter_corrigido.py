# ==========================================================
# BLOCO 7: INTERFACE FRONTEND COM TKINTER
# ==========================================================
import tkinter as tk
from tkinter import messagebox
import requests
import threading

# ==============================
# BLOCO 7.1 — INTERFACE BASE
# ==============================
root = tk.Tk()
root.title("IA de Emoções 💡")
root.geometry("600x400")
root.config(bg="white")

label = tk.Label(root, text="Digite uma frase e clique em 'Analisar'",
                 font=("Arial", 16), bg="white")
label.pack(pady=30)

entry = tk.Entry(root, width=60, font=("Arial", 12))
entry.pack(pady=10)

frame = tk.Frame(root, bg="white")
frame.pack(pady=10)

# ==============================
# BLOCO 7.2 — MAPEAMENTO DE EMOÇÕES
# ==============================
REACTION_MAP = {
    "love": ("❤️ Amor / Admiração", "#FFD1DC"),
    "admiration": ("❤️ Amor / Admiração", "#FFD1DC"),
    "caring": ("❤️ Amor / Admiração", "#FFD1DC"),
    "joy": ("😄 Alegria / Otimismo", "lightgreen"),
    "amusement": ("😄 Alegria / Otimismo", "lightgreen"),
    "excitement": ("😄 Alegria / Otimismo", "lightgreen"),
    "gratitude": ("😄 Alegria / Otimismo", "lightgreen"),
    "optimism": ("😄 Alegria / Otimismo", "lightgreen"),
    "anger": ("😡 Raiva / Ódio", "tomato"),
    "disgust": ("😡 Raiva / Ódio", "tomato"),
    "annoyance": ("😡 Raiva / Ódio", "tomato"),
    "approval": ("👍 Julgamento (Positivo)", "#AAB7B8"),
    "disapproval": ("🧐 Crítica / Julgamento", "#F8C471"),
    "sadness": ("😢 Tristeza / Medo", "#5DADE2"),
    "grief": ("😢 Tristeza / Medo", "#5DADE2"),
    "fear": ("😨 Tristeza / Medo", "#5DADE2"),
    "nervousness": ("😨 Tristeza / Medo", "#5DADE2"),
    "remorse": ("😢 Tristeza / Medo", "#5DADE2"),
    "disappointment": ("😢 Tristeza / Medo", "#5DADE2"),
    "surprise": ("😲 Surpresa / Confusão", "#D7BDE2"),
    "realization": ("😲 Surpresa / Confusão", "#D7BDE2"),
    "confusion": ("😲 Surpresa / Confusão", "#D7BDE2"),
    "curiosity": ("😲 Surpresa / Confusão", "#D7BDE2"),
    "neutral": ("😐 Neutro", "lightblue"),
}

API_URL = "http://127.0.0.1:8000/analyze"

# ==============================
# BLOCO 7.3 — LÓGICA DE ANÁLISE
# ==============================
def analisar_frase():
    frase = entry.get().strip()
    if not frase:
        messagebox.showwarning("Aviso", "Digite algo primeiro!")
        return

    label.config(text="Analisando...", bg="white")
    root.config(bg="white")
    botao.config(state="disabled")
    root.update()

    thread = threading.Thread(target=perform_api_request, args=(frase,), daemon=True)
    thread.start()

def perform_api_request(texto_para_analisar: str):
    try:
        response = requests.post(API_URL, json={"text": texto_para_analisar}, timeout=10)
        response.raise_for_status()
        data = response.json()
        api_labels = data.get("labels", ["neutral"])

        primeira_emocao = api_labels[0] if isinstance(api_labels, list) and api_labels else "neutral"
        humor_texto, cor = REACTION_MAP.get(primeira_emocao, ("😐 Neutro", "lightblue"))
        display_text = f"{humor_texto}\n(IA detectou: {primeira_emocao})"

        root.after(0, update_reaction, display_text, cor)

    except requests.exceptions.ConnectionError:
        root.after(0, update_reaction, "Erro: API de IA está offline.", "tomato")
    except requests.exceptions.Timeout:
        root.after(0, update_reaction, "Erro: A análise demorou muito.", "tomato")
    except Exception as e:
        root.after(0, update_reaction, f"Erro: {e}", "tomato")
    finally:
        root.after(0, lambda: botao.config(state="normal"))

def update_reaction(texto: str, cor: str):
    label.config(text=texto, bg=cor)
    root.config(bg=cor)

def limpar_campos():
    entry.delete(0, tk.END)
    label.config(text="Digite uma frase e clique em 'Analisar'", bg="white")
    root.config(bg="white")

# ==============================
# BLOCO 7.4 — BOTÕES
# ==============================
botao = tk.Button(frame, text="Analisar Emoção", bg="#AED6F1",
                  font=("Arial", 12), command=analisar_frase)
botao.pack(side="left", padx=10)

botao_limpar = tk.Button(frame, text="Limpar", bg="#F5B7B1",
                         font=("Arial", 12), command=limpar_campos)
botao_limpar.pack(side="left", padx=10)

# ==============================
# BLOCO FINAL — LOOP PRINCIPAL
# ==============================
root.mainloop()