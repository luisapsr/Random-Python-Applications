import tkinter as tk
from tkinter import messagebox

# ==============================
# BLOCO 1 — INTERFACE BASE
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
# BLOCO 2 — MAPEAMENTO DE EMOÇÕES
# ==============================
humores = {
    "amor": ("❤️ Amor / Alegria", "#FFD1DC"),
    "felicidade": ("😄 Feliz", "lightgreen"),
    "neutro": ("😐 Neutro", "lightblue"),
    "raiva": ("😡 Raiva", "tomato"),
    "tristeza": ("😢 Triste", "#5DADE2"),
    "medo": ("😨 Medo", "#F4D03F"),
    "odio": ("🤬 Ódio", "#C0392B"),
    "surpresa": ("😲 Surpresa", "#D7BDE2"),
    "critica": ("🧐 Crítica", "#F8C471"),
    "julgamento": ("🤔 Julgamento", "#AAB7B8"),
}

# ==============================
# BLOCO 3 — FUNÇÕES AUXILIARES
# ==============================
def mapear_emocao(label_predito):
    label_predito = label_predito.lower()
    if "love" in label_predito or "joy" in label_predito or "5" in label_predito:
        return humores["amor"]
    elif "happy" in label_predito or "positive" in label_predito or "4" in label_predito:
        return humores["felicidade"]
    elif "anger" in label_predito or "hate" in label_predito:
        return humores["odio"]
    elif "sad" in label_predito or "negative" in label_predito or "1" in label_predito:
        return humores["tristeza"]
    elif "fear" in label_predito:
        return humores["medo"]
    elif "surprise" in label_predito:
        return humores["surpresa"]
    elif "criticism" in label_predito or "critica" in label_predito:
        return humores["critica"]
    elif "judgment" in label_predito or "julgamento" in label_predito:
        return humores["julgamento"]
    else:
        return humores["neutro"]

# ==============================
# BLOCO 4 — ANÁLISE DE EMOÇÃO
# ==============================
def analisar_frase():
    frase = entry.get().strip()
    if not frase:
        messagebox.showwarning("Aviso", "Digite algo primeiro!")
        return

    label.config(text="Analisando...", bg="white")
    root.update()

    try:
        # Importa aqui pra não travar a inicialização do Tkinter
        from transformers import pipeline

        analisador = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

        resultado = analisador(frase)[0]
        label_predito = resultado["label"]
        humor, cor = mapear_emocao(label_predito)

        label.config(text=f"{humor}\n({label_predito})", bg=cor)
        root.config(bg=cor)

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um problema na análise:\n{e}")

# ==============================
# BLOCO 5 — BOTÕES
# ==============================
botao = tk.Button(frame, text="Analisar Emoção", bg="#AED6F1",
                  font=("Arial", 12), command=analisar_frase)
botao.pack(side="left", padx=10)

botao_limpar = tk.Button(frame, text="Limpar", bg="#F5B7B1",
                         font=("Arial", 12),
                         command=lambda: entry.delete(0, tk.END))
botao_limpar.pack(side="left", padx=10)

# ==============================
# BLOCO FINAL — LOOP PRINCIPAL
# ==============================
root.mainloop()