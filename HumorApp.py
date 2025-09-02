import tkinter as tk

# Dicionário com humores e cores
humores = {
    "1": ("😄 Muito feliz", "yellow"),
    "2": ("🙂 Feliz", "lightgreen"),
    "3": ("😐 Bem", "lightblue"),
    "4": ("😕 Triste", "orange"),
    "5": ("😢 Muito triste", "red"),
}

def mudar_humor_por_tecla(event):
    tecla = event.char
    if tecla in humores:
        aplicar_humor(tecla)

def aplicar_humor(tecla):
    humor, cor = humores[tecla]
    label.config(text=humor, bg=cor)
    root.config(bg=cor)

# Janela principal
root = tk.Tk()
root.title("Alterador de Humor")
root.geometry("500x300")

label = tk.Label(root, text="Escolha um humor (1 a 5 ou clique no botão)", font=("Arial", 14), bg="white")
label.pack(expand=True, fill="both")

# Frame pros botões
frame = tk.Frame(root, bg="white")
frame.pack(pady=10)

# Criar botões a partir do dicionário
for tecla, (texto, cor) in humores.items():
    btn = tk.Button(frame, text=texto, bg=cor, width=15, height=2,
                    command=lambda t=tecla: aplicar_humor(t))
    btn.pack(side="left", padx=5)

# Bind das teclas 1 a 5
root.bind("<Key>", mudar_humor_por_tecla)

root.mainloop()