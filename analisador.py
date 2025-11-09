import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os 

MODEL_PATH = './bert_model_final'
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnalisadorSentimentos:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.id_to_label = None
        self.num_labels = 0

        self.emojis = {
            'feliz': '😊',
            'triste': '😢',
            'raiva': '😠',
            'medo': '😰',
            'neutro': '😐'
        }

        self._carregar_modelo()

    def _carregar_modelo(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Pasta do modelo '{MODEL_PATH}' não encontrada.")

        try:
            with open(f'{MODEL_PATH}/metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
                self.id_to_label = metadata['id_to_label']
                self.num_labels = len(self.id_to_label)

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH, num_labels=self.num_labels
            ).to(DEVICE)
            self.model.eval()
            print(f"✅ Modelo carregado com sucesso no dispositivo: {DEVICE}")
        except Exception as e:
            raise Exception(f"Erro ao carregar modelo: {e}")

    def analisar(self, texto: str) -> dict:
        if not self.model or not self.tokenizer:
            return {'sentimento': 'indefinido', 'confianca': 0.0, 'probabilidades': {}, 'emoji': '❓'}

        encoding = self.tokenizer.encode_plus(
            texto,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        predicted_id = np.argmax(probabilities)

        sentimento = self.id_to_label[predicted_id]
        confianca = probabilities[predicted_id]
        prob_dict = {self.id_to_label[i]: float(prob) for i, prob in enumerate(probabilities)}

        return {
            'sentimento': sentimento,
            'confianca': confianca,
            'probabilidades': prob_dict,
            'emoji': self.emojis.get(sentimento, '❓')
        }

if __name__ == "__main__":
    analisador = AnalisadorSentimentos()
    frases = [
        "Estou muito feliz com o resultado!",
        "Sinto uma raiva enorme.",
        "Estou com medo de falhar.",
        "Nada demais, estou neutro.",
        "Hoje me sinto muito triste."
    ]

    for frase in frases:
        r = analisador.analisar(frase)
        print(f"→ '{frase}' → {r['emoji']} {r['sentimento'].upper()} ({r['confianca']*100:.2f}%)")