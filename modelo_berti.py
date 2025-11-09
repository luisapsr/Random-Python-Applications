import os
import numpy as np
import pandas as pd
import torch
import pickle
import kagglehub

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# ----------------------------------------------------
# 1️⃣ Baixa e prepara o dataset do Kaggle
# ----------------------------------------------------
print("📦 Baixando dataset do Kaggle (GoEmotions PT-BR)...")
path = kagglehub.dataset_download("antoniomenezes/go-emotions-ptbr")

from pathlib import Path

dataset_dir = Path(path)
print("\n📂 Arquivos encontrados no dataset:")
for file in dataset_dir.glob("*"):
    print("  -", file.name)

# Força o script a parar aqui só pra inspeção
import sys

# ✅ Adicione daqui
from pathlib import Path
dataset_dir = Path(path)

print("📂 Arquivos encontrados no dataset:")
for file in dataset_dir.glob("*"):
    print("  -", file.name)

# Junta todos os CSVs do dataset em um só DataFrame
csv_files = list(dataset_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("❌ Nenhum CSV encontrado no dataset baixado!")

print(f"📊 Encontrados {len(csv_files)} arquivos CSV. Carregando todos...")

dataframes = [pd.read_csv(f) for f in csv_files]
data = pd.concat(dataframes, ignore_index=True)

print(f"✅ Dataset combinado com {len(data)} linhas no total.")
print(f"Colunas: {list(data.columns)}")

print(f"✅ Dataset carregado com {len(data)} linhas.")
print(f"Colunas: {list(data.columns)}")

# ----------------------------------------------------
# Ajusta colunas do dataset e prepara emoções
# ----------------------------------------------------
# Detecta automaticamente as colunas de emoção (são binárias)
emotion_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Garante que só usa as que realmente existem no CSV
emotion_cols = [c for c in emotion_cols if c in data.columns]

# Cria a coluna "emotion" pegando a emoção dominante por linha
data["emotion"] = data[emotion_cols].idxmax(axis=1)

# Mantém apenas as colunas necessárias
data = data[["text", "emotion"]].dropna()

print(f"✅ Dataset preparado: {len(data)} linhas e colunas {list(data.columns)}")

# Mantém apenas as 5 emoções principais
emoes = ["joy", "sadness", "anger", "fear", "neutral"]
data = data[data["emotion"].isin(emoes)]

# Tradução dos rótulos para português
traduz = {
    "joy": "feliz",
    "sadness": "triste",
    "anger": "raiva",
    "fear": "medo",
    "neutral": "neutro"
}
data["emotion"] = data["emotion"].map(traduz)
data = data.rename(columns={"text": "frase", "emotion": "sentimento"})

dataset_global = data.reset_index(drop=True)
print("🎯 Emoções selecionadas:", dataset_global["sentimento"].unique())

# ----------------------------------------------------
# 2️⃣ Configurações do modelo
# ----------------------------------------------------
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
BATCH_SIZE = 8
EPOCHS = 4
MAX_LEN = 128


class ModeloSentimentosBERT:
    def __init__(self, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.label_map = None
        self.treinado = False
        print(f"⚙️  Treinando em: {self.device}")

    def carregar_dados(self, df):
        dados = df.copy().dropna(subset=['frase', 'sentimento'])
        unique_labels = dados['sentimento'].unique()
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        dados['label'] = dados['sentimento'].map(self.label_map)
        print(f"✓ Base carregada: {len(dados)} frases | Classes: {self.label_map}")
        return dados

    def treinar(self, df):
        print("\n🤖 Iniciando Treinamento BERTimbau...")
        print("=" * 60)

        dados = self.carregar_dados(df)
        train_df, val_df = train_test_split(dados, test_size=0.2, random_state=42, stratify=dados['label'])

        def encode_data(tokenizer, texts, max_len):
            encoded = tokenizer.batch_encode_plus(
                texts.tolist(),
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return encoded['input_ids'], encoded['attention_mask']

        train_input_ids, train_attention_masks = encode_data(self.tokenizer, train_df['frase'], MAX_LEN)
        val_input_ids, val_attention_masks = encode_data(self.tokenizer, val_df['frase'], MAX_LEN)

        train_labels = torch.tensor(train_df['label'].tolist())
        val_labels = torch.tensor(val_df['label'].tolist())

        train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)

        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)

        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_dataloader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        best_val_acc = 0
        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0

            for batch in train_dataloader:
                b_input_ids, b_input_mask, b_labels = [b.to(self.device) for b in batch]

                self.model.zero_grad()
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            val_loss, val_acc = self._avaliar(val_dataloader)

            print(f"📊 Época {epoch+1} | Loss treino: {avg_train_loss:.4f} | Acc validação: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.salvar_modelo('./bert_model_final')
                print("💾 Novo melhor modelo salvo!")

        print("=" * 60)
        print(f"🏁 Treinamento concluído — Melhor Acurácia: {best_val_acc*100:.2f}%")
        self.treinado = True

    def _avaliar(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_correct = 0

        for batch in dataloader:
            b_input_ids, b_input_mask, b_labels = [b.to(self.device) for b in batch]
            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == b_labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / len(dataloader.dataset)
        return avg_loss, avg_acc

    def salvar_modelo(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(f'{path}/metadata.pkl', 'wb') as f:
            pickle.dump({'label_map': self.label_map, 'id_to_label': self.id_to_label}, f)
        print(f"💾 Modelo salvo com sucesso em: {path}")

    def carregar_modelo(self, path='./bert_model_final'):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        with open(f'{path}/metadata.pkl', 'rb') as f:
            meta = pickle.load(f)
            self.label_map = meta['label_map']
            self.id_to_label = meta['id_to_label']
        num_labels = len(self.label_map)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels).to(self.device)
        self.treinado = True
        print(f"✅ Modelo carregado com {num_labels} classes.")

    def prever(self, texto):
        if not self.treinado:
            self.carregar_modelo()
        self.model.eval()

        tokens = self.tokenizer.encode_plus(
            texto,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokens['input_ids'].to(self.device)
        att_mask = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=att_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred_idx = np.argmax(probs)
        sentimento = self.id_to_label[int(np.argmax(logits))]
        confianca = probs[pred_idx]

        return sentimento, confianca, {self.id_to_label[i]: p for i, p in enumerate(probs)}


# ----------------------------------------------------
# 3️⃣ Execução
# ----------------------------------------------------
# ----------------------------------------------------
# 3️⃣ Execução
# ----------------------------------------------------
def main():
    print("🚀 Script de treino iniciado!")
    modelo = ModeloSentimentosBERT(num_labels=4)
    modelo.treinar(dataset_global)

    testes = [
        'Estou muito feliz hoje!',
        'Sinto uma raiva absurda!',
        'Tenho medo de perder tudo.',
        'Estou triste e sem vontade de sair.'
    ]

    print("\n🧪 Testes de previsão:")
    for frase in testes:
        sent, conf, probs = modelo.prever(frase)
        print(f"'{frase}' → {sent.upper()} ({conf*100:.1f}%)")


if __name__ == "__main__":
    main()