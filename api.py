import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from langdetect import detect, LangDetectException
import logging

# --- CONFIGURAÇÃO DE LOG ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INICIALIZAÇÃO DO APP ---
app = FastAPI(
    title="API de Análise de Emoções Multilíngue",
    description="API que detecta emoções em textos PT/EN usando Transformers."
)

# --- MODELOS DE ENTRADA/SAÍDA ---
class TextIn(BaseModel):
    text: str

class AnalysisOut(BaseModel):
    language: str
    labels: list[str]
    raw_text: str

# --- CARREGAMENTO DOS MODELOS ---
logger.info("Carregando modelos... isso pode levar alguns segundos.")

try:
    # Modelo PT treinado localmente
    MODEL_PT_PATH = "./bertimbau-emocoes-multi-label"
    tokenizer_pt = BertTokenizer.from_pretrained(MODEL_PT_PATH)
    model_pt = BertForSequenceClassification.from_pretrained(MODEL_PT_PATH)
    logger.info("Modelo BERTimbau (PT) carregado com sucesso.")

    # Modelo EN do Hugging Face
    MODEL_EN_NAME = "SamLowe/roberta-base-go_emotions"
    tokenizer_en = AutoTokenizer.from_pretrained(MODEL_EN_NAME)
    model_en = AutoModelForSequenceClassification.from_pretrained(MODEL_EN_NAME)
    logger.info("Modelo RoBERTa (EN) carregado com sucesso.")

except Exception as e:
    logger.error(f"Erro ao carregar modelos: {e}")
    model_pt = model_en
    tokenizer_pt = tokenizer_en

# --- FUNÇÃO DE ANÁLISE ---
def analyze_emotion(text: str) -> dict:
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "en"

    if lang == "pt":
        model = model_pt
        tokenizer = tokenizer_pt
        id2label = model_pt.config.id2label
    else:
        lang = "en"
        model = model_en
        tokenizer = tokenizer_en
        id2label = model_en.config.id2label

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.sigmoid(logits).squeeze()
    active_indices = (probs > 0.5).nonzero(as_tuple=False)
    active_labels = [id2label[i.item()] for i in active_indices]

    if not active_labels:
        top_label_index = torch.argmax(probs)
        active_labels = [id2label[top_label_index.item()]]

    if "neutral" in active_labels and len(active_labels) > 1:
        active_labels.remove("neutral")

    return {"language": lang, "labels": active_labels}

# --- ENDPOINT ---
@app.post("/analyze", response_model=AnalysisOut)
async def analyze_text_endpoint(request: TextIn):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode estar vazio.")
    try:
        result = analyze_emotion(request.text)
        return AnalysisOut(language=result["language"], labels=result["labels"], raw_text=request.text)
    except Exception as e:
        logger.error(f"Erro durante análise: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
