from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os

# Ruta relativa a la carpeta del modelo
MODEL_PATH = "./modelo_bert_final_92"

# Mapeo de etiquetas 
LABEL_MAP = {
    0: 'SADNESS', 
    1: 'JOY', 
    2: 'LOVE', 
    3: 'ANGER', 
    4: 'FEAR', 
    5: 'SURPRISE'
}

class ClasificadorEmociones:
    def __init__(self):
        print("Cargando modelo BERT local...")
        
        # Verificamos que la carpeta exista
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No encuentro la carpeta {MODEL_PATH}. ¿La descomprimiste bien?")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            
            # Detectar si hay GPU o usar CPU (para que no falle en tu PC)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"Modelo cargado exitosamente en: {self.device}")
        except Exception as e:
            print(f"Error fatal cargando el modelo: {e}")

    def predecir(self, texto):
        # 1. Tokenizar
        inputs = self.tokenizer(texto, return_tensors="pt", truncation=True, max_length=64, padding=True)
        # Mover datos al mismo dispositivo que el modelo
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Predicción (sin guardar gradientes para ahorrar RAM)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 3. Procesar resultados
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        pred_idx = torch.argmax(probs).item()
        confianza = probs[0][pred_idx].item()
        emocion = LABEL_MAP[pred_idx]
        
        return emocion, confianza