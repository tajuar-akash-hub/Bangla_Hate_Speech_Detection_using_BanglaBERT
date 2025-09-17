from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import zipfile
import tempfile
import shutil
import os

app = FastAPI()




# Global variables
model = None
tokenizer = None

def load_model_from_zip():
    global model, tokenizer
    
    # Find zip file
    zip_file = None
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".zip"):
                zip_file = os.path.join(root, file)
                break
        if zip_file:
            break
    
    if not zip_file:
        raise Exception("No zip file found")
    
    # Extract and load
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find model folder
    model_folder = None
    for root, dirs, files in os.walk(temp_dir):
        if 'config.json' in files:
            model_folder = root
            break
    
    if not model_folder:
        raise Exception("No model found in zip")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_folder, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_folder, local_files_only=True)
    model.eval()
    
    shutil.rmtree(temp_dir)

# Load on startup
try:
    load_model_from_zip()
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bangla Hate Speech API"}

MODEL_VERSION = "1.0"
#status or api 
@app.get('/health')

def health_check():
    return{
        'status':'OK',
        'version': MODEL_VERSION,
        'Model_loaded':  model is not None
    }

@app.post("/predict")
def predict(input_data: TextInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Simple preprocessing
    text = input_data.text.strip()
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted = probs.argmax(dim=1).item()
        confidence = probs[0][predicted].item()
    
    labels = ["Non-Hate", "Hate"]
    
    return {
        "text": text,
        "prediction": labels[predicted],
        "confidence": confidence,
        "probabilities": {
            "Non-Hate": probs[0][0].item(),
            "Hate": probs[0][1].item()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)