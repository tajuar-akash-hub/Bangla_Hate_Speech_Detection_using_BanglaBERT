from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import zipfile
import tempfile
import shutil
import os

# Find the zip file in current directory
zip_file = None
for file in os.listdir("."):
    if file.endswith(".zip"):
        zip_file = file
        break

# Create temporary directory and extract the zip
temp_dir = tempfile.mkdtemp()
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Find model folder (contains config.json)
model_folder = None
for root, dirs, files in os.walk(temp_dir):
    if 'config.json' in files:
        model_folder = root
        break

# Load model and tokenizer from extracted folder
model = AutoModelForSequenceClassification.from_pretrained(model_folder, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_folder, local_files_only=True)

# Set device to CPU and put model in evaluation mode
device = torch.device("cpu")
model.to(device)
model.eval()

# Test text
text = "আমি তোমাকে ভালোবাসি।"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

# Run inference (no gradient calculation needed)
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)

# Get prediction results
predicted_class = probabilities.argmax(dim=1).item()
confidence = probabilities[0][predicted_class].item()

# Define labels (0=Non-Hate, 1=Hate)
labels = ["Non-Hate", "Hate"]

# Print results
print(f"Text: {text}")
print(f"Prediction: {labels[predicted_class]}")
print(f"Confidence: {confidence:.4f}")
print(f"Non-Hate: {probabilities[0][0]:.4f}, Hate: {probabilities[0][1]:.4f}")

# Clean up temporary files
shutil.rmtree(temp_dir)