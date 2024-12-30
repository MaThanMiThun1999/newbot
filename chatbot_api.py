from flask import Flask, request, jsonify
import re
import random
import torch
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
from googletrans import Translator
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import os
import logging
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import torch.nn as nn
import psutil

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_memory_usage():
    # Get the current process memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"Memory used: {memory_info.rss / 1024 / 1024} MB")  # in MB

get_memory_usage()
# --- Load necessary data and models ---
# Load intents data from JSON
def load_data():
    with open('intents.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['intents'])
    dic = {"tag": [], "patterns": [], "responses": []}
    for i in range(len(df)):
        ptrns = df[df.index == i]['patterns'].values[0]
        rspns = df[df.index == i]['responses'].values[0]
        tag = df[df.index == i]['tag'].values[0]
        for j in range(len(ptrns)):
            dic['tag'].append(tag)
            dic['patterns'].append(ptrns[j])
            dic['responses'].append(rspns)
    df = pd.DataFrame.from_dict(dic)
    return df

df = load_data()

# Preprocessing function
def preprocess_text(s):
    s = re.sub("[^a-zA-Z']", ' ', s)
    s = s.lower()
    s = s.split()
    s = " ".join(s)
    return s

df['patterns'] = df['patterns'].apply(preprocess_text)
df['tag'] = df['tag'].apply(preprocess_text)

# Initialize DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
max_len = 128

# Encoding labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['tag'])
num_labels = len(np.unique(y_encoded))

def encode_texts(texts, max_len):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

# Encode the patterns
X = df['patterns']
input_ids, attention_masks = encode_texts(X, max_len)
labels = torch.tensor(y_encoded)

# Define a classification model using DistilBert Model
class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(DistilBertClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert/distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels) # 768 is the output size of base distilbert

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] # Use the [CLS] token's output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Load the Model (Load pre-trained weights):
model = DistilBertClassifier(num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model Loading Logic
MODEL_PATH = 'trained_model.pth'
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only = True))
    logging.info("Trained model loaded successfully.")
else:
    # Splitting the dataset into training and validation
    dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    validation_dataloader = DataLoader(val_dataset, batch_size=16)

    # Initialize DistilBERT model
    
    optimizer = AdamW(model.parameters(), lr=2e-5)

    #Training the Model
    epochs = 10 # Increased Epochs
    for epoch in range(epochs):
      model.train()
      total_train_loss = 0
      for batch in train_dataloader:
          b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
          model.zero_grad()        
          outputs = model(b_input_ids, attention_mask=b_input_mask)
          loss = nn.CrossEntropyLoss()(outputs, b_labels)
          total_train_loss += loss.item()
          loss.backward()
          optimizer.step()
      avg_train_loss = total_train_loss / len(train_dataloader)            
      logging.info(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Model trained and saved to {MODEL_PATH}")

model.eval()  # Set model to evaluation mode

translator = Translator()

# --- Chatbot Functions ---
def get_response(user_input, language_code):
    translated_input = translator.translate(user_input, src=language_code, dest='en').text
    txt = re.sub("[^a-zA-Z']", ' ', translated_input)
    txt = txt.lower().strip()

    encoded_dict = tokenizer.encode_plus(
        txt,
        txt,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation = True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    with torch.no_grad():
      outputs = model(input_ids, attention_mask=attention_mask)


    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    predicted_label_idx = np.argmax(probabilities, axis=1)[0]
    tag = label_encoder.inverse_transform([predicted_label_idx])[0]
    logging.info(f"Predicted Tag: {tag}")

    if tag in df['tag'].values:
        responses = df[df['tag'] == tag]['responses'].values[0]
        response = random.choice(responses)
    else:
        response = "I'm not sure how to respond to that. Can you rephrase?"

    translated_response = translator.translate(response, src='en', dest=language_code).text
    return translated_response


# --- Flask Routes ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'user_input' not in data or 'language_code' not in data:
          return jsonify({"error": "Invalid request format. Please provide 'user_input' and 'language_code' in the request body."}), 400
    user_input = data['user_input']
    print("user_input :", user_input)
    language_code = data['language_code']
    print("language_code :", language_code)
    response = get_response(user_input, language_code)
    print("response :" ,response)
    return jsonify({'response': response})

@app.route('/supported_languages', methods=['GET'])
def supported_languages():
    languages = {
      'languages': [
          {'code': 'en', 'name': 'English'},
            {'code': 'hi', 'name': 'Hindi'},
            {'code': 'es', 'name': 'Spanish'},
            {'code': 'fr', 'name': 'French'},
            {'code': 'de', 'name': 'German'},
            {'code': 'zh-cn', 'name': 'Chinese Simplified'},
            {'code': 'ar', 'name': 'Arabic'},
            {'code': 'ta', 'name': 'Tamil'},
            {'code': 'te', 'name': 'Telugu'},
            {'code': 'ru', 'name': 'Russian'},
            {'code': 'ja', 'name': 'Japanese'},
            {'code': 'ko', 'name': 'Korean'}
      ],
        'message': "For a complete list, visit: https://cloud.google.com/translate/docs/languages"
    }
    return jsonify(languages)


@app.route('/')
def index():
    return "ElevateMind's Mental Health Chatbot API"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)