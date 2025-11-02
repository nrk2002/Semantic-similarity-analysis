import os
import re
import pickle
import numpy as np
import fitz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename

# Download necessary NLP resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer and stopwords set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Advanced preprocessing:
      - Lowercase
      - Remove non-alphabetic characters
      - Tokenize using NLTK
      - Remove stopwords
      - Lemmatize tokens
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return processed_words

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)  # Open PDF
        for page in doc:
            page_text = page.get_text()  # Extract text from each page
            if page_text:
                text += page_text
        doc.close()
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
    return text

# Flask app configuration
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load saved models and data
def load_models():
    """Load all necessary models and data with error handling"""
    models = {
        'tokenizer': None,
        'word_embeddings': None,
        'train_words': None,
        'train_filenames': None,
        'skipgram_metrics': None,
        'cnn_model': None,
        'lstm_model': None
    }
    
    try:
        with open("tokenizer.pkl", "rb") as f:
            models['tokenizer'] = pickle.load(f)
        models['word_embeddings'] = np.load("word_embeddings.npy")
        with open("train_words.pkl", "rb") as f:
            models['train_words'] = pickle.load(f)
        with open("train_filenames.pkl", "rb") as f:
            models['train_filenames'] = pickle.load(f)
        with open("skipgram_metrics.pkl", "rb") as f:
            models['skipgram_metrics'] = pickle.load(f)
            
        # Initialize CNN and LSTM models
        embedding_layer = Embedding(
            input_dim=len(models['tokenizer'].word_index) + 1,
            output_dim=models['word_embeddings'].shape[1],
            weights=[models['word_embeddings']],
            trainable=False
        )
        
        # CNN Model
        inputs = Input(shape=(None,))
        x = embedding_layer(inputs)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        models['cnn_model'] = Model(inputs, x)
        
        # LSTM Model
        inputs = Input(shape=(None,))
        x = embedding_layer(inputs)
        x = LSTM(128, return_sequences=True)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        models['lstm_model'] = Model(inputs, x)
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
    
    return models

models = load_models()

# Similarity calculation functions
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

def jaccard_similarity(set1, set2):
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union if union else 0

def hybrid_similarity(train_vec, test_vec, train_tokens, test_tokens, w1=0.2, w2=0.4, w3=0.4):
    euc_sim = 1 / (1 + euclidean_distance(train_vec, test_vec))
    cos_sim = cosine_similarity(train_vec, test_vec)
    jac_sim = jaccard_similarity(train_tokens, test_tokens)
    return (w1 * euc_sim) + (w2 * cos_sim) + (w3 * jac_sim)

def get_simple_embeddings(tokens):
    embeddings = [models['word_embeddings'][models['tokenizer'].word_index[word]] 
                 for word in tokens if word in models['tokenizer'].word_index]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(models['word_embeddings'].shape[1])

def get_model_embeddings(tokens, model):
    sequence = [models['tokenizer'].word_index[word] 
               for word in tokens if word in models['tokenizer'].word_index]
    sequence = pad_sequences([sequence], padding='post', truncating='post', dtype='int32')
    return model.predict(sequence)[0]

train_vectors = [get_simple_embeddings(tokens) for tokens in models['train_words']]

def analyze_similarity(test_tokens, top_n=None):
    if not models['tokenizer'] or not models['train_words']:
        return {"error": "Models not properly loaded"}
    
    test_simple = get_simple_embeddings(test_tokens)
    test_cnn = get_model_embeddings(test_tokens, models['cnn_model'])
    test_lstm = get_model_embeddings(test_tokens, models['lstm_model'])
    
    results = {"simple": [], "cnn": [], "lstm": [], "hybrid": []}
    
    for i, (train_vec, train_tokens, train_filename) in enumerate(zip(train_vectors, models['train_words'], models['train_filenames'])):
        simple_sim = 1 / (1 + euclidean_distance(test_simple, train_vec))
        train_cnn = get_model_embeddings(train_tokens, models['cnn_model'])
        cnn_sim = 1 / (1 + euclidean_distance(test_cnn, train_cnn))
        train_lstm = get_model_embeddings(train_tokens, models['lstm_model'])
        lstm_sim = 1 / (1 + euclidean_distance(test_lstm, train_lstm))
        hybrid_score = hybrid_similarity(train_vec, test_simple, train_tokens, test_tokens)
        
        results["simple"].append({"train_doc": train_filename, "score": round(simple_sim, 4)})
        results["cnn"].append({"train_doc": train_filename, "score": round(cnn_sim, 4)})
        results["lstm"].append({"train_doc": train_filename, "score": round(lstm_sim, 4)})
        results["hybrid"].append({"train_doc": train_filename, "score": round(hybrid_score, 4)})
    
    for key in results:
        results[key] = sorted(results[key], key=lambda x: x["score"], reverse=True)
        if top_n is not None:
            results[key] = results[key][:top_n]
    
    if models['skipgram_metrics']:
        results['model_info'] = {
            'embedding_dim': models['word_embeddings'].shape[1],
            'vocab_size': len(models['tokenizer'].word_index) + 1,
            'training_date': models['skipgram_metrics'].get('training_date', 'unknown'),
            'roc_auc': models['skipgram_metrics'].get('roc_auc', 'unknown'),
            'roc_curve_image': url_for('static', filename='skipgram_roc_curve.png')  # Add this line
        }
    
    return results

@app.route("/")
def index():
    return render_template("index.html")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        try:
            text = extract_text_from_pdf(filepath)
            tokens = preprocess_text(text)
            results = analyze_similarity(tokens, top_n=10)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"})
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    return jsonify({"error": "Invalid file type"})

@app.route("/get_model_metrics")
def get_model_metrics():
    if not models['skipgram_metrics']:
        return jsonify({"error": "Metrics not available"})
    
    metrics = models['skipgram_metrics']
    
    response = {
        'model_info': {
            'embedding_dim': models['word_embeddings'].shape[1],
            'vocab_size': len(models['tokenizer'].word_index) + 1,
            'training_date': metrics.get('training_date', 'unknown')
        },
        'performance': {
            'final_train_accuracy': metrics.get('final_train_accuracy', 0),
            'final_val_accuracy': metrics.get('final_val_accuracy', 0),
            'final_train_loss': metrics.get('final_train_loss', 0),
            'final_val_loss': metrics.get('final_val_loss', 0),
            'roc_auc': metrics.get('roc_auc', 0),
            'training_history': metrics.get('training_history', {}),
            'roc_curve_image': url_for('static', filename='skipgram_roc_curve.png')  # Include ROC curve image URL
        }
    }
    
    return jsonify(response)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=5000, debug=True, use_reloader=False)