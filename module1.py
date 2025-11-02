import os
import re
import fitz
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLP resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer and stopwords set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def extract_text_from_folder(folder_path):
    """Extract and preprocess text from all PDFs in a folder and store filenames."""
    all_texts = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            words = preprocess_text(text)
            all_texts.append(words)
            filenames.append(filename)  # Store the filename
    
    return all_texts, filenames

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return text.strip()

def preprocess_text(text):
    """
    Advanced preprocessing:
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

# Define paths
train_folder = "train_pdfs"

# Extract text from training PDFs and store filenames
train_words, train_filenames = extract_text_from_folder(train_folder)

# Save processed words and filenames
with open("train_words.pkl", "wb") as f:
    pickle.dump(train_words, f)

with open("train_filenames.pkl", "wb") as f:
    pickle.dump(train_filenames, f)

print("\n\nText extraction and preprocessing completed. Train data saved!")
