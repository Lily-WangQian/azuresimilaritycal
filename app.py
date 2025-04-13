import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import re
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app and configure upload folder
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Load pre-trained BERT model for sentence embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Hugging Face pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Custom stopwords
custom_stopwords = {'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'on', 'for', 'with', 'as', 'by', 'it', 'an'}

# Helper functions
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(file_path) as pdf:
        return ' '.join(page.extract_text() for page in pdf.pages if page.extract_text())

def clean_text(text):
    """Clean and preprocess text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = ' '.join(word for word in text.split() if word.lower() not in custom_stopwords)
    return text.strip()

def extract_tfidf_keywords(text, top_n=5):
    """Extract top TF-IDF keywords."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def extract_contextual_keywords(text, top_n=5):
    """Extract top contextual keywords using sentence embeddings."""
    sentences = text.split('.')
    embeddings = bert_model.encode(sentences, convert_to_tensor=True)
    scores = []
    for i, sentence in enumerate(sentences):
        similarity = util.pytorch_cos_sim(embeddings[i], embeddings).mean().item()
        scores.append((sentence.strip(), similarity))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in scores[:top_n]]

def summarize_text(text):
    """Summarize text using BART model from Hugging Face."""
    try:
        # Ensure text is not too long for BART input
        max_input_length = 1024  # BART max token limit
        text = text[:max_input_length]  # Truncate if necessary
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error in summarizing: {str(e)}"

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity."""
    return util.cos_sim(embedding1, embedding2).item()

@app.route('/')
def index():
    return render_template('index.html', results=None, keyword_results=None, summary_results=None, error=None)

@app.route('/compare', methods=['POST'])
def compare():
    files = request.files.getlist('files')

    if len(files) < 2 or len(files) > 10:
        return render_template('index.html', results=None, keyword_results=None, summary_results=None, error="Please upload between 2 and 10 documents.")

    file_paths = []
    filenames = []

    for file in files:
        if file.filename == '':
            continue
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        file_paths.append(file_path)
        filenames.append(file.filename)

    if len(file_paths) < 2:
        return render_template('index.html', results=None, keyword_results=None, summary_results=None, error="Please upload between 2 and 10 valid documents.")

    try:
        texts = [clean_text(extract_text_from_pdf(file_path)) for file_path in file_paths]
        embeddings = [bert_model.encode(text, convert_to_tensor=True) for text in texts]

        similarity_results = []
        keyword_results = []
        summary_results = []

        for i in range(len(files)):
            for j in range(i + 1, len(files)):  
                similarity = calculate_similarity(embeddings[i], embeddings[j])
                similarity_results.append({
                    'file1': filenames[i],
                    'file2': filenames[j],
                    'similarity': round(similarity, 4),
                })

        for i, text in enumerate(texts):
            tfidf_keywords = extract_tfidf_keywords(text)
            contextual_keywords = extract_contextual_keywords(text)
            summary = summarize_text(text)
            keyword_results.append({
                'filename': filenames[i],
                'tfidf_keywords': tfidf_keywords,
                'contextual_keywords': contextual_keywords
            })
            summary_results.append({
                'filename': filenames[i],
                'summary': summary
            })

        return render_template('index.html', results=similarity_results, keyword_results=keyword_results, summary_results=summary_results, error=None)

    except Exception as e:
        return f"Error: {str(e)}"
    
    finally:
        for file_path in file_paths:
            os.remove(file_path)

            

if __name__ == '__main__':
    app.run(debug=True)

