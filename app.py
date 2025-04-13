import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import re
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
custom_stopwords = {'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'on', 'for', 'with', 'as', 'by', 'it', 'an'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return ' '.join(page.extract_text() for page in pdf.pages if page.extract_text())

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return ' '.join(word for word in text.split() if word.lower() not in custom_stopwords)

def extract_tfidf_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def extract_contextual_keywords(text, top_n=5):
    sentences = text.split('.')
    embeddings = bert_model.encode(sentences, convert_to_tensor=True)
    scores = [(s.strip(), util.pytorch_cos_sim(e, embeddings).mean().item())
              for s, e in zip(sentences, embeddings)]
    top = sorted(scores, key=lambda x: x[1], reverse=True)
    return [s for s, _ in top[:top_n]]

def summarize_text(text):
    try:
        return summarizer(text[:1024], max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"Summary error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    files = request.files.getlist('files')
    if not files or len(files) < 2:
        return render_template('index.html', error="Please upload at least 2 PDF files.")

    filenames, file_paths = [], []

    for file in files:
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(path)
            filenames.append(fname)
            file_paths.append(path)

    try:
        texts = [clean_text(extract_text_from_pdf(fp)) for fp in file_paths]
        embeddings = [bert_model.encode(t, convert_to_tensor=True) for t in texts]

        results = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                results.append({'file1': filenames[i], 'file2': filenames[j], 'similarity': round(sim, 4)})

        keyword_results = [{
            'filename': filenames[i],
            'tfidf_keywords': extract_tfidf_keywords(texts[i]),
            'contextual_keywords': extract_contextual_keywords(texts[i])
        } for i in range(len(filenames))]

        summary_results = [{
            'filename': filenames[i],
            'summary': summarize_text(texts[i])
        } for i in range(len(filenames))]

        return render_template("index.html", results=results,
                               keyword_results=keyword_results,
                               summary_results=summary_results)

    except Exception as e:
        return render_template("index.html", error=f"Server error: {str(e)}")

if __name__ == "__main__":
    app.run()
