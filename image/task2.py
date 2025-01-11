import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from pathway.xpacks.llm.vector_store import VectorStore
from pathway.xpacks.llm.document_store import DocumentStore

# Initialize Pathway stores
doc_store = DocumentStore()
vector_store = VectorStore()

# Function to extract text from PDFs
def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
            return text.strip() if text else None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Load PDFs into a DataFrame
def load_pdfs_to_dataframe(pdf_folder):
    data = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            text = extract_text_from_pdf(file_path)
            if text:
                data.append({"file_name": file_name, "text": text})
    return pd.DataFrame(data)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"(References|REFERENCES|Bibliography)[\s\S]+", "", text)
    text = re.sub(r"(Figure|Fig|TABLE|Table)\s?\d+.*", "", text, flags=re.IGNORECASE)
    return text

# Feature extraction
def extract_features(data, vectorizer=None, train=True):
    if train:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        features = vectorizer.fit_transform(data)
    else:
        features = vectorizer.transform(data)
    return features, vectorizer

# Train models
def train_model(X, y, model):
    model.fit(X, y)
    return model

# Predict publishability
def predict_publishability(models, X):
    num_samples = X.shape[0]
    predictions = [model.predict(X) for model in models]
    final_predictions = [0 if any(preds[i] == 0 for preds in predictions) else 1 for i in range(num_samples)]
    return final_predictions

# Store papers in DocumentStore
def store_papers_in_documentstore(df):
    for _, row in df.iterrows():
        doc_store.add_document({"file_name": row["file_name"], "text": row["text"]})

# Store conference profiles in VectorStore
def store_conference_profiles():
    conference_profiles = {
        "CVPR": "computer vision, deep learning, image processing",
        "NeurIPS": "machine learning, artificial intelligence, optimization",
        "EMNLP": "natural language processing, text analysis, transformers",
        "TMLR": "theory of machine learning, neural networks, optimization",
        "KDD": "data mining, big data, analytics, business intelligence"
    }
    for conference, description in conference_profiles.items():
        vector_store.add_embedding(conference, description)

# Compute similarity between papers and conferences
def compute_similarity(paper_texts):
    paper_embeddings = vector_store.generate_embeddings(paper_texts)
    conference_embeddings = vector_store.get_all_embeddings()
    similarities = cosine_similarity(paper_embeddings, conference_embeddings)
    return similarities

# Recommend conferences
def recommend_conferences(publishable_papers):
    recommendations = []
    for paper in publishable_papers:
        text = paper["text"]
        similarity_scores = compute_similarity([text])
        best_match_idx = similarity_scores.argmax()
        best_conference = vector_store.get_label_by_index(best_match_idx)
        best_score = similarity_scores[0, best_match_idx]
        recommendations.append((paper["file_name"], best_conference, best_score))
    return recommendations

# Main function
def main():
    # Load and preprocess data
    pdf_folder = r"D:\coding\Python\Papers"  # Update with your path
    df = load_pdfs_to_dataframe(pdf_folder)
    df["text"] = df["text"].apply(preprocess_text)

    # Store papers in DocumentStore
    store_papers_in_documentstore(df)

    # Separate labeled and unlabeled data
    labeled_df = df[-15:]  # Last 15 papers are labeled
    unlabeled_df = df[:-15]  # Remaining papers are unlabeled
    labeled_df["label"] = [0] * 5 + [1] * 10  # First 5 Non-Publishable, next 10 Publishable

    # Feature extraction
    X_train, vectorizer = extract_features(labeled_df["text"], train=True)
    y_train = labeled_df["label"]

    # Train models
    models = [
        train_model(X_train, y_train, LogisticRegression()),
        train_model(X_train, y_train, RandomForestClassifier()),
        train_model(X_train, y_train, SVC(probability=True))
    ]

    # Predict publishability for unlabeled data
    X_unlabeled, _ = extract_features(unlabeled_df["text"], vectorizer, train=False)
    unlabeled_predictions = predict_publishability(models, X_unlabeled)

    # Add publishability predictions
    unlabeled_df["publishability"] = unlabeled_predictions
    publishable_papers = unlabeled_df[unlabeled_df["publishability"] == 1]

    # Store conference profiles
    store_conference_profiles()

    # Recommend conferences
    recommendations = recommend_conferences(publishable_papers.to_dict("records"))

    # Save results
    results = []
    for paper, conference, score in recommendations:
        results.append({"Paper ID": paper, "Publishable": 1, "Conference": conference, "Rationale": f"High similarity score: {score:.2f}"})
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)

    print("Results saved to 'results.csv'.")

if _name_ == "_main_":
    main()
