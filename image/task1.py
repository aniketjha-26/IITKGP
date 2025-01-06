import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pdfplumber

# Function to extract text from a PDF file
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
            if text:  # Only include PDFs with successfully extracted text
                data.append({"file_name": file_name, "text": text})
    return pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    # Remove references and figure captions
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

# Train and evaluate a model
def train_model(X, y, model):
    model.fit(X, y)
    return model

# Predict publishability
def predict_publishability(models, X):
    # Get the number of rows in the sparse matrix
    num_samples = X.shape[0]
    
    predictions = []
    for model in models:
        predictions.append(model.predict(X))
    
    # Combine predictions: if any sub-model predicts "Non-Publishable", the paper is "Non-Publishable"
    final_predictions = [0 if any(preds[i] == 0 for preds in predictions) else 1 for i in range(num_samples)]
    
    return final_predictions


# Main function
def main():
    # Step 1: Load and preprocess data
    # Use relative path to the folder
    pdf_folder = os.path.join(os.getcwd(), "Papers")  # Adjust relative path as needed
    df = load_pdfs_to_dataframe(pdf_folder)
    print(f"Loaded {len(df)} papers.")

    # Preprocess the text
    df['text'] = df['text'].apply(preprocess_text)

    # Separate labeled and unlabeled data
    labeled_df = df[-15:]  # Last 15 papers are labeled
    unlabeled_df = df[:-15]  # Remaining papers are unlabeled

    # Assign labels: First 5 are Non-Publishable (0), next 10 are Publishable (1)
    labeled_df['label'] = [0] * 5 + [1] * 10

    # Step 2: Feature extraction
    X_train, vectorizer = extract_features(labeled_df['text'], train=True)

    # Step 3: Train models for different issues
    y = labeled_df['label']
    model_methodologies = train_model(X_train, y, LogisticRegression())
    model_arguments = train_model(X_train, y, RandomForestClassifier())
    model_claims = train_model(X_train, y, SVC(probability=True))

    # Step 4: Predict on unlabeled data
    X_unlabeled, _ = extract_features(unlabeled_df['text'], vectorizer, train=False)

    models = [model_methodologies, model_arguments, model_claims]
    predictions = predict_publishability(models, X_unlabeled)

    # Add predictions to the DataFrame
    unlabeled_df['publishability'] = predictions
    unlabeled_df['publishability_label'] = unlabeled_df['publishability'].apply(lambda x: "Publishable" if x == 1 else "Non-Publishable")

    # Step 5: Save results
    output_file = os.path.join(os.getcwd(), "predictions.csv")  # Save the predictions to a CSV file using relative path
    unlabeled_df[['file_name', 'publishability_label']].to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'.")

    # Evaluation (Optional, only works with labeled data)
    print("\nModel Evaluation Metrics (On Labeled Data):")
    labeled_predictions = predict_publishability(models, X_train)
    print(f"Accuracy: {accuracy_score(y, labeled_predictions):.2f}")
    print(f"Precision: {precision_score(y, labeled_predictions):.2f}")
    print(f"Recall: {recall_score(y, labeled_predictions):.2f}")
    print(f"F1-Score: {f1_score(y, labeled_predictions):.2f}")

if __name__ == "__main__":
    main()
