from flask import Flask, request, jsonify
import os
import pandas as pd
import tempfile
from task1 import (
    extract_text_from_pdf,
    preprocess_text,
    extract_features,
    train_model,
    predict_publishability,
)  # Replace 'task1' with the filename of your logic file

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Initialize the Flask app
app = Flask(__name__)

# Load models and vectorizer
def load_labeled_data(folder_path):
    # Initialize an empty list to hold the data
    labeled_data = []

    # Get the first 5 non-publishable and remaining as publishable
    non_publishable_count = 5
    publishable_count = 10
    non_publishable_files = []
    publishable_files = []

    # Loop through each file in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            
            # Extract text from the PDF file
            text = extract_text_from_pdf(file_path)
            if text:  # Only include valid PDF files
                if len(non_publishable_files) < non_publishable_count:
                    non_publishable_files.append(file_name)
                    labeled_data.append({"file_name": file_name, "text": text, "label": 0})  # Label as non-publishable
                elif len(publishable_files) < publishable_count:
                    publishable_files.append(file_name)
                    labeled_data.append({"file_name": file_name, "text": text, "label": 1})  # Label as publishable

    # Ensure we have exactly the required number of labeled files
    assert len(non_publishable_files) == non_publishable_count, "Not enough non-publishable files!"
    assert len(publishable_files) == publishable_count, "Not enough publishable files!"

    # Convert the data to a DataFrame
    return pd.DataFrame(labeled_data)

def initialize_models_from_folder(folder_path):
    # Load labeled data from the folder (use relative path)
    folder_path = os.path.join(os.getcwd(), folder_path)  # Ensure folder_path is relative to current working directory
    labeled_data = load_labeled_data(folder_path)
    
    # Preprocess the text
    labeled_data['text'] = labeled_data['text'].apply(preprocess_text)
    
    # Feature extraction for training
    X_train, vectorizer = extract_features(labeled_data['text'], train=True)
    y = labeled_data['label']

    # Train models
    model_methodologies = train_model(X_train, y, LogisticRegression())
    model_arguments = train_model(X_train, y, RandomForestClassifier())
    model_claims = train_model(X_train, y, SVC(probability=True))

    return vectorizer, [model_methodologies, model_arguments, model_claims]

# Example usage: Pass the folder path containing labeled PDFs
folder_path = "doc_l"  # Folder path is relative to the script
vectorizer, models = initialize_models_from_folder(folder_path)

# Route: Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

# Route: Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded."}), 400

    files = request.files.getlist('files')
    
    if not files:
        return jsonify({"error": "No files found."}), 400

    # Extract text from each uploaded PDF
    data = []
    for file in files:
        if file.filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_path = temp_file.name

            text = extract_text_from_pdf(temp_path)
            os.remove(temp_path)  # Clean up temporary file

            if text:
                text = preprocess_text(text)
                data.append({"file_name": file.filename, "text": text})

    if not data:
        return jsonify({"error": "No valid PDFs found or text could not be extracted."}), 400

    # Create a DataFrame with the extracted text
    df = pd.DataFrame(data)

    # Extract features from the extracted text
    X_unlabeled, _ = extract_features(df['text'], vectorizer, train=False)

    # Make predictions using the trained models
    predictions = predict_publishability(models, X_unlabeled)

    # Add predictions to the DataFrame
    df['publishability'] = predictions
    df['publishability_label'] = df['publishability'].apply(lambda x: "Publishable" if x == 1 else "Non-Publishable")
     
    # Prepare results
    results = df[['file_name', 'publishability_label']].to_dict(orient='records')

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
