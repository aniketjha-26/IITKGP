
# def load_pdfs_to_dataframe(pdf_folder):
#     data = []
#     for file_name in os.listdir(pdf_folder):
#         if file_name.endswith(".pdf"):
#             file_path = os.path.join(pdf_folder, file_name)
#             text = extract_text_from_pdf(file_path)
#             if text:  # Only include PDFs with successfully extracted text
#                 data.append({"file_name": file_name, "text": text})
#     return pd.DataFrame(data)