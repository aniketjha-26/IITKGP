file_path = "d:\\coding\\Hackathon\\IITKGP-1\\image\\task2.py"  # Path to your problematic script

with open(file_path, "r", encoding="utf-8") as file:
    code = file.read()

# Replace non-breaking spaces with regular spaces
clean_code = code.replace("\u00A0", " ")

with open(file_path, "w", encoding="utf-8") as file:
    file.write(clean_code)

print("Non-printable characters have been replaced.")
