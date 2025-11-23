# Fraud_detection
Fraud Detection System Using Machine Learning

This project detects fraudulent financial transactions using a machine learning model trained on transactional behavior data. The dataset is stored in PKL format, where each file contains one day of transaction records. The system can train a fraud model and run predictions via script or Streamlit web interface.

📌 Features

Train a fraud detection model using Random Forest

Support for PKL dataset folders or CSV files

Predict fraud probability for new transactions

Output results as a CSV with FRAUD_PROB and PRED_FRAUD

Optional Streamlit UI for easier interaction

🧠 Fraud Rules Simulated in Dataset

The dataset simulates fraud using these conditions:

Scenario	Description
High Amount Fraud	Transactions with amount > 220 are labeled fraud
Compromised Terminals	2 random terminals become fraudulent for 28 days
Compromised Customers	3 random customers have 1/3 transactions multiplied by 5 (fraud)
📂 Project Structure
fraud_detection/

│
├── artifacts/                # model will be saved here (fraud_model.joblib)

├── app.py                     # Streamlit UI

├── inference.py               # Script for batch predictions

├── train.py                   # Model training script

├── requirements.txt           # Dependencies

└── data/ (not included)       # PKL dataset folder (user must provide)


🚀 Installation
pip install -r requirements.txt

📊 Training the Model

Make sure your dataset folder contains multiple .pkl files.

Example:

data/
   2018-04-01.pkl
   2018-04-02.pkl
   ...


Run training:

python train.py --data data


Model output:

artifacts/fraud_model.joblib

🔎 Fraud Prediction (Inference)
1️⃣ Predict on PKL folder:
python inference.py --data data --out predictions.csv

2️⃣ Predict on a single PKL file:
python inference.py --data data/2018-04-01.pkl --out day1_predictions.csv

3️⃣ Predict on CSV file:
python inference.py --data transactions.csv --out predictions.csv

💻 Streamlit App

Run the UI:

streamlit run app.py

<img width="933" height="972" alt="Screenshot 2025-11-16 195704" src="https://github.com/user-attachments/assets/33e38707-80c3-4070-bd0f-b09d4de09b6c" />

Upload a CSV or PKL
<img width="918" height="870" alt="Screenshot 2025-11-16 195718" src="https://github.com/user-attachments/assets/fd1478e6-c4da-468c-ac0a-f0e2cf0eeda6" />
file and download fraud predictions.

📈 Output Example

The output CSV includes:

Column	Description
TX_AMOUNT	Transaction amount
CUSTOMER_ID	Encoded customer id
TERMINAL_ID	Encoded terminal id
FRAUD_PROB	Fraud probability (0–1)
PRED_FRAUD	1 = fraud, 0 = legitimate
🛠 Requirements
pandas
numpy
scikit-learn
joblib
streamlit

📜 License

This project is for educational and research purposes.
