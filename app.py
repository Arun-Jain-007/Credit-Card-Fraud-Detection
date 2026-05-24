from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load ML model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')


# MANUAL TRANSACTION ANALYSIS
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get form data
        amount = float(request.form['amount'])

        transaction_type = request.form['transaction_type']
        location = request.form['location']
        time = request.form['time']
        device = request.form['device']

        # Risk score logic
        risk_score = 0

        # Amount Risk
        if amount > 50000:
            risk_score += 40

        elif amount > 10000:
            risk_score += 20

        # Transaction Type Risk
        if transaction_type == 'online':
            risk_score += 25

        elif transaction_type == 'upi':
            risk_score += 15

        # Location Risk
        if location == 'international':
            risk_score += 30

        # Time Risk
        if time == 'night':
            risk_score += 15

        # Device Risk
        if device == 'mobile':
            risk_score += 10

        # Final Prediction
        if risk_score >= 50:

            result = f'''
            <h2>⚠ Fraudulent Transaction Detected</h2>

            <br>

            🚨 Fraud Risk Score: {risk_score}%

            <br><br>

            🔍 High-risk transaction pattern identified.
            '''

        else:

            result = f'''
            <h2>✅ Genuine Transaction</h2>

            <br>

            🟢 Fraud Risk Score: {risk_score}%

            <br><br>

            ✔ Transaction appears safe.
            '''

        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


# CSV FILE UPLOAD ANALYSIS
@app.route('/upload', methods=['POST'])
def upload():

    try:

        file = request.files['file']

        # Read CSV
        data = pd.read_csv(file)

        # Remove target column if present
        if 'Class' in data.columns:
            data = data.drop('Class', axis=1)

        # Scale data
        scaled_data = scaler.transform(data)

        # ML prediction
        predictions = model.predict(scaled_data)

        # Count fraud/genuine
        fraud_count = sum(predictions == 1)
        genuine_count = sum(predictions == 0)

        # Total rows
        total_transactions = len(predictions)

        # Fraud percentage
        fraud_percentage = round(
            (fraud_count / total_transactions) * 100,
            2
        )

        # Result output
        result = f'''
        <h2>📊 Fraud Analysis Report</h2>

        <br>

        📁 Total Transactions: {total_transactions}

        <br><br>

        ⚠ Fraud Transactions: {fraud_count}

        <br><br>

        ✅ Genuine Transactions: {genuine_count}

        <br><br>

        🚨 Fraud Percentage: {fraud_percentage}%

        <br><br>

        🔍 System Status: Analysis Completed Successfully
        '''

        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"CSV Upload Error: {str(e)}"
        )


# RUN FLASK APP
if __name__ == "__main__":

    app.run(debug=True)