from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from newkiva import KivaPrediction
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ---------------- LOAD MODELS ----------------
model = joblib.load('RF_Reg.pkl')

model1 = joblib.load('SVM_model.pkl')
scaler = joblib.load('SVM_scaler.pkl')
feature_columns = joblib.load('svm_feature_columns.pkl')

obj = KivaPrediction()

# ---------------- MAPPINGS ----------------
careunit_mapping = {
    'MICU': 0,
    'SICU': 1,
    'CCU': 2,
    'TSICU': 3,
    'CSRU': 4
}

# ---------------- HOME ROUTE ----------------
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    total_cost = None
    loan = None

    if request.method == 'POST':
        first_careunit = request.form['first_careunit']
        last_careunit = request.form['last_careunit']
        first_wardid = int(request.form['first_wardid'])
        last_wardid = int(request.form['last_wardid'])
        age = int(request.form['age'])
        gender = request.form['gender']
        cp = int(request.form['cp'])
        ac = int(request.form['ac'])

        input_data = pd.DataFrame([{
            'FIRST_CAREUNIT': careunit_mapping[first_careunit],
            'LAST_CAREUNIT': careunit_mapping[last_careunit],
            'FIRST_WARDID': first_wardid,
            'LAST_WARDID': last_wardid,
            'AGE': age,
            'GENDER': 1 if gender == 'M' else 0
        }])

        prediction = round(model.predict(input_data)[0] * 24, 2)
        total_cost = (prediction * cp) + ac
        loan = "Do you want to apply for loan?"

    return render_template(
        'index.html',
        prediction=prediction,
        loan=loan,
        total_cost=total_cost
    )

# ---------------- KIVA ROUTE ----------------
@app.route('/kiva/<pred>/<total_cost>', methods=['GET', 'POST'])
def kiva(pred, total_cost):

    pred = float(pred)
    total_cost = float(total_cost)

    emi = None
    repayment_interval = None

    if request.method == 'POST':
        term = int(request.form['term_in_months'])
        loan_amt = int(request.form['loan_amount'])

        input_data = {
            'funded_amount': float(request.form['funded_amount']),
            'loan_amount': loan_amt,
            'term_in_months': term,
            'lender_count': int(request.form['lender_count']),
            'male_count': int(request.form['male_count']),
            'female_count': int(request.form['female_count']),
        }

        prediction = obj.predictKivaLoan(
            model1,
            scaler,
            input_data,
            feature_columns
        )

        repayment_map = {
            0: 'Irregular',
            1: 'Bullet',
            3: 'Monthly'
        }

        repayment_interval = repayment_map.get(
            prediction,
            'Loan Request Rejected'
        )

        emi = loan_amt / term if term != 0 else 0

    return render_template(
        'kiva.html',
        prediction=repayment_interval,
        pred=pred,
        total_cost=total_cost,
        emi=emi
    )

# ---------------- APP ENTRY ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)