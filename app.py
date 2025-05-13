from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load the pickled model and preprocessing objects
with open('insurance_model.pkl', 'rb') as file:
    model_objects = pickle.load(file)

scaler = model_objects['scaler']
encoder_sex = model_objects['encoder_sex']
encoder_smoker = model_objects['encoder_smoker']
column_transformer = model_objects['column_transformer']
model = model_objects['best_lasso']  # or best_ridge / best_elasticnet

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    # Create DataFrame from inputs
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])

    # Apply manual preprocessing (same order as notebook!)
    # 1. Scale numerical
    input_df[['age', 'bmi', 'children']] = scaler.transform(input_df[['age', 'bmi', 'children']])

    # 2. Encode sex and smoker
    input_df['sex'] = encoder_sex.transform(input_df['sex'])
    input_df['smoker'] = encoder_smoker.transform(input_df['smoker'])

    # 3. Column transformer (region)
    transformed_input = column_transformer.transform(input_df)

    # 4. Make prediction
    prediction = model.predict(transformed_input)[0]

    return render_template('index.html', prediction_text=f'Predicted Insurance Charges: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
