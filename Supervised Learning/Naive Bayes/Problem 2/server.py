from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import CategoricalNB

app = Flask(__name__)

HTML_TEMPLATE = open("index.html", "r").read()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Load uploaded Excel file
    file = request.files['excel']
    df = pd.read_excel(file)

    # 2. Preprocess data
    X = df[['Outlook', 'Temperature', 'Humidity', 'Windy']]
    y = df['Play']

    categorical_columns = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    column_transformer = ColumnTransformer(
        transformers=[('cat', OrdinalEncoder(), categorical_columns)],
        remainder='passthrough'
    )
    X_encoded = column_transformer.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. Train model
    model = CategoricalNB()
    model.fit(X_encoded, y_encoded)

    # 4. Collect test input
    test_instance = pd.DataFrame([{
        'Outlook': request.form['Outlook'],
        'Temperature': request.form['Temperature'],
        'Humidity': request.form['Humidity'],
        'Windy': request.form['Windy'] == 'True'
    }])

    # 5. Encode test instance and predict
    test_encoded = column_transformer.transform(test_instance)
    pred_encoded = model.predict(test_encoded)
    prediction = le.inverse_transform(pred_encoded)[0]

    return f"<h3>Prediction: {prediction}</h3><a href='/'>Go Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
