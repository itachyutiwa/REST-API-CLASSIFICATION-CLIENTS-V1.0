from flask import Flask, jsonify, request
import joblib
import pandas as pd

from flask_cors import CORS

model = joblib.load("../model/RandomForest_One_Vs_All.pkl")
app = Flask(__name__)
CORS(app)

@app.route('/predictions', methods=['POST'])
def predict():
    # Récupération des données JSON envoyées en POST
    data = request.get_json(force=True)
    # Transformation des données en dataframe pandas
    X = pd.DataFrame.from_dict(data, orient='index').transpose()
    # Prédiction avec le modèle chargé
    y_pred = model.predict(X)
    # Renvoi de la prédiction sous forme de réponse JSON
    return jsonify(prediction=y_pred.tolist())

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)

