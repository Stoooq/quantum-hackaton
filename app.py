from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from main import qsvc_predict_many, qsvc_predict_one

app = Flask(__name__)
CORS(app)

trained_model = None
scaler = None
poly = None
pca = None
feature_scaler = None
X_train_final = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        required_fields = [
            'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
            'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
            'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
            'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
            'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
            'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
            'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
            'Forgetfulness'
        ]
        
        patient_data = []
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Brakuje pola: {field}'}), 400
            patient_data.append(float(data[field]))
        
        if trained_model is None:
            return jsonify({'error': 'Model nie został jeszcze wytrenowany. Uruchom ponownie serwer.'}), 500
        
        prediction, confidence = qsvc_predict_one(patient_data, trained_model, scaler, poly, pca, feature_scaler, X_train_final)
        
        abs_confidence = abs(confidence)
        if abs_confidence < 0.5:
            risk_level = "Wysokie ryzyko"
            risk_description = "Niska pewność predykcji - zalecana dodatkowa diagnostyka"
        elif abs_confidence < 0.8:
            risk_level = "Średnie ryzyko"
            risk_description = "Umiarkowana pewność predykcji"
        else:
            risk_level = "Niskie ryzyko" if prediction == 0 else "Wysokie ryzyko"
            risk_description = "Wysoka pewność predykcji"
        
        result = {
            'prediction': prediction,
            'diagnosis': 'Alzheimer' if prediction == 1 else 'Zdrowy',
            'confidence': confidence,
            'abs_confidence': abs_confidence,
            'risk_level': risk_level,
            'risk_description': risk_description
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Trenowanie modelu...")
    trained_model, scaler, poly, pca, feature_scaler, X_train_final, y_train = qsvc_predict_many()
    print("Uruchamianie serwera...")
    app.run(debug=True, host='0.0.0.0', port=8000)
