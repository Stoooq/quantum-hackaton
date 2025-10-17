import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

def qsvc_predict_many(X=None, y=None, n_qubits=5, random_state=42, p_reps=2, entanglement='full'):
    df = pd.read_csv('alzheimers_disease_data.csv')
    y = df['Diagnosis']
    X = df.drop(['Diagnosis', 'DoctorInCharge', 'PatientID'], axis=1)

    X = X.head(150)
    y = y.head(150)

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_final = feature_scaler.fit_transform(X_reduced)

    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=p_reps, entanglement=entanglement)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    K_train = quantum_kernel.evaluate(x_vec=X_final)

    trained_model = SVC(kernel='precomputed')
    trained_model.fit(K_train, y)

    y_pred_train = trained_model.predict(K_train)
    
    print("Model został wytrenowany!")
    print("\n=== WYNIKI EWALUACJI MODELU ===")
    print(f"Accuracy na danych treningowych: {accuracy_score(y, y_pred_train):.3f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred_train)
    print(cm)
    
    print("\nClassification Report:")
    cr = classification_report(y, y_pred_train, target_names=['Zdrowy', 'Alzheimer'])
    print(cr)
    print("=" * 40)
    
    return trained_model, scaler, pca, feature_scaler, X_final, y

def qsvc_predict_one(patient_data, trained_model, scaler, pca, feature_scaler, X_train_final):
    df = pd.read_csv('alzheimers_disease_data.csv')
    feature_names = df.drop(['Diagnosis', 'DoctorInCharge', 'PatientID'], axis=1).columns.tolist()
    
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_scaled = scaler.transform(patient_df)
    patient_reduced = pca.transform(patient_scaled)
    
    patient_final = feature_scaler.transform(patient_reduced)
    n_qubits = 5
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='full')
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    K_test = quantum_kernel.evaluate(x_vec=patient_final, y_vec=X_train_final)

    prediction = trained_model.predict(K_test)
    probability = trained_model.decision_function(K_test)
    
    return int(prediction[0]), float(probability[0])
