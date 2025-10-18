import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

def qsvc_predict_many(X=None, y=None, n_qubits=5, random_state=42, p_reps=2, entanglement='full', test_size=0.3):
    df = pd.read_csv('alzheimers_disease_data.csv')
    y = df['Diagnosis']
    X = df.drop(['Diagnosis', 'DoctorInCharge', 'PatientID'], axis=1)

    X = X.head(150)
    y = y.head(150)

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_scaled)

    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_reduced = pca.fit_transform(X_poly)

    feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_final = feature_scaler.fit_transform(X_reduced)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=test_size, random_state=random_state, stratify=y)

    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=p_reps, entanglement=entanglement)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    K_train = quantum_kernel.evaluate(x_vec=X_train)

    trained_model = SVC(kernel='precomputed')
    trained_model.fit(K_train, y_train)

    K_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
    y_pred_test = trained_model.predict(K_test)
    
    print("Model został wytrenowany!")
    print(f"Rozmiar zbioru testowego: {len(X_test)}")
    
    print(f"Accuracy na danych testowych: {accuracy_score(y_test, y_pred_test):.3f}")
    
    print("Confusion Matrix:")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(cm_test)
    
    print("\nClassification Report:")
    cr_test = classification_report(y_test, y_pred_test, target_names=['Zdrowy', 'Alzheimer'])
    print(cr_test)
    
    return trained_model, scaler, pca, feature_scaler, X_train, y_train

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