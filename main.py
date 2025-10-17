import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.kernel_approximation import Nystroem
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

df = pd.read_csv('alzheimers_disease_data.csv')
y = df['Diagnosis']
X = df.drop(['Diagnosis', 'DoctorInCharge', 'PatientID'], axis=1)
X = X.head(150)
y = y.head(150)
scale = MinMaxScaler()
X = pd.DataFrame(scale.fit_transform(X), columns=X.columns)

def quantum_kernel_svc_pipeline(X, y,
                                n_qubits=5,
                                test_size=0.3,
                                random_state=42,
                                p_reps=2,
                                entanglement='full',):
    feature_names = X.columns.tolist()
    X = X.values
    y = np.asarray(y)

    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    loadings = (pca.components_.T) * np.sqrt(pca.explained_variance_)
    importance = np.sum(loadings**2, axis=1)
    importance_norm = importance / np.max(importance)

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'importance_norm': importance_norm
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(df.head())

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_reduced)

    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)
    
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=p_reps, entanglement=entanglement)
    
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    print("przed")
    K_train = quantum_kernel.evaluate(x_vec=X_train)
    print("za")

    K_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

    svc = SVC(kernel='precomputed')
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(acc)
    print(cm)
    print(cr)

quantum_kernel_svc_pipeline(X, y)
