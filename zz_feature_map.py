from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class ZZFeatureMap:
    def __init__(self, num_qubits: int, feature_dimension: int, reps: int, entanglement: str):
        self.num_qubits = num_qubits
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.entanglement = entanglement
        self.params = ParameterVector('x', feature_dimension)
        self._circuit = None
    
    def __call__(self):
        if self._circuit is None:
            self._circuit = self.create()
        return self._circuit

    def create(self):
        qc = QuantumCircuit(self.feature_dimension)

        for r in range(self.reps):
            for q in range(self.feature_dimension):
                qc.h(q)
                qc.rz(self.params[q], q)

            if self.entanglement == 'full':
                pairs = [(i, j) for i in range(self.num_qubits) for j in range(i+1, self.num_qubits)]
            elif self.entanglement == 'linear':
                pairs = [(i, i+1) for i in range(self.num_qubits-1)]
            else:
                pairs = []

            for (i, j) in pairs:
                if i < self.feature_dimension and j < self.feature_dimension:
                    theta = self.params[i] * self.params[j]
                    qc.rzz(theta, i, j)
        
        return qc