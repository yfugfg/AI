import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from sklearn.preprocessing import MinMaxScaler

class QuantumML:
    def __init__(self, num_qubits=4, shots=1024):
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def create_quantum_circuit(self, features, weights):
        feature_map = ZZFeatureMap(self.num_qubits, reps=2)
        ansatz = RealAmplitudes(self.num_qubits, reps=2)
        
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        qc.measure_all()
        
        return qc
    
    def train(self, X, y, epochs=100, learning_rate=0.1):
        X_scaled = self.scaler.fit_transform(X)
        weights = np.random.randn(self.num_qubits * 4)
        
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X_scaled)):
                qc = self.create_quantum_circuit(X_scaled[i], weights)
                job = execute(qc, self.backend, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                
                prediction = self._counts_to_prediction(counts)
                error = prediction - y[i]
                loss += error ** 2
                
                weights -= learning_rate * error * self._compute_gradient(X_scaled[i], weights)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss/len(X_scaled):.4f}")
        
        return weights
    
    def predict(self, X, weights):
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for features in X_scaled:
            qc = self.create_quantum_circuit(features, weights)
            job = execute(qc, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            prediction = self._counts_to_prediction(counts)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _counts_to_prediction(self, counts):
        total = sum(counts.values())
        ones = sum(int(k, 2) * v for k, v in counts.items()) / total
        return ones / (2 ** self.num_qubits - 1)
    
    def _compute_gradient(self, features, weights):
        gradient = np.zeros_like(weights)
        delta = 1e-6
        
        for i in range(len(weights)):
            weights_plus = weights.copy()
            weights_minus = weights.copy()
            weights_plus[i] += delta
            weights_minus[i] -= delta
            
            qc_plus = self.create_quantum_circuit(features, weights_plus)
            qc_minus = self.create_quantum_circuit(features, weights_minus)
            
            job_plus = execute(qc_plus, self.backend, shots=self.shots)
            job_minus = execute(qc_minus, self.backend, shots=self.shots)
            
            result_plus = job_plus.result()
            result_minus = job_minus.result()
            
            counts_plus = result_plus.get_counts()
            counts_minus = result_minus.get_counts()
            
            pred_plus = self._counts_to_prediction(counts_plus)
            pred_minus = self._counts_to_prediction(counts_minus)
            
            gradient[i] = (pred_plus - pred_minus) / (2 * delta)
        
        return gradient

class QuantumRiskAnalyzer:
    def __init__(self, quantum_ml):
        self.quantum_ml = quantum_ml
    
    def analyze_portfolio_risk(self, portfolio_data, market_data):
        X = np.hstack([portfolio_data, market_data])
        
        if hasattr(self, 'risk_model_weights'):
            risk_scores = self.quantum_ml.predict(X, self.risk_model_weights)
        else:
            raise ValueError("Risk model not trained")
        
        return {
            'individual_risks': risk_scores.tolist(),
            'portfolio_risk': np.mean(risk_scores),
            'risk_distribution': {
                'low': np.sum(risk_scores < 0.3),
                'medium': np.sum((risk_scores >= 0.3) & (risk_scores < 0.7)),
                'high': np.sum(risk_scores >= 0.7)
            }
        }
    
    def train_risk_model(self, X, y):
        self.risk_model_weights = self.quantum_ml.train(X, y)
        return self.risk_model_weights
    
    def predict_market_crashes(self, market_indicators, threshold=0.7):
        if not hasattr(self, 'crash_model_weights'):
            raise ValueError("Crash prediction model not trained")
        
        crash_probabilities = self.quantum_ml.predict(market_indicators, self.crash_model_weights)
        crash_alarms = crash_probabilities > threshold
        
        return {
            'probabilities': crash_probabilities.tolist(),
            'alarms': crash_alarms.tolist(),
            'high_risk_areas': np.where(crash_alarms)[0].tolist()
        }
    
    def train_crash_prediction_model(self, X, y):
        self.crash_model_weights = self.quantum_ml.train(X, y)
        return self.crash_model_weights