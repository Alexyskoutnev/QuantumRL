import pennylane as qml
import qiskit

qubits = 4
dev_aer = qml.device("qiskit.aer", wires=qubits)
dev_basicaer = qml.device("qiskit.basicaer", wires=qubits)
try:
    dev_ibmq = qml.device('qiskit.ibmq', wires=2, backend='ibm_kyoto')
except Exception as e:
    print(e)

@qml.qnode(dev_ibmq)
def my_quantum_circuit():
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

result = my_quantum_circuit()
print("Expectation value:", result)