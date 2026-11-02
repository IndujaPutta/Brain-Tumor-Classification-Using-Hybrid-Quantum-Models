import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)

# 🔑 IMPORTANT: argument name MUST be "inputs"
@qml.qnode(dev, interface="torch")
def qcnn_circuit(inputs, weights):
    # Quantum convolution
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # Entanglement
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (1,)}
qcnn_layer = qml.qnn.TorchLayer(qcnn_circuit, weight_shapes)


class QCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Linear(512, N_QUBITS)
        self.expand = nn.Linear(N_QUBITS, 512)

    def forward(self, x):
        # x shape: [B, 512]
        x = self.reduce(x)
        x = torch.tanh(x) * 3.1415

        outputs = []
        for i in range(x.shape[0]):  # batch-safe QCNN
            q_out = qcnn_layer(x[i])
            outputs.append(q_out)

        x = torch.stack(outputs)     # [B, 4]
        x = self.expand(x)
        return x
