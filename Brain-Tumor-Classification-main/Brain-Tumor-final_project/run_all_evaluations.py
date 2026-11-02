import os

print("\n==========================================")
print("   BRAIN TUMOR – HYBRID MODEL EVALUATION")
print("==========================================\n")

print(" Running CNN + QNN Evaluation\n")
os.system("python qnn/evaluate_qnn_cnn.py")

print("\n Running CNN + QCNN Evaluation\n")
os.system("python qcnn/evaluate_cnn_qcnn.py")

print("\n==========================================")
print("       SELECTED EVALUATIONS COMPLETED")
print("==========================================")