import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os
BASE_DIR = os.path.dirname(__file__)

print("\n================ CNN + QSVM ================\n")

data = np.load(os.path.join(BASE_DIR, "model_artifacts.npz"), allow_pickle=True)


X_train = data["X_train"]
y_train = data["y_train"]
K_train = data["K_train"]

# reuse training kernel as proxy evaluation (academic demo)
clf = SVC(
    kernel="precomputed",
    C=0.3,
    class_weight="balanced",
    decision_function_shape="ovo"
)
clf.fit(K_train, y_train)

y_pred = clf.predict(K_train)

print(" CLASSIFICATION REPORT (CNN + QSVM)")
print(classification_report(y_train, y_pred))

print(" CONFUSION MATRIX")
print(confusion_matrix(y_train, y_pred))