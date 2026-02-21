import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("../data/mitbih_train.csv", header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Feature extraction
features = []

for i in range(len(X)):
    signal = X.iloc[i]
    signal = (signal - np.mean(signal)) / np.std(signal)

    mean = np.mean(signal)
    std = np.std(signal)
    maximum = np.max(signal)
    minimum = np.min(signal)
    energy = np.sum(signal**2)

    peaks, _ = find_peaks(signal, height=0)
    peak_count = len(peaks)

    features.append([mean, std, maximum, minimum, energy, peak_count])

feature_df = pd.DataFrame(features, columns=[
    "Mean", "Std", "Max", "Min", "Energy", "Peak_Count"
])

labels = y.apply(lambda x: 0 if x == 0 else 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    feature_df, labels, test_size=0.2, random_state=42
)

# Train SVM
model = SVC(kernel="rbf", class_weight="balanced")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()
