ECG Signal Classification Using Machine Learning

Objective
Build a machine learning system to classify ECG signals as Normal or Abnormal.

Dataset
MIT-BIH Arrhythmia Dataset (Kaggle version)

Project Overview
This project implements an end-to-end ECG signal classification pipeline using Python.
Raw ECG signals are preprocessed, important statistical and signal-based features are
extracted, and machine learning models are trained to detect abnormal heartbeats.

Approach
1. Loaded ECG data using Pandas.
2. Normalized ECG signals to remove scale differences.
3. Extracted features from each signal:
   - Mean
   - Standard Deviation
   - Maximum
   - Minimum
   - Energy
   - Peak Count
4. Converted multi-class labels into binary classes:
   Normal (0) and Abnormal (1).
5. Split data into training and testing sets.
6. Trained Logistic Regression and Support Vector Machine (SVM) models.
7. Handled class imbalance using class_weight='balanced'.
8. Evaluated model performance using accuracy and confusion matrix.

Results
The SVM model was able to detect abnormal ECG beats with approximately 69% recall.
A confusion matrix is saved in the results folder.

Folder Structure
ECG_Project/
│
├── code/
│   └── day1_ecg.py
├── data/
│   └── mitbih_train.csv
├── results/
│   └── confusion_matrix.png
└── README.md

Tools Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy

Conclusion
This mini-project demonstrates the application of machine learning and signal
processing techniques to healthcare data. It provided hands-on experience with
biomedical signal analysis, feature engineering, model training, and evaluation.

Author
Arshi Khan