import pandas as pd
import numpy as np
import plotext as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Membaca dataset dari file CSV
data = pd.read_csv("heart_cleveland_upload.csv")  # Ganti dengan path dataset Anda

# Memisahkan fitur dan label
X = data.drop('condition', axis=1)  # 'condition' adalah label target
y = data['condition']

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: Standardisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model menggunakan SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # Kernel RBF sebagai default
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)

print("\nEvaluasi Model:")
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Visualisasi Confusion Matrix menggunakan plotext (berbasis teks)
conf_matrix_flat = conf_matrix.flatten()
labels = ["TN", "FP", "FN", "TP"]
plt.bar(labels, conf_matrix_flat)
plt.title("Confusion Matrix (Counts)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Visualisasi Distribusi Prediksi
unique, counts = np.unique(y_pred, return_counts=True)
plt.bar(["No Disease", "Disease"], counts)
plt.title("Distribution of Predictions")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.show()

# Prediksi contoh baru
new_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]  # Contoh data baru

# Buat DataFrame untuk data baru dengan nama kolom yang sesuai
new_data_df = pd.DataFrame(new_data, columns=X.columns)

# Standardisasi data baru
new_data_scaled = scaler.transform(new_data_df)

# Prediksi
prediction = model.predict(new_data_scaled)
result_text = "Penyakit Jantung" if prediction[0] == 1 else "Tidak Penyakit Jantung"
print("\nPrediction for new data (0 = no disease, 1 = disease):", prediction[0])
print("Hasil Prediksi:", result_text)
