import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("WineQT.csv")

# Drop Id column if present
if 'Id' in data.columns:
    data.drop('Id', axis=1, inplace=True)

# Binary classification
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = data.drop('quality', axis=1)
y = data['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = SVC(probability=True)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully")
model = SVC(probability=True)
