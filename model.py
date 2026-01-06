import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("WineQT.csv")

# Drop unnecessary column
if "Id" in data.columns:
    data.drop("Id", axis=1, inplace=True)

# -----------------------------
# Quality Labeling (3 Classes)
# -----------------------------
def quality_label(q):
    if q >= 7:
        return 2   # Good
    elif q == 6:
        return 1   # Average
    else:
        return 0   # Bad

data["quality_label"] = data["quality"].apply(quality_label)

X = data.drop(["quality", "quality_label"], axis=1)
y = data["quality_label"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -----------------------------
# Train SVC Model (Multi-class)
# -----------------------------
model = SVC(probability=True)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Save Model & Scaler
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model and scaler saved successfully")
