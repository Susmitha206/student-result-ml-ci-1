
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Create student dataset
data = {
    "attendance": [
        90, 85, 70, 60, 95, 40, 80, 75, 55, 88,
        92, 65, 78, 45, 82, 50, 98, 68, 73, 35,
        87, 62, 91, 48, 76, 84, 58, 96, 71, 43
    ],
    "internal_marks": [
        85, 78, 65, 50, 92, 30, 75, 68, 45, 80,
        88, 55, 72, 35, 79, 40, 96, 60, 67, 25,
        83, 52, 90, 32, 70, 81, 48, 94, 63, 28
    ],
    "assignment_marks": [
        90, 80, 70, 55, 95, 35, 78, 72, 50, 85,
        92, 60, 75, 40, 82, 45, 98, 65, 70, 30,
        88, 58, 93, 38, 73, 86, 52, 97, 68, 33
    ],
    "previous_score": [
        82, 75, 68, 52, 90, 35, 77, 70, 48, 84,
        89, 58, 74, 38, 80, 42, 95, 62, 69, 28,
        85, 55, 91, 34, 72, 83, 50, 96, 65, 31
    ]
}

df = pd.DataFrame(data)

# Define result: 1 means Pass, 0 means Needs Improvement
df["result"] = (
    (df["attendance"] >= 60) &
    (df["internal_marks"] >= 50) &
    (df["assignment_marks"] >= 50) &
    (df["previous_score"] >= 50)
).astype(int)

# Save dataset
df.to_csv("student_results.csv", index=False)

# Prepare features and target
X = df.drop("result", axis=1)
y = df["result"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save trained model
joblib.dump(model, "student_result_model.pkl")

# Save evaluation metrics
metrics = {
    "accuracy": round(float(accuracy), 4),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed!")
print("Accuracy:", metrics["accuracy"])
print("Training records:", metrics["training_records"])
print("Testing records:", metrics["testing_records"])
print("Model, dataset, and metrics saved successfully.")
