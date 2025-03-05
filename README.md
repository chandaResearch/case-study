# case-study
case study on pre- post study
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load dataset
data_path = '/kaggle/input/impact-of-data-cleaning/diabetesRaw.csv'
data = pd.read_csv(data_path)

# Print column names before renaming
print("Before cleaning:", data.columns.tolist())

# Ensure all column names are lowercase and stripped of spaces
data.columns = data.columns.str.lower().str.strip()

# Print column names after renaming
print("After cleaning:", data.columns.tolist())

# Define target column
target_column = "outcome"

# Ensure the target column exists
if target_column not in data.columns:
    raise ValueError(f"❌ Column '{target_column}' not found! Available columns: {data.columns.tolist()}")

# Convert target column to numeric and handle missing values
data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
data[target_column].fillna(data[target_column].mode()[0], inplace=True)  # Fill NaN with mode
data[target_column] = data[target_column].astype(int)

# Select only numeric columns (ensuring target column is included)
numeric_data = data.select_dtypes(include=[np.number])
if target_column not in numeric_data.columns:
    numeric_data[target_column] = data[target_column]

# Handle missing values (replacing with median values)
imputer = SimpleImputer(strategy='median')
numeric_data.iloc[:, :-1] = imputer.fit_transform(numeric_data.iloc[:, :-1])

# Check for zero values in key medical columns
expected_cols = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}
available_cols = set(numeric_data.columns)
zero_cols = list(expected_cols.intersection(available_cols))

for col in zero_cols:
    zero_count = (numeric_data[col] == 0).sum()
    if zero_count > 0:
        print(f"⚠️ {col} contains {zero_count} zero values. Replacing zeros with median.")
        median_value = numeric_data[col].median()
        numeric_data[col] = numeric_data[col].replace(0, median_value)

# Split data into features and target
X = numeric_data.drop(columns=[target_column])
y = numeric_data[target_column]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (using stratify to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM Model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Make Predictions
y_pred = svm_model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print Results
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")
