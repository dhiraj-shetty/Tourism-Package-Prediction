# Data Preparation Script
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------------------------------------------------
# Step 1: Load the dataset directly from the Hugging Face data space
# -----------------------------------------------------------------------
DATASET_PATH = "hf://datasets/dhirajshetty/tourism-package-prediction/tourism.csv"
tourism_data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from Hugging Face.")
print(f"Shape: {tourism_data.shape}")

# -----------------------------------------------------------------------
# Step 2: Data Cleaning
# -----------------------------------------------------------------------

# Drop unnecessary columns
tourism_data.drop(columns=["CustomerID"], inplace=True)
print("Dropped 'CustomerID' column.")

# Fix Gender typo: "Fe Male" -> "Female"
tourism_data["Gender"] = tourism_data["Gender"].replace("Fe Male", "Female")
print("Fixed 'Fe Male' typo in Gender column.")

# Handle missing values - Numeric columns: fill with median
numeric_cols_with_missing = [
    "Age", "DurationOfPitch", "MonthlyIncome",
    "NumberOfFollowups", "NumberOfChildrenVisiting",
    "NumberOfTrips", "PreferredPropertyStar"
]
for col in numeric_cols_with_missing:
    if col in tourism_data.columns:
        tourism_data[col].fillna(tourism_data[col].median(), inplace=True)
        print(f"Filled missing values in '{col}' with median.")

# Handle missing values - Categorical columns: fill with mode
categorical_cols_with_missing = ["TypeofContact"]
for col in categorical_cols_with_missing:
    if col in tourism_data.columns:
        tourism_data[col].fillna(tourism_data[col].mode()[0], inplace=True)
        print(f"Filled missing values in '{col}' with mode.")

print(f"\nRemaining missing values: {tourism_data.isnull().sum().sum()}")

# -----------------------------------------------------------------------
# Step 3: Define features and target
# -----------------------------------------------------------------------
target = "ProdTaken"

# Numeric features
numeric_features = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome", "CityTier"
]

# Categorical features
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "MaritalStatus", "ProductPitched", "Designation"
]

# Define predictors and target
X = tourism_data[numeric_features + categorical_features]
y = tourism_data[target]

# -----------------------------------------------------------------------
# Step 4: Split the dataset into training and testing sets
# -----------------------------------------------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Stratify to maintain class balance
)

print(f"\nTrain set size: {Xtrain.shape[0]}")
print(f"Test set size: {Xtest.shape[0]}")

# Save the splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train/test splits saved locally.")

# -----------------------------------------------------------------------
# Step 5: Upload train and test datasets back to HF data space
# -----------------------------------------------------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="dhirajshetty/tourism-package-prediction",
        repo_type="dataset",
    )
print("Train/test datasets uploaded to Hugging Face.")
