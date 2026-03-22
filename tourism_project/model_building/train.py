# Model Training Script (Production)
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# MLflow configuration for production (runs on GitHub Actions)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

api = HfApi()

# -----------------------------------------------------------------------
# Step 1: Load the train and test data from the Hugging Face data space
# -----------------------------------------------------------------------
Xtrain_path = "hf://datasets/dhirajshetty/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/dhirajshetty/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/dhirajshetty/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/dhirajshetty/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print(f"Data loaded: Train={Xtrain.shape}, Test={Xtest.shape}")

# -----------------------------------------------------------------------
# Step 2: Define features
# -----------------------------------------------------------------------
numeric_features = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome", "CityTier"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "MaritalStatus", "ProductPitched", "Designation"
]

# -----------------------------------------------------------------------
# Step 3: Handle class imbalance
# -----------------------------------------------------------------------
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# -----------------------------------------------------------------------
# Step 4: Define preprocessing and model pipeline
# -----------------------------------------------------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid (larger grid for production)
param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100, 125, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------------------------------------------------
# Step 5: Train with MLflow tracking
# -----------------------------------------------------------------------
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"],
    })

    # Save the model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # -----------------------------------------------------------------------
    # Step 6: Register the best model in the Hugging Face model hub
    # -----------------------------------------------------------------------
    repo_id = "dhirajshetty/tourism-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_model_v1.joblib",
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Best model uploaded to Hugging Face Model Hub.")
