import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_track():
    df = pd.read_csv("data/heart_clean.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])
    
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {"model__C": [0.1, 1.0]}),
        "RandomForest": (RandomForestClassifier(random_state=42), {"model__n_estimators": [50, 100]})
    }
    
    mlflow.set_experiment("Heart_Disease_Risk_Analysis")
    best_auc = 0
    best_pipeline = None

    for model_name, (model, param_grid) in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            final_model = grid_search.best_estimator_
            y_pred = final_model.predict(X_test)
            y_proba = final_model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Explicitly force pickle serialization format to bypass skops validation checks
            mlflow.sklearn.log_model(final_model, artifact_path=model_name, serialization_format="pickle")
            
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_pipeline = final_model
                
    joblib.dump(best_pipeline, "artifacts/best_model_pipeline.pkl")
    print("Training complete. Saved best_model_pipeline.pkl")

if __name__ == '__main__':
    train_and_track()
