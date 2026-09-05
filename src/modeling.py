"""
Supervised Classification Modeling Pipeline for Inventory Imbalance Risk.

Benchmarks Logistic Regression, Decision Tree, and Random Forest using
Stratified K-Fold Cross-Validation, preventing data leakage via ColumnTransformer pipelines.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class InventoryRiskClassifier:
    """
    Production machine learning pipeline for multi-class inventory imbalance prediction.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.target_col: str = "inventory_imbalance_risk"
        
        # Define operational feature space
        self.numeric_features: List[str] = [
            "monthly_demand",
            "monthly_supply",
            "current_inventory",
            "safety_stock",
            "reorder_point",
            "lead_time_days",
            "lead_time_variability",
            "fulfillment_rate",
            "stockout_incidents",
            "overstock_quantity",
            "warehouse_capacity",
            "capacity_utilization",
            "inventory_turnover",
            "supplier_reliability",
            "transit_delay_days",
            "demand_to_supply_ratio",
            "dsr_equilibrium_deviation",
            "lead_time_variability_index",
            "stockout_severity_frequency_score",
            "safety_stock_coverage_ratio",
            "inventory_days_of_supply",
            "demand_volatility_index",
            "supply_gap_ratio",
            "reorder_gap",
            "shortage_gap_units",
            "transit_risk_index"
        ]
        
        self.categorical_features: List[str] = [
            "region",
            "warehouse_type"
        ]
        
        self.preprocessor: Optional[ColumnTransformer] = None
        self.models: Dict[str, Any] = {}
        self.fitted_pipelines: Dict[str, Pipeline] = {}
        self.cv_results: Dict[str, Dict[str, float]] = {}
        self.test_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
        self.classes_: Optional[np.ndarray] = None
        
        self._init_models()

    def _init_models(self):
        """Initialize candidate benchmarking classifiers."""
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=15,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state
            )
        }

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Construct ColumnTransformer for numerical scaling and categorical encoding."""
        avail_num = [c for c in self.numeric_features if c in X.columns]
        avail_cat = [c for c in self.categorical_features if c in X.columns]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), avail_num),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), avail_cat)
            ],
            remainder="drop"
        )
        return preprocessor

    def train_and_benchmark(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        cv_folds: int = 5
    ) -> pd.DataFrame:
        """
        Execute stratified train/test split, 5-fold cross-validation, and fit final models.
        """
        # Ensure target column exists
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe.")
            
        feature_cols = [c for c in self.numeric_features + self.categorical_features if c in df.columns]
        X = df[feature_cols].copy()
        y = df[self.target_col].copy()
        
        # Stratified 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        self.test_data = (X_test, y_test)
        self.classes_ = np.sort(y.unique())
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro"
        }
        
        benchmark_records = []
        
        for name, clf in self.models.items():
            preprocessor = self._build_preprocessor(X_train)
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])
            
            # 5-fold cross-validation on training partition
            cv_scores = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
            )
            
            mean_acc = float(np.mean(cv_scores["test_accuracy"]))
            mean_f1_macro = float(np.mean(cv_scores["test_f1_macro"]))
            mean_precision = float(np.mean(cv_scores["test_precision_macro"]))
            mean_recall = float(np.mean(cv_scores["test_recall_macro"]))
            
            self.cv_results[name] = {
                "cv_accuracy": round(mean_acc, 4),
                "cv_f1_macro": round(mean_f1_macro, 4),
                "cv_precision_macro": round(mean_precision, 4),
                "cv_recall_macro": round(mean_recall, 4)
            }
            
            # Fit final model on full training set
            pipeline.fit(X_train, y_train)
            self.fitted_pipelines[name] = pipeline
            
            benchmark_records.append({
                "Model": name,
                "CV_Accuracy": round(mean_acc, 4),
                "CV_Precision": round(mean_precision, 4),
                "CV_Recall": round(mean_recall, 4),
                "CV_Macro_F1": round(mean_f1_macro, 4)
            })
            
        return pd.DataFrame(benchmark_records)

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Generate class predictions using a specific trained model."""
        if model_name not in self.fitted_pipelines:
            raise ValueError(f"Model '{model_name}' has not been fitted.")
        return self.fitted_pipelines[model_name].predict(X)

    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Generate class probability estimates using a specific trained model."""
        if model_name not in self.fitted_pipelines:
            raise ValueError(f"Model '{model_name}' has not been fitted.")
        return self.fitted_pipelines[model_name].predict_proba(X)

    def save_models(self, output_dir: str = "models/") -> Dict[str, str]:
        """Serialize trained pipelines to disk."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        for name, pipe in self.fitted_pipelines.items():
            slug = name.lower().replace(" ", "_")
            path = os.path.join(output_dir, f"{slug}_pipeline.joblib")
            joblib.dump(pipe, path)
            paths[name] = path
        return paths
