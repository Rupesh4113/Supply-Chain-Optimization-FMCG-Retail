"""
Unit and Integration Tests for FMCG / Retail Supply Chain Analytics Pipeline.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.data_generation import generate_supply_chain_data
from src.preprocessing import DataPreprocessor, aggregate_warehouse_profiles
from src.feature_engineering import SupplyChainFeatureEngineer
from src.clustering import WarehouseClusterer
from src.modeling import InventoryRiskClassifier
from src.evaluation import evaluate_model_performance
from src.recommendations import ReplenishmentRecommender
from src.simulation import SupplyReallocationSimulator


@pytest.fixture(scope="module")
def sample_raw_data():
    """Generate reproducible sample operational dataset (60 DC x 12 Months)."""
    return generate_supply_chain_data(n_warehouses=60, n_months=12, random_seed=101)


def test_data_generation_dimensions(sample_raw_data):
    """Verify generated dataset dimensions, facilities count, and timeline."""
    df = sample_raw_data
    assert len(df) == 60 * 12
    assert df["warehouse_id"].nunique() == 60
    assert df["date"].nunique() == 12
    assert "inventory_imbalance_risk" in df.columns


def test_data_generation_non_negative_constraints(sample_raw_data):
    """Verify physical non-negativity operational constraints."""
    df = sample_raw_data
    assert (df["monthly_demand"] >= 0).all()
    assert (df["monthly_supply"] >= 0).all()
    assert (df["current_inventory"] >= 0).all()
    assert (df["safety_stock"] >= 0).all()
    assert (df["warehouse_capacity"] > 0).all()


def test_target_class_distribution(sample_raw_data):
    """Ensure all three operational target states are present."""
    df = sample_raw_data
    target_classes = set(df["inventory_imbalance_risk"].unique())
    expected_classes = {"Understock / Shortage", "Balanced", "Overstock"}
    assert target_classes == expected_classes


def test_preprocessor_pipeline(sample_raw_data):
    """Verify preprocessor schema validation and transformations."""
    preprocessor = DataPreprocessor(impute_strategy="median", handle_outliers=True)
    valid, issues = preprocessor.validate_schema(sample_raw_data)
    assert valid is True
    assert len(issues) == 0
    
    df_transformed = preprocessor.fit_transform(sample_raw_data)
    assert "year" in df_transformed.columns
    assert "month" in df_transformed.columns
    assert "is_festive_quarter" in df_transformed.columns
    assert df_transformed.isnull().sum().sum() == 0


def test_feature_engineering_calculations(sample_raw_data):
    """Verify business feature mathematical formulas and logic."""
    fe = SupplyChainFeatureEngineer(benchmark_dsr=1.0)
    df_fe = fe.fit_transform(sample_raw_data)
    
    # DSR formula check
    expected_dsr = np.round(df_fe["monthly_demand"] / np.maximum(df_fe["monthly_supply"], 1.0), 4)
    np.testing.assert_allclose(df_fe["demand_to_supply_ratio"], expected_dsr, atol=1e-3)
    
    # Safety stock coverage check
    assert (df_fe["safety_stock_coverage_ratio"] >= 0).all()
    assert "shortage_gap_units" in df_fe.columns
    assert (df_fe["shortage_gap_units"] >= 0).all()


def test_warehouse_clustering(sample_raw_data):
    """Verify K-Means clustering convergence and dynamic profiling."""
    fe = SupplyChainFeatureEngineer(benchmark_dsr=1.0)
    df_fe = fe.fit_transform(sample_raw_data)
    df_profiles = aggregate_warehouse_profiles(df_fe)
    
    clusterer = WarehouseClusterer(k_range=(2, 5), random_state=42)
    eval_df = clusterer.evaluate_k_range(df_profiles)
    assert len(eval_df) == 4
    assert "silhouette_score" in eval_df.columns
    
    df_clustered = clusterer.fit_predict(df_profiles, k=3)
    assert "cluster_id" in df_clustered.columns
    assert "cluster_name" in df_clustered.columns
    assert df_clustered["cluster_id"].nunique() == 3


def test_classification_and_metrics(sample_raw_data):
    """Verify model training, test prediction, and metric computation."""
    fe = SupplyChainFeatureEngineer(benchmark_dsr=1.0)
    df_fe = fe.fit_transform(sample_raw_data)
    
    clf_engine = InventoryRiskClassifier(random_state=42)
    cv_df = clf_engine.train_and_benchmark(df_fe, test_size=0.25, cv_folds=3)
    assert len(cv_df) == 3 # LR, DT, RF
    assert (cv_df["CV_Accuracy"] > 0.65).all()
    
    # Test holdout evaluation
    X_test, y_test = clf_engine.test_data
    rf_pipe = clf_engine.fitted_pipelines["Random Forest"]
    eval_results = evaluate_model_performance(rf_pipe, X_test, y_test)
    
    assert eval_results["accuracy"] > 0.70
    assert eval_results["shortage_recall"] > 0.60
    assert eval_results["confusion_matrix"].shape == (3, 3)


def test_recommendation_and_simulation(sample_raw_data):
    """Verify replenishment priority assignment and supply reallocation balances."""
    fe = SupplyChainFeatureEngineer(benchmark_dsr=1.0)
    df_fe = fe.fit_transform(sample_raw_data)
    
    clf_engine = InventoryRiskClassifier(random_state=42)
    clf_engine.train_and_benchmark(df_fe, test_size=0.25, cv_folds=3)
    rf_pipe = clf_engine.fitted_pipelines["Random Forest"]
    
    # Recommendation engine
    latest_snapshot = df_fe[df_fe["date"] == df_fe["date"].max()].copy()
    recommender = ReplenishmentRecommender()
    recs_df = recommender.generate_recommendations(latest_snapshot, rf_pipe)
    
    assert len(recs_df) == 60
    assert "priority" in recs_df.columns
    assert set(recs_df["priority"].unique()).issubset({"Critical", "High", "Medium", "Low"})
    
    # Simulation engine
    latest_snapshot["predicted_risk"] = rf_pipe.predict(latest_snapshot)
    simulator = SupplyReallocationSimulator()
    sim_out = simulator.run_reallocation_simulation(latest_snapshot, predicted_risk_col="predicted_risk")
    
    assert sim_out["total_shortage_after"] <= sim_out["total_shortage_before"]
    assert sim_out["total_units_reallocated"] >= 0
    assert "simulated_assumptions" in sim_out
