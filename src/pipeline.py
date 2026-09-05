"""
End-to-End Execution Pipeline for FMCG / Retail Supply Chain Analytics.

Orchestrates data generation, preprocessing, feature engineering, clustering,
classification benchmarking, interpretability, replenishment, and reallocation simulation.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_generation import save_raw_dataset
from src.preprocessing import DataPreprocessor, aggregate_warehouse_profiles
from src.feature_engineering import SupplyChainFeatureEngineer
from src.clustering import WarehouseClusterer
from src.modeling import InventoryRiskClassifier
from src.evaluation import (
    evaluate_model_performance,
    generate_model_comparison_table,
    extract_logistic_regression_interpretability,
    extract_tree_feature_importances
)
from src.recommendations import ReplenishmentRecommender
from src.simulation import SupplyReallocationSimulator


def run_full_pipeline(
    raw_path: str = "data/raw/fmcg_supply_chain_raw.csv",
    processed_path: str = "data/processed/fmcg_supply_chain_engineered.csv",
    profiles_path: str = "data/processed/warehouse_profiles.csv",
    models_dir: str = "models/",
    figures_dir: str = "reports/figures/",
    metrics_path: str = "reports/pipeline_metrics.json",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Execute complete end-to-end data science and simulation workflow.
    """
    print("=" * 70)
    print("STARTING END-TO-END SUPPLY CHAIN OPTIMIZATION PIPELINE")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    # ---------------------------------------------------------
    # STEP 1: DATA GENERATION
    # ---------------------------------------------------------
    print("\n[Step 1/7] Generating synthetic FMCG operational dataset (60 DC × 24 Months)...")
    df_raw = save_raw_dataset(output_path=raw_path, n_warehouses=60, n_months=24, random_seed=random_seed)
    print(f" Raw dataset created: {df_raw.shape[0]} records, {df_raw.shape[1]} columns.")
    
    # ---------------------------------------------------------
    # STEP 2: PREPROCESSING & FEATURE ENGINEERING
    # ---------------------------------------------------------
    print("\n[Step 2/7] Executing schema validation, cleaning, and feature engineering...")
    preprocessor = DataPreprocessor(impute_strategy="median", handle_outliers=True)
    valid, issues = preprocessor.validate_schema(df_raw)
    if not valid:
        print(f" Schema validation warning: {issues}")
        
    df_cleaned = preprocessor.fit_transform(df_raw)
    
    fe = SupplyChainFeatureEngineer(benchmark_dsr=1.0)
    df_engineered = fe.fit_transform(df_cleaned)
    df_engineered.to_csv(processed_path, index=False)
    print(f" Feature engineering complete. Saved to: {processed_path}")
    
    # Aggregate static warehouse profiles for clustering
    df_profiles = aggregate_warehouse_profiles(df_engineered)
    df_profiles.to_csv(profiles_path, index=False)
    print(f" Warehouse aggregated profiles created for {len(df_profiles)} facilities.")

    # ---------------------------------------------------------
    # STEP 3: WAREHOUSE SEGMENTATION (K-MEANS CLUSTERING)
    # ---------------------------------------------------------
    print("\n[Step 3/7] Evaluating K-Means clustering (K=2..8) via Elbow & Silhouette...")
    clusterer = WarehouseClusterer(k_range=(2, 8), random_state=random_seed)
    k_eval_df = clusterer.evaluate_k_range(df_profiles)
    print(" K Evaluation Table:")
    print(k_eval_df.to_string(index=False))
    
    # Fit clustering on profiles
    df_profiles_clustered = clusterer.fit_predict(df_profiles, k=3)
    df_profiles_clustered.to_csv(profiles_path, index=False)
    
    # Map cluster assignments back to monthly transactions
    cluster_map = df_profiles_clustered.set_index("warehouse_id")["cluster_name"].to_dict()
    df_engineered["cluster_name"] = df_engineered["warehouse_id"].map(cluster_map)
    df_engineered.to_csv(processed_path, index=False)
    
    print("\n Discovered Cluster Profiles:")
    print(clusterer.get_cluster_summary()[["Operational_Segment"]])

    # ---------------------------------------------------------
    # STEP 4: SUPERVISED CLASSIFICATION BENCHMARKING
    # ---------------------------------------------------------
    print("\n[Step 4/7] Benchmarking classification models with Stratified 5-Fold CV...")
    clf_engine = InventoryRiskClassifier(random_state=random_seed)
    cv_benchmark_df = clf_engine.train_and_benchmark(df_engineered, test_size=0.20, cv_folds=5)
    print("\n 5-Fold Cross-Validation Benchmark:")
    print(cv_benchmark_df.to_string(index=False))
    
    # Holdout test set evaluation & model selection
    X_test, y_test = clf_engine.test_data
    comparison_df, selected_model = generate_model_comparison_table(
        clf_engine.fitted_pipelines, X_test, y_test
    )
    print("\n Out-of-Sample Holdout Comparison Matrix:")
    print(comparison_df.to_string(index=False))
    print(f"\n >>> Selected Production Model: '{selected_model}' <<<")
    
    # Save trained models
    saved_model_paths = clf_engine.save_models(output_dir=models_dir)
    print(f" Models serialized to: {models_dir}")

    # Detailed metrics for selected production model
    prod_pipeline = clf_engine.fitted_pipelines[selected_model]
    prod_eval = evaluate_model_performance(prod_pipeline, X_test, y_test)
    
    # Save test set predictions
    df_test_preds = X_test.copy()
    df_test_preds["actual_risk"] = y_test.values
    df_test_preds["predicted_risk"] = prod_eval["predictions"]
    df_test_preds.to_csv("data/processed/test_predictions.csv", index=False)

    # ---------------------------------------------------------
    # STEP 5: MODEL EXPLAINABILITY & INTERPRETABILITY
    # ---------------------------------------------------------
    print("\n[Step 5/7] Extracting model interpretability & feature importance...")
    lr_pipe = clf_engine.fitted_pipelines["Logistic Regression"]
    df_odds = extract_logistic_regression_interpretability(lr_pipe)
    
    rf_pipe = clf_engine.fitted_pipelines["Random Forest"]
    df_imp = extract_tree_feature_importances(rf_pipe, top_n=15)
    
    print("\n Top 5 Logistic Regression Odds Ratios for Shortage Risk:")
    print(df_odds[["Feature_Clean", "Coefficient", "Odds_Ratio", "Direction"]].head(5).to_string(index=False))
    
    print("\n Top 5 Random Forest Gini Feature Importances:")
    print(df_imp[["Feature_Clean", "Importance"]].head(5).to_string(index=False))

    # ---------------------------------------------------------
    # STEP 6: REPLENISHMENT & REALLOCATION SIMULATION
    # ---------------------------------------------------------
    print("\n[Step 6/7] Simulating replenishment recommendations and lateral reallocation...")
    # Isolate latest operational month snapshot (2024-12)
    latest_month = df_engineered["date"].max()
    df_latest = df_engineered[df_engineered["date"] == latest_month].copy().reset_index(drop=True)
    
    df_latest["predicted_risk"] = prod_pipeline.predict(df_latest)
    recommender = ReplenishmentRecommender(shortage_prob_threshold=0.50)
    df_recommendations = recommender.generate_recommendations(df_latest, prod_pipeline)
    df_recommendations.to_csv("data/processed/latest_replenishment_recommendations.csv", index=False)
    
    simulator = SupplyReallocationSimulator(
        holding_cost_per_unit_month=2.50,
        stockout_penalty_per_unit=8.00,
        intra_region_transfer_cost_per_unit=0.45,
        inter_region_transfer_cost_per_unit=1.10
    )
    sim_results = simulator.run_reallocation_simulation(df_latest, predicted_risk_col="predicted_risk")
    
    print(f"\n Simulation Outcome for {latest_month}:")
    print(f" - Shortage Units Before: {sim_results['total_shortage_before']:,.0f} -> After: {sim_results['total_shortage_after']:,.0f} (-{sim_results['shortage_reduction_pct']}%)")
    print(f" - Overstock Units Before: {sim_results['total_overstock_before']:,.0f} -> After: {sim_results['total_overstock_after']:,.0f} (-{sim_results['overstock_reduction_pct']}%)")
    print(f" - Units Reallocated: {sim_results['total_units_reallocated']:,}")
    print(f" - Estimated Holding Cost Savings: ${sim_results['holding_cost_savings']:,.2f}")
    print(f" - Net Financial Benefit: ${sim_results['net_financial_benefit']:,.2f}")
    print(f" - Affected Facilities: {sim_results['affected_warehouses_count']}")

    # ---------------------------------------------------------
    # STEP 7: GENERATE & EXPORT STATIC REPORT FIGURES
    # ---------------------------------------------------------
    print("\n[Step 7/7] Generating publication-grade report figures in reports/figures/...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    # Figure 1: Demand vs Supply Time Series
    plt.figure(figsize=(10, 5))
    monthly_trend = df_engineered.groupby("date")[["monthly_demand", "monthly_supply"]].sum() / 1e3
    plt.plot(monthly_trend.index, monthly_trend["monthly_demand"], marker="o", color="#1f77b4", label="Total Demand (k Units)", linewidth=2.2)
    plt.plot(monthly_trend.index, monthly_trend["monthly_supply"], marker="s", color="#2ca02c", linestyle="--", label="Total Supply (k Units)", linewidth=2.2)
    plt.title("FMCG Network Monthly Demand vs. Supply Trajectory (2023–2024)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Month", fontsize=11)
    plt.ylabel("Volume (Thousands of Units)", fontsize=11)
    plt.xticks(rotation=45)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_demand_vs_supply_trend.png"), dpi=300)
    plt.close()

    # Figure 2: DSR Distribution & Risk Categories
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df_engineered,
        x="inventory_imbalance_risk",
        y="demand_to_supply_ratio",
        palette=["#d62728", "#2ca02c", "#ff7f0e"]
    )
    plt.axhline(1.0, color="gray", linestyle=":", label="Parity Equilibrium (DSR = 1.0)")
    plt.title("Demand-to-Supply Ratio Distribution Across Risk Classes", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Ground Truth Risk State", fontsize=11)
    plt.ylabel("Demand-to-Supply Ratio (DSR)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_dsr_by_risk_class.png"), dpi=300)
    plt.close()

    # Figure 3: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        prod_eval["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=prod_eval["classes"],
        yticklabels=prod_eval["classes"]
    )
    plt.title(f"Confusion Matrix: {selected_model} (Holdout Test)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Risk Class", fontsize=10)
    plt.ylabel("Actual Risk Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_confusion_matrix.png"), dpi=300)
    plt.close()

    # Figure 4: Feature Importance
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=df_imp,
        x="Importance",
        y="Feature_Clean",
        palette="viridis"
    )
    plt.title("Top Feature Importances (Random Forest Model)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("MDI Feature Importance Score", fontsize=11)
    plt.ylabel("Operational Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_feature_importance.png"), dpi=300)
    plt.close()

    # Figure 5: Reallocation Impact Waterfall
    plt.figure(figsize=(8, 5))
    categories = ["Shortage Before", "Shortage After", "Overstock Before", "Overstock After"]
    values = [
        sim_results["total_shortage_before"] / 1e3,
        sim_results["total_shortage_after"] / 1e3,
        sim_results["total_overstock_before"] / 1e3,
        sim_results["total_overstock_after"] / 1e3
    ]
    colors = ["#d62728", "#ff9896", "#ff7f0e", "#ffbb78"]
    bars = plt.bar(categories, values, color=colors, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}k", ha="center", va="bottom", fontweight="bold")
    plt.title("Lateral Inventory Reallocation Impact on Network Imbalance", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Inventory Units (Thousands)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig5_reallocation_impact.png"), dpi=300)
    plt.close()

    # Export dynamic pipeline metrics JSON for Streamlit & README
    pipeline_summary = {
        "execution_timestamp": str(pd.Timestamp.now()),
        "warehouses_count": len(df_profiles),
        "total_records_count": len(df_engineered),
        "selected_production_model": selected_model,
        "model_accuracy": prod_eval["accuracy"],
        "model_macro_f1": prod_eval["macro_f1"],
        "shortage_detection_recall": prod_eval["shortage_recall"],
        "shortage_detection_precision": prod_eval["shortage_precision"],
        "comparison_table": comparison_df.to_dict(orient="records"),
        "cluster_k": 3,
        "clusters": clusterer.get_cluster_summary()[["Operational_Segment"]].to_dict(orient="index"),
        "simulation_latest_month": latest_month,
        "shortage_units_before": sim_results["total_shortage_before"],
        "shortage_units_after": sim_results["total_shortage_after"],
        "shortage_reduction_pct": sim_results["shortage_reduction_pct"],
        "overstock_reduction_pct": sim_results["overstock_reduction_pct"],
        "total_units_reallocated": sim_results["total_units_reallocated"],
        "holding_cost_savings": sim_results["holding_cost_savings"],
        "net_financial_benefit": sim_results["net_financial_benefit"],
        "affected_warehouses_count": sim_results["affected_warehouses_count"]
    }
    
    with open(metrics_path, "w") as f:
        json.dump(pipeline_summary, f, indent=2)
        
    print(f"\n Summary metrics exported to: {metrics_path}")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return pipeline_summary


if __name__ == "__main__":
    run_full_pipeline()
