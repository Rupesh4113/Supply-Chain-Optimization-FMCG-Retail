"""
Script to generate the six structured, reproducible Jupyter Notebooks for the Supply Chain Optimization case study.
"""

import json
import os

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def save_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {path}")

# ==============================================================================
# Notebook 1: Data Generation
# ==============================================================================
nb1_cells = [
    make_cell("markdown", """# 01. Data Generation & Synthetic Simulation Engine
## Supply Chain Optimization — FMCG / Retail
**Author**: Senior Data Scientist & AI Analytics Team  
**Environment**: Production Case Study

---

### Objective
This notebook demonstrates the high-fidelity operational data generation process for **60 FMCG distribution facilities** across 5 macro-regions in India over **24 historical operational months** (totaling 1,440 warehouse-month records).

In compliance with **Data Integrity Requirements**, all downstream models, metrics, and business recommendations are calculated dynamically from this seed-controlled dataset."""),
    make_cell("code", """import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generation import generate_supply_chain_data, save_raw_dataset

# Reproducibility seed
SEED = 42
sns.set_theme(style="whitegrid")"""),
    make_cell("markdown", """### 1. Execute Data Generation
Generate the 1,440-row operational transaction dataset."""),
    make_cell("code", """df_raw = save_raw_dataset(
    output_path="../data/raw/fmcg_supply_chain_raw.csv",
    n_warehouses=60,
    n_months=24,
    random_seed=SEED
)
print(f"Operational Dataset Shape: {df_raw.shape}")
df_raw.head()"""),
    make_cell("markdown", """### 2. Validate Operational Distributions & Target Classes
Verify that all 3 operational target states are realistically populated."""),
    make_cell("code", """risk_counts = df_raw["inventory_imbalance_risk"].value_counts()
risk_pcts = df_raw["inventory_imbalance_risk"].value_counts(normalize=True) * 100

summary_target = pd.DataFrame({
    "Record Count": risk_counts,
    "Percentage (%)": risk_pcts.round(2)
})
display(summary_target)"""),
    make_cell("code", """plt.figure(figsize=(8, 4))
sns.countplot(
    data=df_raw,
    x="inventory_imbalance_risk",
    palette=["#EF4444", "#10B981", "#F59E0B"]
)
plt.title("Target Distribution: Inventory Imbalance Risk State", fontsize=12, fontweight="bold")
plt.xlabel("Risk State")
plt.ylabel("Observation Count")
plt.show()"""),
    make_cell("markdown", """### Summary
The dataset has been generated with physical operational constraints (non-negativity, capacity thresholds, lead-time variance, and correlated shortage/overstock drivers) and saved to `data/raw/fmcg_supply_chain_raw.csv`.""")
]
save_notebook("01_data_generation.ipynb", nb1_cells)

# ==============================================================================
# Notebook 2: Data Preparation & Feature Engineering
# ==============================================================================
nb2_cells = [
    make_cell("markdown", """# 02. Data Preparation & Feature Engineering Pipeline
## Supply Chain Optimization — FMCG / Retail

---

### Objective
1. Validate incoming data schema and check for data leakage risks.
2. Execute missing value analysis and outlier treatment using `DataPreprocessor`.
3. Engineer core domain features:
   - **Demand-to-Supply Ratio (DSR)**: $\\text{Demand} / \\text{Supply}$
   - **Lead-Time Variability Index**: $\\sigma_L / L$
   - **Stockout Severity Frequency Score (SSFS)**
   - **Safety Stock Coverage & Inventory Days of Supply (DOS)**
   - **Reorder & Shortage Gaps**"""),
    make_cell("code", """import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor, aggregate_warehouse_profiles
from src.feature_engineering import SupplyChainFeatureEngineer, get_feature_documentation"""),
    make_cell("markdown", """### 1. Ingestion and Schema Validation"""),
    make_cell("code", """df_raw = pd.read_csv("../data/raw/fmcg_supply_chain_raw.csv")
preprocessor = DataPreprocessor(impute_strategy="median", handle_outliers=True)

valid, issues = preprocessor.validate_schema(df_raw)
print("Schema Validated:", valid)
if not valid:
    print("Schema Issues:", issues)"""),
    make_cell("markdown", """### 2. Preprocessing & Outlier Treatment"""),
    make_cell("code", """df_cleaned = preprocessor.fit_transform(df_raw)
print(f"Cleaned Data Dimensions: {df_cleaned.shape}")"""),
    make_cell("markdown", """### 3. Advanced Feature Engineering"""),
    make_cell("code", """fe = SupplyChainFeatureEngineer(benchmark_dsr=1.0)
df_engineered = fe.fit_transform(df_cleaned)

# Save processed dataset
df_engineered.to_csv("../data/processed/fmcg_supply_chain_engineered.csv", index=False)
print("Engineered Features Count:", len(fe.get_feature_names()))
df_engineered[fe.get_feature_names()].head()"""),
    make_cell("markdown", """### 4. Feature Dictionary Documentation"""),
    make_cell("code", """feature_docs = get_feature_documentation()
pd.DataFrame.from_dict(feature_docs, orient="index")"""),
    make_cell("markdown", """### 5. Warehouse Profile Aggregation
Aggregate transaction history into static facility-level profiles for clustering."""),
    make_cell("code", """df_profiles = aggregate_warehouse_profiles(df_engineered)
df_profiles.to_csv("../data/processed/warehouse_profiles.csv", index=False)
print(f"Aggregated Profiles for {len(df_profiles)} Facilities.")
df_profiles.head()""")
]
save_notebook("02_data_preparation.ipynb", nb2_cells)

# ==============================================================================
# Notebook 3: Exploratory Data Analysis
# ==============================================================================
nb3_cells = [
    make_cell("markdown", """# 03. Exploratory Data Analysis (EDA)
## Supply Chain Optimization — FMCG / Retail

---

### Objective
Perform comprehensive supply chain exploratory analytics:
- **Demand Analytics**: Trends, regional distributions, and seasonal peaks.
- **Supply Analytics**: Inbound volume vs. demand run-rate.
- **Inventory Analytics**: Turnover velocity, safety stock coverage, and capacity utilization.
- **Risk Analytics**: Stockout frequency and correlation matrix."""),
    make_cell("code", """import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")
df = pd.read_csv("../data/processed/fmcg_supply_chain_engineered.csv")
df.head()"""),
    make_cell("markdown", """### 1. Demand & Supply Trajectory Over Time"""),
    make_cell("code", """monthly_summary = df.groupby("date")[["monthly_demand", "monthly_supply"]].sum() / 1e3

plt.figure(figsize=(12, 5))
plt.plot(monthly_summary.index, monthly_summary["monthly_demand"], marker="o", label="Total Demand (k Units)", color="#1f77b4", linewidth=2.5)
plt.plot(monthly_summary.index, monthly_summary["monthly_supply"], marker="s", label="Total Supply (k Units)", color="#2ca02c", linestyle="--", linewidth=2.5)
plt.title("FMCG Network Monthly Demand vs. Inbound Supply (2023–2024)", fontsize=13, fontweight="bold")
plt.xlabel("Date (YYYY-MM)")
plt.ylabel("Volume (Thousands of Units)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()"""),
    make_cell("markdown", """### 2. Demand-to-Supply Ratio (DSR) by Region"""),
    make_cell("code", """plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df,
    x="region",
    y="demand_to_supply_ratio",
    palette="Set2"
)
plt.axhline(1.0, color="red", linestyle=":", label="Parity Equilibrium (DSR = 1.0)")
plt.title("Regional Demand-to-Supply Ratio Distributions", fontsize=12, fontweight="bold")
plt.legend()
plt.show()"""),
    make_cell("markdown", """### 3. Inventory Turnover vs. Stockout Severity"""),
    make_cell("code", """plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="inventory_turnover",
    y="stockout_severity_frequency_score",
    hue="inventory_imbalance_risk",
    palette={"Understock / Shortage": "#EF4444", "Balanced": "#10B981", "Overstock": "#F59E0B"},
    alpha=0.75
)
plt.title("Inventory Turnover Velocity vs. Stockout Severity Frequency Score", fontsize=12, fontweight="bold")
plt.show()"""),
    make_cell("markdown", """### 4. Correlation Matrix of Operational Drivers"""),
    make_cell("code", """corr_cols = [
    "monthly_demand", "monthly_supply", "demand_to_supply_ratio",
    "lead_time_days", "lead_time_variability", "supplier_reliability",
    "capacity_utilization", "inventory_turnover", "stockout_incidents",
    "safety_stock_coverage_ratio"
]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Operational Drivers Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()""")
]
save_notebook("03_eda.ipynb", nb3_cells)

# ==============================================================================
# Notebook 4: Warehouse Segmentation (Clustering)
# ==============================================================================
nb4_cells = [
    make_cell("markdown", """# 04. Warehouse Segmentation & K-Means Clustering
## Supply Chain Optimization — FMCG / Retail

---

### Objective
Segment distribution centers based on multi-year operational behavior:
1. Evaluate K=2..8 using **Elbow Method (Inertia)** and **Silhouette Scores**.
2. Determine mathematically optimal number of clusters.
3. Fit K-Means on normalized operational feature profiles.
4. Profile operational segments (e.g., High-Velocity Shortage Risk, Balanced Flow, Chronic Overstock)."""),
    make_cell("code", """import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.clustering import WarehouseClusterer

df_profiles = pd.read_csv("../data/processed/warehouse_profiles.csv")
clusterer = WarehouseClusterer(k_range=(2, 8), random_state=42)"""),
    make_cell("markdown", """### 1. Evaluate K via Elbow and Silhouette Scores"""),
    make_cell("code", """k_eval_df = clusterer.evaluate_k_range(df_profiles)
display(k_eval_df)"""),
    make_cell("code", """fig, ax1 = plt.subplots(figsize=(10, 4))

color = "tab:blue"
ax1.set_xlabel("Number of Clusters (K)", fontsize=11)
ax1.set_ylabel("Inertia (Elbow Method)", color=color, fontsize=11)
ax1.plot(k_eval_df["k"], k_eval_df["inertia"], marker="o", color=color, linewidth=2)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Silhouette Score", color=color, fontsize=11)
ax2.plot(k_eval_df["k"], k_eval_df["silhouette_score"], marker="s", color=color, linestyle="--", linewidth=2)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Elbow Method & Silhouette Score vs. Number of Clusters K", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()"""),
    make_cell("markdown", """### 2. Fit K-Means (K=3) & Dimensionality Reduction Mapping"""),
    make_cell("code", """df_clustered = clusterer.fit_predict(df_profiles, k=3)
df_clustered.to_csv("../data/processed/warehouse_profiles.csv", index=False)

plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df_clustered,
    x="pca_x",
    y="pca_y",
    hue="cluster_name",
    style="cluster_name",
    s=100,
    palette="tab10"
)
plt.title("2D PCA Projection of Warehouse Operational Clusters", fontsize=13, fontweight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()"""),
    make_cell("markdown", """### 3. Discovered Cluster Profiles & Operational Characteristics"""),
    make_cell("code", """summary_table = clusterer.get_cluster_summary()
display(summary_table[["Operational_Segment", "demand_to_supply_ratio_avg", "capacity_utilization_avg", "inventory_turnover_avg", "stockout_incidents_sum"]])""")
]
save_notebook("04_clustering.ipynb", nb4_cells)

# ==============================================================================
# Notebook 5: Classification Modeling & Interpretability
# ==============================================================================
nb5_cells = [
    make_cell("markdown", """# 05. Supervised Classification Modeling & Explainable AI
## Supply Chain Optimization — FMCG / Retail

---

### Objective
1. Benchmark supervised models:
   - **Logistic Regression (Multinomial)**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
2. Apply **Stratified 5-Fold Cross Validation** on 80% train partition to prevent data leakage.
3. Evaluate on 20% holdout test set: Accuracy, Macro F1, Shortage Recall & Precision.
4. Extract **Odds Ratios** and **MDI Feature Importances** for operational decision-making."""),
    make_cell("code", """import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.modeling import InventoryRiskClassifier
from src.evaluation import (
    evaluate_model_performance,
    generate_model_comparison_table,
    extract_logistic_regression_interpretability,
    extract_tree_feature_importances
)

df = pd.read_csv("../data/processed/fmcg_supply_chain_engineered.csv")"""),
    make_cell("markdown", """### 1. Cross-Validation Benchmarking"""),
    make_cell("code", """clf_engine = InventoryRiskClassifier(random_state=42)
cv_results_df = clf_engine.train_and_benchmark(df, test_size=0.20, cv_folds=5)
display(cv_results_df)"""),
    make_cell("markdown", """### 2. Out-of-Sample Holdout Evaluation & Model Selection Matrix"""),
    make_cell("code", """X_test, y_test = clf_engine.test_data
comparison_df, selected_model = generate_model_comparison_table(
    clf_engine.fitted_pipelines, X_test, y_test
)
display(comparison_df)
print(f"Production Selected Model: {selected_model}")"""),
    make_cell("markdown", """### 3. Detailed Metrics & Confusion Matrix for Selected Model"""),
    make_cell("code", """prod_pipeline = clf_engine.fitted_pipelines[selected_model]
eval_metrics = evaluate_model_performance(prod_pipeline, X_test, y_test)

print(f"Test Accuracy: {eval_metrics['accuracy']:.2%}")
print(f"Shortage Recall: {eval_metrics['shortage_recall']:.2%}")
print(f"Shortage Precision: {eval_metrics['shortage_precision']:.2%}")

display(eval_metrics["per_class_metrics"])"""),
    make_cell("code", """plt.figure(figsize=(6, 5))
sns.heatmap(
    eval_metrics["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=eval_metrics["classes"],
    yticklabels=eval_metrics["classes"]
)
plt.title(f"Holdout Confusion Matrix: {selected_model}", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()"""),
    make_cell("markdown", """### 4. Model Interpretability: Logistic Regression Odds Ratios"""),
    make_cell("code", """lr_pipe = clf_engine.fitted_pipelines["Logistic Regression"]
df_odds = extract_logistic_regression_interpretability(lr_pipe)
display(df_odds.head(10))"""),
    make_cell("markdown", """### 5. Feature Importance: Random Forest Classifier"""),
    make_cell("code", """rf_pipe = clf_engine.fitted_pipelines["Random Forest"]
df_imp = extract_tree_feature_importances(rf_pipe, top_n=12)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_imp, x="Importance", y="Feature_Clean", palette="viridis")
plt.title("Random Forest MDI Feature Importances", fontsize=12, fontweight="bold")
plt.show()""")
]
save_notebook("05_classification.ipynb", nb5_cells)

# ==============================================================================
# Notebook 6: Business Simulation & Replenishment
# ==============================================================================
nb6_cells = [
    make_cell("markdown", """# 06. Business Simulation & Replenishment Optimization
## Supply Chain Optimization — FMCG / Retail

---

### Objective
Translate machine learning risk probabilities into operational supply chain actions:
1. Generate warehouse-level **Replenishment Recommendations** with action plans, target quantities, and priority badges.
2. Execute **Supply Reallocation Simulation** pairing surplus and deficit facilities to eliminate stockouts and reduce excess holding capital.
3. Compute simulated ROI, holding cost savings, and shortage reduction percentages."""),
    make_cell("code", """import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.recommendations import ReplenishmentRecommender
from src.simulation import SupplyReallocationSimulator

df_engineered = pd.read_csv("../data/processed/fmcg_supply_chain_engineered.csv")
prod_pipeline = joblib.load("../models/random_forest_pipeline.joblib")

latest_month = df_engineered["date"].max()
df_latest = df_engineered[df_engineered["date"] == latest_month].copy().reset_index(drop=True)
print(f"Evaluating Snapshot for: {latest_month} ({len(df_latest)} Warehouses)")"""),
    make_cell("markdown", """### 1. Generate Replenishment Recommendations"""),
    make_cell("code", """recommender = ReplenishmentRecommender(shortage_prob_threshold=0.50)
df_recs = recommender.generate_recommendations(df_latest, prod_pipeline)

# Priority breakdown
display(df_recs["priority"].value_counts())"""),
    make_cell("code", """# Critical action facilities
critical_facilities = df_recs[df_recs["priority"] == "Critical"]
display(critical_facilities[["warehouse_id", "warehouse_name", "region", "priority", "recommended_action", "recommended_quantity", "explainable_reason"]])"""),
    make_cell("markdown", """### 2. Lateral Inventory Reallocation Simulation"""),
    make_cell("code", """simulator = SupplyReallocationSimulator(
    holding_cost_per_unit_month=2.50,
    stockout_penalty_per_unit=8.00,
    intra_region_transfer_cost_per_unit=0.45,
    inter_region_transfer_cost_per_unit=1.10
)
sim_results = simulator.run_reallocation_simulation(df_latest, predicted_risk_col="predicted_risk")

print(f"Shortage Units Reduction: {sim_results['shortage_reduction_pct']}%")
print(f"Overstock Units Reduction: {sim_results['overstock_reduction_pct']}%")
print(f"Total Units Reallocated: {sim_results['total_units_reallocated']:,}")
print(f"Holding Cost Savings: ${sim_results['holding_cost_savings']:,.2f}")
print(f"Net Financial Benefit: ${sim_results['net_financial_benefit']:,.2f}")
print(f"Facilities Participating: {sim_results['affected_warehouses_count']}")"""),
    make_cell("markdown", """### 3. Lateral Transfer Manifest Sample"""),
    make_cell("code", """display(sim_results["transfer_log"].head(10))"""),
    make_cell("markdown", """### 4. Before vs. After Impact Visualization"""),
    make_cell("code", """categories = ["Shortage Before", "Shortage After", "Overstock Before", "Overstock After"]
values = [
    sim_results["total_shortage_before"] / 1e3,
    sim_results["total_shortage_after"] / 1e3,
    sim_results["total_overstock_before"] / 1e3,
    sim_results["total_overstock_after"] / 1e3
]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(categories, values, color=["#d62728", "#ff9896", "#ff7f0e", "#ffbb78"])
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, y + 0.3, f"{y:.1f}k", ha="center", fontweight="bold")
plt.title("Lateral Inventory Reallocation Simulation Impact", fontsize=12, fontweight="bold")
plt.ylabel("Inventory Units (Thousands)")
plt.tight_layout()
plt.show()""")
]
save_notebook("06_business_simulation.ipynb", nb6_cells)

print("\nAll 6 notebooks created successfully!")
