# Supply Chain Optimization — FMCG / Retail
### Predictive Imbalance Classification, Facility Clustering & Autonomous Replenishment Reallocation Engine

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-Scikit--Learn%20%7C%20Streamlit-orange.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/status-Production--Ready-brightgreen.svg)]()

---

## 1. Project Overview

This repository demonstrates a complete, production-grade Data Science and Machine Learning case study tailored for enterprise **Fast-Moving Consumer Goods (FMCG) and Omnichannel Retail supply chains**.

Modern FMCG supply chain networks operate under tight service-level agreements (OTIF > 95%), rapid demand velocity, and high supplier lead-time volatility. Distribution networks frequently suffer from simultaneous, contradictory failure modes: **chronic stockouts at high-velocity regional depots** alongside **damaging capital lockup from overstocking at peripheral warehouses**.

This project implements an end-to-end analytical solution that:
1. **Audits Network Equilibrium**: Computes multi-echelon demand-versus-supply dynamics and Demand-to-Supply Ratios ($DSR$).
2. **Discovers Behavioral DC Segments**: Groups 60 distribution facilities using unsupervised **K-Means Clustering** evaluated via Elbow Inertia and Silhouette Analysis.
3. **Predicts Imbalance Risk Early**: Benchmarks multi-class supervised classifiers (**Random Forest**, **Logistic Regression**, **Decision Tree**) with Stratified 5-Fold Cross-Validation, achieving **93.40% holdout accuracy** and **95.92% recall on critical stockout events**.
4. **Delivers Explainable Insights**: Exposes **Logistic Regression Odds Ratios** and **MDI Gini Feature Importances** in plain operational language.
5. **Prescribes Replenishment Actions**: Automatically generates prioritized, batch-rounded replenishment purchase recommendations.
6. **Optimizes Lateral Supply Reallocation**: Simulates intra- and inter-regional stock transfers from surplus to deficit nodes, eliminating **100.0% of immediate shortage exposure** and capturing **\$737,825.00 in holding cost savings** (\$2.97M net benefit).
7. **Empowers Executives**: Delivers a full-featured, interactive **Streamlit Enterprise Control Tower**.

---

## 2. Business Problem & Operational Context

In large-scale retail distribution, inventory imbalances directly erode top-line revenue and gross margin:
- **Stockout Penalties**: Unfulfilled orders lead to cancelled purchase orders, lost retail shelf share, and SLA penalties.
- **Working Capital Trap**: Overstocked warehouses exhaust physical floor capacity, incur severe holding costs (capital financing, insurance, shrinkage, and obsolescence), and trigger costly emergency clearance markdowns.
- **Static Reorder Inefficiencies**: Traditional ERP/WMS rules apply static min-max buffers or static safety-stock formulas that fail to adapt to lead-time variability shocks, supplier reliability slips, or regional demand momentum.

### Strategic Questions Answered:
1. Which warehouses are actively at risk of stockouts before stock buffers are depleted?
2. Which warehouses are carrying dead, excess working capital?
3. What operational variables drive supply-demand friction across regional nodes?
4. How can existing network surplus be redistributed to neutralize shortages without waiting for long-lead manufacturing cycles?

---

## 3. Data Integrity Notice & Dataset Architecture

> **Data Integrity Declaration**: In accordance with enterprise governance standards, this project explicitly discloses that the underlying dataset is a **high-fidelity, seed-controlled synthetic operational simulation (`seed=42`)** modeling **60 distribution facilities** across 5 Indian macro-regions over **24 continuous operational months** (totaling 1,440 warehouse-month transaction records).
>
> All model validation statistics, confusion matrices, feature importances, and financial simulation metrics in this repository are **dynamically computed directly from the generated dataset**. Zero metrics or ROI figures are fabricated or hardcoded.

### Key Dataset Variables:
- **Spatial / Node Attributes**: `warehouse_id`, `warehouse_name`, `region` (North, West, South, East, Central), `city`, `warehouse_type` (Mega Hub, Regional DC, Urban Depot), `warehouse_capacity`.
- **Operational Flows**: `monthly_demand`, `monthly_supply`, `current_inventory`, `safety_stock`, `reorder_point`.
- **Logistics Dynamics**: `lead_time_days`, `lead_time_variability` ($\sigma_L$), `transit_delay_days`, `supplier_reliability` (OTD %).
- **Performance Indices**: `fulfillment_rate` (OTIF), `stockout_incidents`, `overstock_quantity`, `inventory_turnover`, `seasonal_demand_index`, `regional_consumption_momentum`.
- **Target Variable (`inventory_imbalance_risk`)**: Multi-class classification target:
  - `Understock / Shortage` (Critical risk of order disruption)
  - `Balanced` (Healthy inventory flow and SLA preservation)
  - `Overstock` (Surplus capital lockup and capacity congestion)

---

## 4. End-to-End System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │   Enterprise WMS / ERP Supply Chain Data Streams       │
       │     (60 Regional DCs × 24 Operational Months)          │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │    Data Ingestion, Schema Validation & Preprocessing   │
       │   (Median Imputation, Outlier Capping, Temporal Split) │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │           Advanced Feature Engineering Pipeline        │
       │   (DSR, Lead-Time Volatility, SSFS, DOS, Gap Metrics)  │
       └─────────────┬────────────────────────────┬─────────────┘
                     │                            │
                     ▼                            ▼
       ┌───────────────────────────┐ ┌──────────────────────────┐
       │   K-Means Clustering      │ │  Supervised Imbalance    │
       │ (Elbow + Silhouette Segs) │ │     Risk Classifier      │
       │ (High-Velocity, Balanced, │ │ (Logistic Reg, Decision  │
       │  Chronic Overstock Segs)  │ │  Tree, Random Forest)    │
       └─────────────┬─────────────┘ └────────────┬─────────────┘
                     │                            │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │          Replenishment Recommendation Engine           │
       │   (Action, Quantity, Priority: Critical/High/Med/Low)  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │        Supply Reallocation Simulation Optimizer        │
       │   (Surplus-to-Deficit Transfers, Freight vs. Holding)  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Executive Streamlit Decision-Support Dashboard     │
       │  (Overview, Warehouse Analytics, ML Engine, Transfers) │
       └────────────────────────────────────────────────────────┘
```

---

## 5. Feature Engineering Dictionary

| Feature Name | Formula / Derivation | Supply Chain Business Interpretation |
| :--- | :--- | :--- |
| `demand_to_supply_ratio` | $\frac{\text{Monthly Demand}}{\text{Monthly Supply}}$ | Primary flow balance. $>1.0$ indicates demand outpacing inbound replenishment; $<1.0$ indicates supply surplus. |
| `dsr_equilibrium_deviation` | $DSR - 1.0$ | Direct distance from parity equilibrium. |
| `lead_time_variability_index` | $\frac{\sigma_{\text{LeadTime}}}{\mu_{\text{LeadTime}}}$ | Coefficient of variation in replenishment cycle time; measures vendor reliability risk. |
| `stockout_severity_frequency_score` | $\text{Stockouts} \times (1 + (1 - \text{OTIF}) \times 5)$ | Weighted historical penalty reflecting both frequency of outages and customer delivery failure severity. |
| `warehouse_capacity_utilization` | $\frac{\text{Current Inventory}}{\text{Warehouse Capacity}}$ | Physical cubic storage strain. $>0.85$ triggers overflow warnings; $<0.35$ signals underutilization. |
| `safety_stock_coverage_ratio` | $\frac{\text{Current Inventory}}{\text{Target Safety Stock}}$ | Realized buffer health. $<1.0$ indicates safety stock breach into stockout zone; $>1.8$ indicates over-buffering. |
| `inventory_days_of_supply` (DOS) | $\frac{\text{Current Inventory}}{\text{Daily Demand Run-Rate}}$ | Operational runway in days before stock depletion without new receipts. |
| `demand_volatility_index` | $\frac{\sigma_{\text{Demand}}}{\mu_{\text{Demand}}}$ | Demand unpredictability driving the bullwhip effect. |
| `supply_gap_ratio` | $\frac{\text{Demand} - \text{Supply}}{\text{Demand}}$ | Relative supply deficit percentage. |
| `shortage_gap_units` | $\max(0, \text{Safety Stock} - \text{Current Inventory})$ | Absolute unit injection required to restore baseline SLA coverage. |
| `transit_risk_index` | $\text{Delay Days} \times (1 + (1 - \text{Supplier Reliability}) \times 3)$ | Logistics vulnerability compounding transit delays with vendor inconsistency. |

---

## 6. Warehouse Segmentation (K-Means Clustering)

Aggregated 24-month operational profiles for all 60 facilities were analyzed across multi-dimensional feature space ($DSR$, Turnover, Utilization, Lead-Time Variance, Stockouts, Reliability).

### Cluster Evaluation (Elbow & Silhouette Analysis):
| K Clusters | Inertia (Sum of Squares) | Silhouette Score | Analytical Decision |
| :---: | :---: | :---: | :---: |
| K = 2 | 224.61 | 0.5710 | Under-segmented (blends overstock and balanced) |
| **K = 3** | **139.72** | **0.5178** | **Selected Operational Benchmark (Elbow inflection point)** |
| K = 4 | 109.40 | 0.5487 | Fragmented subgroups |
| K = 5 | 81.90 | 0.5601 | Over-segmented |

### Discovered Operational Archetypes:
1. **Cluster 0 — Balanced Operational Flow**:
   - *Characteristics*: Average DSR $\approx 1.00$, high inventory turnover ($6.8 - 9.2$), high fulfillment OTIF ($> 95\%$), stable supplier lead times.
   - *Strategy*: Cyclical replenishment, continuous variance monitoring.
2. **Cluster 1 — High-Velocity Shortage Risk**:
   - *Characteristics*: Elevated DSR ($> 1.05$), high stockout incident frequency, high lead-time standard deviation ($\sigma_L > 4.5$ days), vulnerable safety buffers.
   - *Strategy*: Safety stock injection, vendor SLA renegotiation, priority inbound logistics.
3. **Cluster 2 — Chronic Overstock Buffers**:
   - *Characteristics*: Sub-unity DSR ($< 0.90$), low annualized turnover ($< 4.5$), elevated excess inventory occupying capacity.
   - *Strategy*: Purchase order freeze, clearance promotion, and source nodes for lateral transfers.

---

## 7. Predictive Modeling & Evaluation Matrix

Models were trained to classify each warehouse-month state into `Understock / Shortage`, `Balanced`, or `Overstock` using an 80/20 stratified split and evaluated via **5-Fold Stratified Cross-Validation**.

### Calculated Holdout Benchmark Matrix:
| Model Architecture | Holdout Accuracy | Macro Precision | Macro Recall | Macro F1 | Shortage Recall | Shortage F1 | Production Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (150 Trees)** | **93.40%** | **64.20%** | **63.85%** | **64.02%** | **95.92%** | **95.92%** | **Selected (Production Candidate)** |
| **Logistic Regression (Multinomial)** | 82.99% | 64.76% | 61.38% | 61.44% | 85.71% | 89.84% | Rejected (Higher Misclassification) |
| **Decision Tree (Max Depth 6)** | 76.74% | 63.05% | 53.53% | 57.80% | 85.71% | 89.36% | Rejected (Instability / Variance) |

> **Selection Rationale**: The **Random Forest Classifier** was selected as the enterprise production model because it achieved **93.40% test accuracy** and an outstanding **95.92% recall on the critical `Understock / Shortage` class**, ensuring that stockouts are detected before stockouts occur.

---

## 8. Model Interpretability & Explainable AI

### 1. Logistic Regression Odds Ratios ($e^\beta$ for Shortage Risk):
- **Overstock Quantity** ($\text{Odds Ratio} = 5.56$): A unit increase in unmanaged excess inventory variability strongly signals structural supply-demand mismatch.
- **Transit Delay Days** ($\text{Odds Ratio} = 2.83$): Every additional standard deviation in logistics delay increases the odds of facility shortage by **2.83×**.
- **Current Inventory** ($\text{Odds Ratio} = 0.36$): Higher on-hand stock position reduces the odds of shortage by **64%**.

### 2. Random Forest MDI Feature Importances:
```text
Safety Stock Coverage Ratio   ████████████████████ (0.0921)
Supply Gap Ratio              ██████████████████   (0.0862)
DSR Equilibrium Deviation     ████████████████     (0.0771)
Demand-to-Supply Ratio (DSR)  ██████████████       (0.0672)
Current Inventory Position    █████████████        (0.0638)
```

---

## 9. Replenishment Engine & Supply Reallocation Simulation

### Replenishment Logic:
- **Critical Priority**: Triggered when predicted shortage probability $p \ge 75\%$ or inventory breaches $50\%$ of safety stock. Prescribes immediate safety stock injection and expedited delivery.
- **High Priority**: Triggered when $50\% \le p < 75\%$ or $DSR > 1.0$. Prescribes accelerated inbound replenishment batch.
- **Overstock Throttling**: Triggered when overstock probability $p \ge 50\%$. Recommends purchase order scale-back and lateral transfer out.

### Supply Reallocation Simulation (2024-12 Operational Snapshot):
The simulation pairs **Surplus Warehouses** ($DSR < 0.90$, overstock risk) with **Deficit Warehouses** ($DSR > 1.10$, shortage risk), prioritizing intra-regional transfers to minimize freight overhead.

| Operational & Financial KPI | Before Reallocation | After Reallocation | Delta Impact |
| :--- | :---: | :---: | :---: |
| **Network Shortage Units** | 295,129 units | **0 units** | **-100.0% (Complete SLA Recovery)** |
| **Network Overstock Units** | 2,790,510 units | **2,495,380 units** | **-10.6% (Dead Capital Released)** |
| **Total Units Reallocated** | — | **295,130 units** | Lateral balancing across 27 facilities |
| **Estimated Holding Cost Savings** | — | — | **+\$737,825.00** |
| **Stockout Opportunity Loss Mitigated** | — | — | **+\$2,361,040.00** |
| **Net Financial Benefit (Post-Freight)** | — | — | **+\$2,966,056.50** |

*Simulated Financial Assumptions*: Holding cost = \$2.50/unit-month; Stockout penalty = \$8.00/unit; Intra-regional freight = \$0.45/unit; Inter-regional freight = \$1.10/unit.

---

## 10. Key Analytical Findings & Business Recommendations

1. **Safety Stock Dynamic Adjustment**: Static safety stock policies fail during festive demand surges. Safety stock must dynamically scale with $\sigma_L$ and seasonal demand velocity.
2. **Lead-Time Volatility as Prime Disruptor**: Facilities experiencing lead-time standard deviation $> 4.5$ days accounted for over 70% of total stockout events.
3. **Lateral Network Rebalancing Over New Procurement**: Rather than ordering new manufacturing batches with 20+ day lead times, 100% of imminent shortages were resolved in simulation by routing excess inventory from overstocked sister facilities.
4. **Cluster-Specific Procurement Cadence**: Apply differentiated ordering frequencies:
   - *High-Velocity facilities*: Weekly micro-replenishment with cross-docking.
   - *Balanced facilities*: Standard bi-weekly replenishment.
   - *Overstock facilities*: Dynamic replenishment freeze until turnover exceeds 5.5.

---

## 11. Technology Stack

- **Core Runtime**: Python 3.11 / 3.12
- **Data Engineering**: Pandas, NumPy, Scipy
- **Machine Learning**: Scikit-Learn (Pipelines, ColumnTransformer, StratifiedKFold, RandomForest, LogisticRegression, KMeans, PCA)
- **Data Visualization**: Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
- **Enterprise Web App**: Streamlit (with responsive CSS layout and KPI metric cards)
- **Quality Assurance**: Pytest (8 comprehensive unit tests covering all components)
- **Serialization**: Joblib, JSON

---

## 12. Project Structure

```text
Supply-Chain-Optimization-FMCG-Retail/
├── README.md                                 # Master enterprise documentation
├── requirements.txt                          # Production dependency specifications
├── .gitignore                                # Git ignore configurations
│
├── data/
│   ├── README.md                             # Data dictionary & synthetic methodology
│   ├── raw/
│   │   └── fmcg_supply_chain_raw.csv         # 1,440-row operational transaction dataset
│   └── processed/
│       ├── fmcg_supply_chain_engineered.csv  # Cleaned, validated, and feature-engineered data
│       ├── warehouse_profiles.csv            # 60-facility aggregated segmentation profiles
│       ├── test_predictions.csv              # Holdout test set predictions & actuals
│       └── latest_replenishment_recommendations.csv # Prescriptive replenishment actions
│
├── notebooks/
│   ├── 01_data_generation.ipynb              # Step-by-step synthetic generation
│   ├── 02_data_preparation.ipynb             # Cleaning, schema validation, and feature engineering
│   ├── 03_eda.ipynb                          # Exploratory demand/supply/inventory analytics
│   ├── 04_clustering.ipynb                   # K-Means Elbow & Silhouette evaluation
│   ├── 05_classification.ipynb               # Multi-model cross-validation & explainability
│   ├── 06_business_simulation.ipynb          # Reallocation simulation & financial ROI
│   ├── create_notebooks.py                   # Notebook generator automation script
│   └── verify_notebooks.py                   # Syntax verification runner
│
├── src/
│   ├── __init__.py                           # Package initialization
│   ├── data_generation.py                    # FMCG operational generator engine
│   ├── preprocessing.py                      # DataPreprocessor & aggregation transformers
│   ├── feature_engineering.py                # SupplyChainFeatureEngineer pipeline
│   ├── clustering.py                         # WarehouseClusterer (K-Means & profiling)
│   ├── modeling.py                           # InventoryRiskClassifier (LR, DT, RF)
│   ├── evaluation.py                         # Out-of-sample metrics, confusion matrix & odds ratios
│   ├── recommendations.py                    # ReplenishmentRecommender engine
│   ├── simulation.py                         # SupplyReallocationSimulator engine
│   └── pipeline.py                           # End-to-end automated orchestration pipeline
│
├── app/
│   └── streamlit_app.py                      # Multi-tab interactive enterprise control tower
│
├── models/
│   ├── decision_tree_pipeline.joblib         # Serialized Decision Tree model
│   ├── logistic_regression_pipeline.joblib   # Serialized Logistic Regression model
│   └── random_forest_pipeline.joblib         # Serialized Random Forest model
│
├── reports/
│   ├── pipeline_metrics.json                 # Dynamically exported operational metrics
│   └── figures/                              # High-resolution report figures
│       ├── fig1_demand_vs_supply_trend.png
│       ├── fig2_dsr_by_risk_class.png
│       ├── fig3_confusion_matrix.png
│       ├── fig4_feature_importance.png
│       └── fig5_reallocation_impact.png
│
└── tests/
    └── test_pipeline.py                      # 8 unit and integration tests (100% passing)
```

---

## 13. How to Run & Reproduce

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/Rupesh4113/Supply-Chain-Optimization-FMCG-Retail.git
cd Supply-Chain-Optimization-FMCG-Retail

# Create virtual environment with uv or python venv
uv venv .venv --python 3.12
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux / macOS

# Install required packages
uv pip install -r requirements.txt
```

### 2. Execute End-to-End Analytics Pipeline
Run the master pipeline script to generate data, preprocess, engineer features, cluster, train models, simulate reallocation, and save figures:
```bash
python -m src.pipeline
```

### 3. Run Automated Unit Tests
Verify pipeline integrity with pytest:
```bash
pytest tests/test_pipeline.py -v
```

### 4. Launch Interactive Streamlit Application
Start the executive dashboard:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 14. Project Limitations & Future Enhancements

### Limitations:
1. **Simulation Model**: Operational transactions are simulated based on realistic FMCG heuristics. In live deployment, integration with live SAP/Oracle WMS tables via Kafka/Airflow would be required.
2. **Static Freight Costs**: Reallocation freight costs are parameterized as constant per-unit metrics rather than road/rail freight distance-matrix rates.
3. **Monthly Granularity**: Analysis is aggregated at monthly warehouse level; daily SKU-level replenishment would provide higher precision.

### Future Enhancements:
1. **SKU-Level Deep Forecasting**: Incorporate Temporal Fusion Transformers (TFT) or LightGBM for hierarchical daily SKU demand forecasting.
2. **Linear Programming / MILP Route Optimization**: Connect lateral reallocation with PuLP or SciPy `milp` to solve multi-echelon vehicle routing with capacity constraints.
3. **Automated ERP Webhook Dispatch**: Integrate REST API triggers to push approved replenishment purchase orders directly to SAP/Blue Yonder ERP systems.

---

## 15. Senior Data Scientist Portfolio Summary

> **Case Study Title**: **Supply Chain Optimization — FMCG / Retail Analytics & AI Engine**  
> **Summary**: Engineered an end-to-end predictive and prescriptive supply chain analytics platform that audits multi-facility demand-versus-supply parity, discovers operational clusters via K-Means segmentation, predicts inventory imbalance risk with 93.40% holdout accuracy and 95.92% critical shortage recall using Random Forest, and translates machine learning predictions into automated replenishment and lateral inventory reallocation recommendations that eliminated 100.0% of immediate network shortages and freed \$737,825.00 in holding capital across 60 simulated distribution centers.
