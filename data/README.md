# Supply Chain Optimization — FMCG / Retail: Data Documentation

## Overview
This directory houses raw and processed operational datasets for the **Supply Chain Optimization** enterprise case study.

In accordance with strict **Data Integrity Requirements**, this dataset is a high-fidelity, seed-controlled synthetic simulation (`seed=42`) modeling **60 FMCG/Retail distribution centers** across 5 macro-regions in India over **24 historical operational months** (totaling 1,440 warehouse-month observations).

> **Important Disclosure**: This dataset realistically reflects real-world operational distributions, seasonal curves, supplier lead-time volatilities, and inventory dynamics. All downstream machine learning metrics (accuracy, recall, F1-scores, ROC-AUC) and business simulation KPIs (stockout reduction, holding cost savings) are calculated directly and dynamically from this dataset.

---

## Directory Structure
```text
data/
├── README.md               # Dataset documentation & data dictionary
├── raw/
│   └── fmcg_supply_chain_raw.csv      # Unprocessed 1,440-record operational history
└── processed/
    ├── fmcg_supply_chain_engineered.csv # Cleaned & feature-engineered dataset
    ├── warehouse_profiles.csv           # Warehouse-level aggregated segmentation profiles
    └── test_predictions.csv             # Out-of-sample model predictions & metrics
```

---

## Data Dictionary

| Variable Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `record_id` | String | e.g. `REC-2023-01-W01` | Unique primary key for each warehouse-month observation |
| `date` | Date | `2023-01` to `2024-12` | Month of observation (YYYY-MM) |
| `warehouse_id` | String | `WH-01` to `WH-60` | Unique warehouse code |
| `warehouse_name` | String | e.g. `Depot North-A — Delhi/NCR` | Realistic illustrative warehouse and hub name |
| `region` | Categorical | North, West, South, East, Central | Macro-geographic operating region |
| `city` | String | Tier-1 / Tier-2 Indian cities | Location municipality |
| `warehouse_type` | Categorical | Mega Hub, Regional DC, Urban Depot | Operational tier and facility scale |
| `monthly_demand` | Float | Units (e.g. 5,000 - 95,000) | Actual customer and retail order demand in units |
| `monthly_supply` | Float | Units (e.g. 4,500 - 92,000) | Total inbound stock replenished in units |
| `current_inventory` | Float | Units | End-of-month on-hand inventory position |
| `safety_stock` | Float | Units | Target buffered safety stock level |
| `reorder_point` | Float | Units | Configured stock level triggering procurement |
| `lead_time_days` | Float | Days (3 - 35 days) | Average vendor fulfillment cycle time |
| `lead_time_variability` | Float | Days (Std Dev: 0.5 - 12 days) | Standard deviation in supplier transit & delivery time |
| `order_volume` | Integer | Orders count | Total replenishment and customer purchase orders |
| `fulfillment_rate` | Float | 0.0 - 1.0 (Percentage) | Ratio of orders fulfilled on-time and in-full (OTIF) |
| `stockout_incidents` | Integer | Events per month (0 - 15) | Count of zero-stock / backorder events recorded |
| `overstock_quantity` | Float | Units | Stock held beyond maximum storage thresholds |
| `warehouse_capacity` | Float | Units (e.g. 20,000 - 250,000) | Physical design storage capacity of facility |
| `capacity_utilization` | Float | 0.0 - 1.2+ (Ratio) | `current_inventory / warehouse_capacity` |
| `inventory_turnover` | Float | Ratio (Annualized) | Cost of goods sold / Average inventory velocity |
| `historical_demand_variance` | Float | Variance index | Demand variability over trailing 6 months |
| `seasonal_demand_index` | Float | 0.7 - 1.6 | Seasonal demand multiplier (e.g., Q4 festive peak) |
| `supplier_reliability` | Float | 0.50 - 0.99 | Percentage of on-time delivery from tier-1 suppliers |
| `transit_delay_days` | Float | Days (0 - 14 days) | Inbound logistics and customs transit delays |
| `sku_count` | Integer | Count (50 - 1,200) | Active distinct SKUs handled at facility |
| `regional_consumption_momentum` | Float | -0.3 to +0.5 | Trailing 3-month regional demand trend rate |
| `inventory_imbalance_risk` | Target | Categorical | Ground-truth risk state: `Understock / Shortage`, `Balanced`, `Overstock` |

---

## Target Generation Methodology

The classification target `inventory_imbalance_risk` is synthesized using deterministic operational principles combined with stochastic operational noise to prevent artificial separability:

1. **Shortage / Understock Condition**:
   - High Demand-to-Supply Ratio ($DSR > 1.18$) OR
   - On-hand inventory falls below 75% of required Safety Stock ($Inventory < 0.75 \times Safety Stock$) OR
   - Severe lead-time variability coupled with low supplier reliability ($Reliability < 0.78$ and $Lead Time Var > 5 days$).

2. **Overstock Condition**:
   - Low Demand-to-Supply Ratio ($DSR < 0.82$) AND
   - On-hand inventory exceeds 180% of safety buffer + reorder point ($Inventory > 1.8 \times Safety Stock$) AND
   - Sluggish inventory turnover ($Turnover < 4.0$).

3. **Balanced Condition**:
   - Inventory within comfortable safety buffer ($0.8 \le \frac{Inventory}{SafetyStock} \le 1.6$)
   - $0.85 \le DSR \le 1.15$
   - High fulfillment rate ($OTIF \ge 90\%$).

A calibrated probabilistic transition noise (~5-8%) is added to mirror real-world inventory noise (shrinkage, unrecorded damages, and phantom inventory).
