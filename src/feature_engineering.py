"""
Feature Engineering Pipeline for FMCG / Retail Supply Chain Analytics.

Computes domain-specific supply chain features including Demand-to-Supply Ratio (DSR),
Lead-Time Variability Index, Stockout Severity Frequency Score, Safety Stock Coverage,
Inventory Days, and Shortage/Excess Gaps.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SupplyChainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer computing domain-specific operational
    and inventory indicators from raw WMS transaction records.
    """

    def __init__(self, benchmark_dsr: float = 1.0):
        self.benchmark_dsr = benchmark_dsr
        self.engineered_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit transformer (stateless for ratio computation, records output feature names)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute supply chain operational features.
        
        Returns:
            pd.DataFrame with original columns and appended engineered business features.
        """
        df = X.copy()
        
        # 1. Demand-to-Supply Ratio (DSR)
        # Avoid zero division with small epsilon
        supply_safe = np.maximum(df["monthly_supply"].values, 1.0)
        df["demand_to_supply_ratio"] = np.round(df["monthly_demand"].values / supply_safe, 4)
        
        # Deviation from ideal equilibrium DSR (1.0)
        df["dsr_equilibrium_deviation"] = np.round(df["demand_to_supply_ratio"].values - self.benchmark_dsr, 4)

        # 2. Lead-Time Variability Index (Coefficient of Variation of Lead Time)
        lead_safe = np.maximum(df["lead_time_days"].values, 1.0)
        df["lead_time_variability_index"] = np.round(df["lead_time_variability"].values / lead_safe, 4)

        # 3. Stockout Severity Frequency Score (SSFS)
        # Combines frequency of stockouts with fulfillment failure penalty
        unfulfilled_rate = np.maximum(0.0, 1.0 - df["fulfillment_rate"].values)
        df["stockout_severity_frequency_score"] = np.round(
            df["stockout_incidents"].values * (1.0 + unfulfilled_rate * 5.0), 3
        )

        # 4. Warehouse Capacity Utilization
        cap_safe = np.maximum(df["warehouse_capacity"].values, 1.0)
        df["warehouse_capacity_utilization"] = np.round(df["current_inventory"].values / cap_safe, 4)

        # 5. Safety Stock Coverage Ratio
        # Ratios < 1.0 indicate inventory has breached safety buffers into shortage zone
        ss_safe = np.maximum(df["safety_stock"].values, 1.0)
        df["safety_stock_coverage_ratio"] = np.round(df["current_inventory"].values / ss_safe, 3)

        # 6. Inventory Days / Days of Supply (DOS)
        daily_demand = np.maximum(df["monthly_demand"].values / 30.0, 1.0)
        df["inventory_days_of_supply"] = np.round(df["current_inventory"].values / daily_demand, 2)

        # 7. Demand Volatility (Coefficient of Variation)
        demand_safe = np.maximum(df["monthly_demand"].values, 1.0)
        demand_std = np.sqrt(np.maximum(0.0, df["historical_demand_variance"].values))
        df["demand_volatility_index"] = np.round(demand_std / demand_safe, 4)

        # 8. Supply Gap Ratio
        # Relative gap between inbound supply and demand
        df["supply_gap_ratio"] = np.round((df["monthly_demand"].values - df["monthly_supply"].values) / demand_safe, 4)

        # 9. Reorder Gap (Net buffer relative to reorder point)
        df["reorder_gap"] = np.round(df["current_inventory"].values - df["reorder_point"].values, 2)

        # 10. Excess Inventory Ratio (Overstock normalized by capacity)
        df["excess_inventory_ratio"] = np.round(df["overstock_quantity"].values / cap_safe, 4)

        # 11. Shortage Gap (Units needed to restore safety stock if breached)
        shortage_units = np.maximum(0.0, df["safety_stock"].values - df["current_inventory"].values)
        df["shortage_gap_units"] = np.round(shortage_units, 2)

        # 12. Transit Risk Multiplier
        # Combined penalty of transit delay and supplier unreliability
        reliability_deficit = np.maximum(0.0, 1.0 - df["supplier_reliability"].values)
        df["transit_risk_index"] = np.round(df["transit_delay_days"].values * (1.0 + reliability_deficit * 3.0), 3)

        # Store engineered feature names
        self.engineered_feature_names_ = [
            "demand_to_supply_ratio",
            "dsr_equilibrium_deviation",
            "lead_time_variability_index",
            "stockout_severity_frequency_score",
            "warehouse_capacity_utilization",
            "safety_stock_coverage_ratio",
            "inventory_days_of_supply",
            "demand_volatility_index",
            "supply_gap_ratio",
            "reorder_gap",
            "excess_inventory_ratio",
            "shortage_gap_units",
            "transit_risk_index"
        ]

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of newly generated feature names."""
        return self.engineered_feature_names_


def get_feature_documentation() -> Dict[str, Dict[str, str]]:
    """Return dictionary explaining business definitions of engineered features."""
    return {
        "demand_to_supply_ratio": {
            "Formula": "Monthly Demand / Monthly Supply",
            "Interpretation": "> 1.0 implies structural demand deficit; < 1.0 implies supply surplus."
        },
        "dsr_equilibrium_deviation": {
            "Formula": "DSR - 1.0",
            "Interpretation": "Distance from optimal supply-demand parity."
        },
        "lead_time_variability_index": {
            "Formula": "Lead Time Std Dev / Mean Lead Time",
            "Interpretation": "Relative supply chain volatility and supplier delivery instability."
        },
        "stockout_severity_frequency_score": {
            "Formula": "Stockout Incidents * (1 + (1 - OTIF) * 5)",
            "Interpretation": "Weighted historical impact of stockouts factoring in customer order fulfillment failure."
        },
        "warehouse_capacity_utilization": {
            "Formula": "Current Inventory / Warehouse Capacity",
            "Interpretation": "Physical storage congestion; > 0.85 risks bottleneck; < 0.35 indicates underutilization."
        },
        "safety_stock_coverage_ratio": {
            "Formula": "Current Inventory / Target Safety Stock",
            "Interpretation": "< 1.0 indicates critical buffer depletion; > 1.8 indicates over-buffered capital lockup."
        },
        "inventory_days_of_supply": {
            "Formula": "Current Inventory / Daily Demand Run-Rate",
            "Interpretation": "Expected days before total stockout without replenishment."
        },
        "demand_volatility_index": {
            "Formula": "Demand Std Dev / Mean Monthly Demand",
            "Interpretation": "Demand unpredictability and bullwhip risk."
        },
        "shortage_gap_units": {
            "Formula": "max(0, Safety Stock - Current Inventory)",
            "Interpretation": "Absolute unit injection required to restore baseline SLA service level."
        },
        "transit_risk_index": {
            "Formula": "Transit Delay Days * (1 + (1 - Supplier Reliability) * 3)",
            "Interpretation": "Logistics exposure reflecting transport delay amplified by supplier inconsistency."
        }
    }
