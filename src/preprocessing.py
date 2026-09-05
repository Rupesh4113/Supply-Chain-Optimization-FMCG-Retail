"""
Data Preprocessing and Validation Pipeline for FMCG / Retail Supply Chain Analytics.

Provides the DataPreprocessor class for robust cleaning, validation,
missing-value imputation, outlier bounding, and feature transformations.
"""

from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade preprocessor for supply chain operational tabular data.
    
    Ensures zero data leakage by computing statistics strictly on fit(),
    and applying them consistently on transform().
    """

    def __init__(
        self,
        impute_strategy: str = "median",
        handle_outliers: bool = True,
        outlier_iqr_multiplier: float = 2.5
    ):
        self.impute_strategy = impute_strategy
        self.handle_outliers = handle_outliers
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        
        # State learned during fit
        self.numeric_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, str] = {}
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.is_fitted_: bool = False

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input dataframe schema and data types."""
        issues = []
        expected_cols = [
            "record_id", "date", "warehouse_id", "warehouse_name", "region",
            "city", "warehouse_type", "monthly_demand", "monthly_supply",
            "current_inventory", "safety_stock", "reorder_point",
            "lead_time_days", "lead_time_variability", "order_volume",
            "fulfillment_rate", "stockout_incidents", "overstock_quantity",
            "warehouse_capacity", "capacity_utilization", "inventory_turnover",
            "historical_demand_variance", "seasonal_demand_index",
            "supplier_reliability", "transit_delay_days", "sku_count",
            "regional_consumption_momentum"
        ]
        
        for col in expected_cols:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
                
        # Check non-negative constraints
        non_negative_cols = [
            "monthly_demand", "monthly_supply", "current_inventory",
            "safety_stock", "lead_time_days", "warehouse_capacity"
        ]
        for col in non_negative_cols:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values detected in strictly positive column: {col}")
                
        return len(issues) == 0, issues

    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Report count and percentage of missing values per column."""
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100.0
        missing_report = pd.DataFrame({
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_pct.round(2)
        })
        return missing_report[missing_report["Missing_Count"] > 0]

    def detect_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
        """Count duplicate observations based on unique composite key."""
        key = subset or ["record_id"]
        if all(col in df.columns for col in key):
            return int(df.duplicated(subset=key).sum())
        return int(df.duplicated().sum())

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learn imputation values, outlier bounds, and column types."""
        df = X.copy()
        
        # Identify column types
        exclude_cols = ["record_id", "date", "warehouse_id", "warehouse_name", "inventory_imbalance_risk"]
        self.numeric_cols_ = [
            c for c in df.select_dtypes(include=[np.number]).columns 
            if c not in exclude_cols
        ]
        self.categorical_cols_ = [
            c for c in df.select_dtypes(include=["object", "category"]).columns 
            if c not in exclude_cols
        ]
        
        # Calculate medians / modes
        for col in self.numeric_cols_:
            self.numeric_medians_[col] = float(df[col].median())
            
            # Compute IQR outlier bounds
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = max(0.0, q25 - self.outlier_iqr_multiplier * iqr) if (df[col] >= 0).all() else q25 - self.outlier_iqr_multiplier * iqr
            upper_bound = q75 + self.outlier_iqr_multiplier * iqr
            self.outlier_bounds_[col] = (float(lower_bound), float(upper_bound))

        for col in self.categorical_cols_:
            mode_val = df[col].mode()
            self.categorical_modes_[col] = str(mode_val[0]) if not mode_val.empty else "Unknown"
            
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing: imputation, outlier bounds, date extraction."""
        if not self.is_fitted_:
            raise ValueError("DataPreprocessor must be fitted before transforming.")
            
        df = X.copy()
        
        # Deduplication
        if "record_id" in df.columns:
            df = df.drop_duplicates(subset=["record_id"]).reset_index(drop=True)
            
        # Missing value imputation
        for col in self.numeric_cols_:
            if col in df.columns:
                df[col] = df[col].fillna(self.numeric_medians_[col])
                
        for col in self.categorical_cols_:
            if col in df.columns:
                df[col] = df[col].fillna(self.categorical_modes_[col])
                
        # Outlier bounding (capping)
        if self.handle_outliers:
            for col in self.numeric_cols_:
                if col in df.columns and col in self.outlier_bounds_:
                    low, high = self.outlier_bounds_[col]
                    df[col] = df[col].clip(lower=low, upper=high)
                    
        # Date transformations
        if "date" in df.columns:
            # Parse YYYY-MM
            date_series = pd.to_datetime(df["date"], format="%Y-%m")
            df["year"] = date_series.dt.year
            df["month"] = date_series.dt.month
            df["quarter"] = date_series.dt.quarter
            # Q4 Festive Season flag (Oct, Nov, Dec in India)
            df["is_festive_quarter"] = df["month"].isin([10, 11, 12]).astype(int)
            
        return df


def aggregate_warehouse_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 24-month operational observations into static warehouse-level profiles
    suitable for facility segmentation and long-term clustering.
    """
    agg_rules = {
        "warehouse_name": "first",
        "region": "first",
        "city": "first",
        "warehouse_type": "first",
        "warehouse_capacity": "first",
        "monthly_demand": ["mean", "std"],
        "monthly_supply": "mean",
        "current_inventory": ["mean", "std"],
        "safety_stock": "mean",
        "lead_time_days": "mean",
        "lead_time_variability": "mean",
        "fulfillment_rate": "mean",
        "stockout_incidents": ["mean", "sum"],
        "overstock_quantity": "mean",
        "capacity_utilization": "mean",
        "inventory_turnover": "mean",
        "supplier_reliability": "mean",
        "transit_delay_days": "mean",
        "sku_count": "first",
        "regional_consumption_momentum": "mean"
    }
    
    grouped = df.groupby("warehouse_id").agg(agg_rules)
    # Flatten multi-index columns
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            if col[1] in ["first", "mean"] and col[1] == "first":
                flat_cols.append(col[0])
            elif col[1] == "mean":
                flat_cols.append(f"{col[0]}_avg")
            else:
                flat_cols.append(f"{col[0]}_{col[1]}")
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols
    
    # Calculate aggregated DSR at warehouse level
    grouped["demand_to_supply_ratio_avg"] = grouped["monthly_demand_avg"] / np.maximum(grouped["monthly_supply_avg"], 1.0)
    grouped = grouped.reset_index()
    return grouped
