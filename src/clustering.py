"""
Warehouse Segmentation and K-Means Clustering Module.

Performs K-Means clustering on aggregated warehouse operational profiles,
evaluates K=2..8 via Elbow Method and Silhouette Scores, and profiles operational clusters.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class WarehouseClusterer:
    """
    K-Means clustering pipeline for warehouse segmentation.
    """

    def __init__(self, k_range: Tuple[int, int] = (2, 8), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        
        self.clustering_features: List[str] = [
            "demand_to_supply_ratio_avg",
            "monthly_demand_avg",
            "capacity_utilization_avg",
            "inventory_turnover_avg",
            "stockout_incidents_sum",
            "lead_time_variability_avg",
            "fulfillment_rate_avg",
            "overstock_quantity_avg",
            "safety_stock_avg",
            "supplier_reliability_avg"
        ]
        
        self.optimal_k: int = 3
        self.kmeans_model: Optional[KMeans] = None
        self.evaluation_metrics: Dict[str, List] = {"k": [], "inertia": [], "silhouette": []}
        self.cluster_profiles_: Optional[pd.DataFrame] = None
        self.cluster_labels_map_: Dict[int, str] = {}

    def evaluate_k_range(self, df_profiles: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate multiple K values using Inertia (Elbow Method) and Silhouette Scores.
        """
        # Ensure required features exist in profiles
        available_features = [f for f in self.clustering_features if f in df_profiles.columns]
        X = df_profiles[available_features].copy()
        
        # Fill any missing values with median
        X = X.fillna(X.median())
        X_scaled = self.scaler.fit_transform(X)
        
        results = []
        best_silhouette = -1.0
        best_k = 3
        
        for k in range(self.k_range[0], self.k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=15, max_iter=300)
            labels = km.fit_predict(X_scaled)
            inertia = float(km.inertia_)
            sil = float(silhouette_score(X_scaled, labels))
            
            results.append({
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4)
            })
            
            # Track best silhouette score
            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k
                
        self.optimal_k = best_k
        self.evaluation_metrics = {
            "k": [r["k"] for r in results],
            "inertia": [r["inertia"] for r in results],
            "silhouette": [r["silhouette_score"] for r in results]
        }
        return pd.DataFrame(results)

    def fit_predict(self, df_profiles: pd.DataFrame, k: Optional[int] = None) -> pd.DataFrame:
        """
        Fit K-Means using selected or optimal K and return profiles with cluster labels.
        """
        target_k = k or self.optimal_k
        available_features = [f for f in self.clustering_features if f in df_profiles.columns]
        X = df_profiles[available_features].copy().fillna(df_profiles[available_features].median())
        
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans_model = KMeans(n_clusters=target_k, random_state=self.random_state, n_init=20, max_iter=400)
        cluster_ids = self.kmeans_model.fit_predict(X_scaled)
        
        # PCA 2D coordinates for visual mapping
        pca_coords = self.pca.fit_transform(X_scaled)
        
        result_df = df_profiles.copy()
        result_df["cluster_id"] = cluster_ids
        result_df["pca_x"] = np.round(pca_coords[:, 0], 3)
        result_df["pca_y"] = np.round(pca_coords[:, 1], 3)
        
        # Build dynamic cluster profiles & operational naming
        self._generate_cluster_names(result_df, available_features)
        result_df["cluster_name"] = result_df["cluster_id"].map(self.cluster_labels_map_)
        
        return result_df

    def _generate_cluster_names(self, df_with_clusters: pd.DataFrame, feature_cols: List[str]):
        """
        Dynamically profile clusters from empirical centroids and assign descriptive operational names.
        """
        centroid_summary = df_with_clusters.groupby("cluster_id")[feature_cols].mean()
        self.cluster_profiles_ = centroid_summary
        
        labels_map = {}
        for c_id, row in centroid_summary.iterrows():
            dsr = row.get("demand_to_supply_ratio_avg", 1.0)
            stockouts = row.get("stockout_incidents_sum", 0)
            overstock = row.get("overstock_quantity_avg", 0)
            turnover = row.get("inventory_turnover_avg", 6.0)
            
            # Dynamic heuristic characterization
            if dsr > 1.05 and stockouts > centroid_summary["stockout_incidents_sum"].median():
                labels_map[c_id] = f"Cluster {c_id} — High-Velocity Shortage Risk"
            elif overstock > centroid_summary["overstock_quantity_avg"].median() and turnover < centroid_summary["inventory_turnover_avg"].median():
                labels_map[c_id] = f"Cluster {c_id} — Chronic Overstock Buffers"
            else:
                labels_map[c_id] = f"Cluster {c_id} — Balanced Operational Flow"
                
        self.cluster_labels_map_ = labels_map

    def get_cluster_summary(self) -> pd.DataFrame:
        """Return formatted summary table of cluster centroids and names."""
        if self.cluster_profiles_ is None:
            raise ValueError("Clusterer must be fit before retrieving summary.")
        df_summary = self.cluster_profiles_.copy()
        df_summary["Operational_Segment"] = [self.cluster_labels_map_.get(i, f"Cluster {i}") for i in df_summary.index]
        return df_summary
