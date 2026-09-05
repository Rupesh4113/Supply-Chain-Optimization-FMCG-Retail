"""
Replenishment Recommendation Engine for Supply Chain Optimization.

Generates warehouse-level replenishment actions, dynamic order quantities,
priority levels, and plain-English explainable rationales.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ReplenishmentRecommender:
    """
    Translates machine learning risk predictions and inventory mechanics
    into operational replenishment orders and tactical adjustments.
    """

    def __init__(
        self,
        shortage_prob_threshold: float = 0.50,
        overstock_prob_threshold: float = 0.50,
        service_factor_z: float = 1.65
    ):
        self.shortage_prob_threshold = shortage_prob_threshold
        self.overstock_prob_threshold = overstock_prob_threshold
        self.service_factor_z = service_factor_z

    def generate_recommendations(
        self,
        df_operational: pd.DataFrame,
        pipeline: Pipeline,
        cluster_column: str = "cluster_name"
    ) -> pd.DataFrame:
        """
        Generate operational recommendations for each warehouse record.
        """
        df = df_operational.copy()
        
        # Predict risk and class probabilities
        preds = pipeline.predict(df)
        probas = pipeline.predict_proba(df)
        classes = list(pipeline.classes_)
        
        shortage_idx = classes.index("Understock / Shortage") if "Understock / Shortage" in classes else 0
        overstock_idx = classes.index("Overstock") if "Overstock" in classes else 1
        balanced_idx = classes.index("Balanced") if "Balanced" in classes else 2
        
        df["predicted_risk"] = preds
        df["shortage_probability"] = np.round(probas[:, shortage_idx], 4)
        df["overstock_probability"] = np.round(probas[:, overstock_idx], 4)
        df["balanced_probability"] = np.round(probas[:, balanced_idx], 4)
        
        # Determine highest probability score
        df["max_risk_probability"] = np.round(np.max(probas, axis=1), 4)
        
        recs = []
        for idx, row in df.iterrows():
            pred_risk = row["predicted_risk"]
            p_shortage = row["shortage_probability"]
            p_overstock = row["overstock_probability"]
            
            cur_inv = row["current_inventory"]
            safety_stock = row["safety_stock"]
            reorder_point = row["reorder_point"]
            demand = row["monthly_demand"]
            supply = row["monthly_supply"]
            dsr = row["demand_to_supply_ratio"]
            lt_var = row["lead_time_variability"]
            otif = row["fulfillment_rate"]
            stockouts = row["stockout_incidents"]
            wh_name = row.get("warehouse_name", row.get("warehouse_id", f"WH-{idx}"))
            cluster = row.get(cluster_column, "Unassigned")
            
            # Inventory Gap: Target Buffer - Current Inventory
            # Target inventory is reorder point + dynamic safety buffer
            target_stock = reorder_point + (safety_stock * 0.5)
            inventory_gap = round(target_stock - cur_inv, 2)
            
            # Recommendation Logic
            if pred_risk == "Understock / Shortage" or p_shortage >= self.shortage_prob_threshold:
                # Calculate required injection
                needed_units = max(inventory_gap, demand * 0.4)
                rec_qty = int(np.ceil(needed_units / 50.0) * 50) # round to batch size 50
                
                if p_shortage >= 0.75 or cur_inv < (safety_stock * 0.5) or stockouts >= 3:
                    priority = "Critical"
                    action = "Urgent Safety Stock Injection & Expedited Replenishment"
                    reason = (
                        f"Critical shortage risk (p={p_shortage:.1%}): DSR={dsr:.2f}, "
                        f"inventory ({cur_inv:,.0f}) has breached safety stock ({safety_stock:,.0f}) "
                        f"with {stockouts} recent stockout incidents. Expedite inbound supplier delivery."
                    )
                else:
                    priority = "High"
                    action = "Accelerate Inbound Replenishment"
                    reason = (
                        f"Elevated shortage probability (p={p_shortage:.1%}): DSR={dsr:.2f} > 1.0. "
                        f"Replenish {rec_qty:,.0f} units to prevent stockout breach under lead-time variance of {lt_var:.1f} days."
                    )
                    
            elif pred_risk == "Overstock" or p_overstock >= self.overstock_prob_threshold:
                excess_units = max(0.0, cur_inv - target_stock)
                # Recommended reduction / freeze
                rec_qty = -int(np.ceil(excess_units / 50.0) * 50)
                
                if p_overstock >= 0.75 or row.get("capacity_utilization", 0.7) > 0.85:
                    priority = "High"
                    action = "Throttle Inbound Replenishment & Initiate Lateral Transfer Out"
                    reason = (
                        f"High overstock risk (p={p_overstock:.1%}): DSR={dsr:.2f} < 1.0, "
                        f"excess stock of {abs(rec_qty):,.0f} units locks working capital. "
                        f"Initiate lateral transfers to deficit regional depots."
                    )
                else:
                    priority = "Medium"
                    action = "Reduce Next Purchase Order Allocation"
                    reason = (
                        f"Moderate overstock trend (p={p_overstock:.1%}): Supply pace exceeds demand run-rate. "
                        f"Scale back upcoming monthly replenishment cycle."
                    )
                    
            else: # Balanced
                rec_qty = int(np.ceil(max(0.0, demand - cur_inv + safety_stock) / 100.0) * 100)
                priority = "Low"
                action = "Maintain Scheduled Standard Replenishment"
                reason = (
                    f"Balanced inventory posture (p={row['balanced_probability']:.1%}): DSR={dsr:.2f} is healthy, "
                    f"fill rate is {otif:.1%}. Execute regular cyclical replenishment."
                )

            recs.append({
                "warehouse_id": row["warehouse_id"],
                "warehouse_name": wh_name,
                "region": row["region"],
                "cluster": cluster,
                "current_inventory": int(cur_inv),
                "monthly_demand": int(demand),
                "monthly_supply": int(supply),
                "demand_to_supply_ratio": round(dsr, 2),
                "predicted_risk": pred_risk,
                "risk_probability": round(float(row["max_risk_probability"]), 3),
                "safety_stock": int(safety_stock),
                "inventory_gap": int(inventory_gap),
                "recommended_action": action,
                "recommended_quantity": rec_qty,
                "priority": priority,
                "explainable_reason": reason
            })

        return pd.DataFrame(recs)
