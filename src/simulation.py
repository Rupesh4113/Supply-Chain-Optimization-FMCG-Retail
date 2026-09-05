"""
Supply Reallocation and Inventory Optimization Simulation Engine.

Simulates lateral inventory redistribution from surplus warehouses to deficit facilities,
quantifying unit balance improvements, holding cost reductions, and stockout mitigation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class SupplyReallocationSimulator:
    """
    Simulates lateral inventory redistribution between regional distribution centers.
    
    Identifies surplus distribution nodes and allocates excess inventory to
    at-risk shortage nodes, prioritizing intra-regional fulfillment to minimize freight penalties.
    """

    def __init__(
        self,
        holding_cost_per_unit_month: float = 2.50, # In currency units (e.g. $ or equivalent)
        stockout_penalty_per_unit: float = 8.00,
        intra_region_transfer_cost_per_unit: float = 0.45,
        inter_region_transfer_cost_per_unit: float = 1.10
    ):
        self.holding_cost_per_unit = holding_cost_per_unit_month
        self.stockout_penalty_per_unit = stockout_penalty_per_unit
        self.intra_cost = intra_region_transfer_cost_per_unit
        self.inter_cost = inter_region_transfer_cost_per_unit

    def run_reallocation_simulation(
        self,
        df_operational: pd.DataFrame,
        predicted_risk_col: str = "predicted_risk"
    ) -> Dict[str, Any]:
        """
        Execute lateral reallocation optimization simulation on the latest operational snapshot.
        """
        df = df_operational.copy()
        
        # Determine available surplus and deficit per facility
        # A facility is eligible to donate surplus if it has excess inventory beyond reorder point + buffer
        df["excess_available"] = np.maximum(
            0.0, df["current_inventory"] - (df["reorder_point"] + df["safety_stock"] * 0.5)
        )
        # Deficit is units required to reach safe buffer
        df["deficit_required"] = np.maximum(
            0.0, df["safety_stock"] - df["current_inventory"]
        )
        
        # Resolve risk series safely with fallbacks
        if predicted_risk_col in df.columns:
            risk_series = df[predicted_risk_col]
        elif "inventory_imbalance_risk" in df.columns:
            risk_series = df["inventory_imbalance_risk"]
        else:
            risk_series = pd.Series(["Balanced"] * len(df), index=df.index)

        # Mark surplus and deficit nodes
        surplus_nodes = df[
            (risk_series == "Overstock") | (df["excess_available"] > 500)
        ].copy()
        
        deficit_nodes = df[
            (risk_series == "Understock / Shortage") | (df["deficit_required"] > 500)
        ].copy()
        
        # Pre-simulation baseline totals
        total_shortage_before = float(df["deficit_required"].sum())
        total_overstock_before = float(df["excess_available"].sum())
        
        # Working tracker for dynamic updates
        inventory_state = df.set_index("warehouse_id")["current_inventory"].copy()
        surplus_remaining = surplus_nodes.set_index("warehouse_id")["excess_available"].copy()
        deficit_needed = deficit_nodes.set_index("warehouse_id")["deficit_required"].copy()
        
        transfer_logs: List[Dict[str, Any]] = []
        
        # Pass 1: Intra-Regional Transfers (Same region pairing)
        for d_id, def_row in deficit_nodes.iterrows():
            d_wh = def_row["warehouse_id"]
            d_reg = def_row["region"]
            d_req = deficit_needed.get(d_wh, 0.0)
            
            if d_req <= 0:
                continue
                
            # Find candidate donors in the same region
            regional_donors = surplus_nodes[
                (surplus_nodes["region"] == d_reg) & 
                (surplus_nodes["warehouse_id"] != d_wh)
            ]
            
            for _, s_row in regional_donors.iterrows():
                s_wh = s_row["warehouse_id"]
                s_avail = surplus_remaining.get(s_wh, 0.0)
                
                if s_avail <= 0:
                    continue
                    
                transfer_qty = min(d_req, s_avail)
                if transfer_qty >= 100: # Minimum economic order quantity
                    transfer_qty = int(round(transfer_qty))
                    
                    # Update trackers
                    deficit_needed[d_wh] -= transfer_qty
                    surplus_remaining[s_wh] -= transfer_qty
                    inventory_state[d_wh] += transfer_qty
                    inventory_state[s_wh] -= transfer_qty
                    d_req -= transfer_qty
                    
                    transfer_logs.append({
                        "from_warehouse_id": s_wh,
                        "from_warehouse_name": s_row.get("warehouse_name", s_wh),
                        "to_warehouse_id": d_wh,
                        "to_warehouse_name": def_row.get("warehouse_name", d_wh),
                        "transfer_type": "Intra-Regional",
                        "origin_region": s_row["region"],
                        "destination_region": d_reg,
                        "transferred_units": transfer_qty,
                        "estimated_freight_cost": round(transfer_qty * self.intra_cost, 2)
                    })
                    
                if d_req <= 0:
                    break

        # Pass 2: Inter-Regional Transfers (Cross-region for critical remaining deficits)
        for d_id, def_row in deficit_nodes.iterrows():
            d_wh = def_row["warehouse_id"]
            d_req = deficit_needed.get(d_wh, 0.0)
            
            if d_req <= 0:
                continue
                
            # Filter remaining donors with healthy surplus
            cross_donors = [s_wh for s_wh, s_avail in surplus_remaining.items() if s_avail >= 500]
            for s_wh in cross_donors:
                s_avail = surplus_remaining[s_wh]
                s_row = surplus_nodes[surplus_nodes["warehouse_id"] == s_wh].iloc[0]
                
                transfer_qty = min(d_req, s_avail)
                if transfer_qty >= 200:
                    transfer_qty = int(round(transfer_qty))
                    
                    deficit_needed[d_wh] -= transfer_qty
                    surplus_remaining[s_wh] -= transfer_qty
                    inventory_state[d_wh] += transfer_qty
                    inventory_state[s_wh] -= transfer_qty
                    d_req -= transfer_qty
                    
                    transfer_logs.append({
                        "from_warehouse_id": s_wh,
                        "from_warehouse_name": s_row.get("warehouse_name", s_wh),
                        "to_warehouse_id": d_wh,
                        "to_warehouse_name": def_row.get("warehouse_name", d_wh),
                        "transfer_type": "Inter-Regional",
                        "origin_region": s_row["region"],
                        "destination_region": def_row["region"],
                        "transferred_units": transfer_qty,
                        "estimated_freight_cost": round(transfer_qty * self.inter_cost, 2)
                    })
                    
                if d_req <= 0:
                    break

        df_transfers = pd.DataFrame(transfer_logs)
        total_units_reallocated = int(df_transfers["transferred_units"].sum()) if not df_transfers.empty else 0
        total_freight_cost = float(df_transfers["estimated_freight_cost"].sum()) if not df_transfers.empty else 0.0
        
        # Post-simulation balances
        total_shortage_after = float(max(0.0, total_shortage_before - total_units_reallocated))
        total_overstock_after = float(max(0.0, total_overstock_before - total_units_reallocated))
        
        shortage_reduction_pct = (
            ((total_shortage_before - total_shortage_after) / max(total_shortage_before, 1.0)) * 100.0
        )
        overstock_reduction_pct = (
            ((total_overstock_before - total_overstock_after) / max(total_overstock_before, 1.0)) * 100.0
        )
        
        # Financial Impact Evaluation
        holding_cost_savings = total_units_reallocated * self.holding_cost_per_unit
        stockout_penalty_mitigated = total_units_reallocated * self.stockout_penalty_per_unit
        net_financial_benefit = (holding_cost_savings + stockout_penalty_mitigated) - total_freight_cost
        
        affected_warehouses = set()
        if not df_transfers.empty:
            affected_warehouses.update(df_transfers["from_warehouse_id"].unique())
            affected_warehouses.update(df_transfers["to_warehouse_id"].unique())

        return {
            "total_shortage_before": round(total_shortage_before, 0),
            "total_shortage_after": round(total_shortage_after, 0),
            "shortage_reduction_units": int(total_shortage_before - total_shortage_after),
            "shortage_reduction_pct": round(shortage_reduction_pct, 1),
            "total_overstock_before": round(total_overstock_before, 0),
            "total_overstock_after": round(total_overstock_after, 0),
            "overstock_reduction_pct": round(overstock_reduction_pct, 1),
            "total_units_reallocated": total_units_reallocated,
            "total_freight_cost": round(total_freight_cost, 2),
            "holding_cost_savings": round(holding_cost_savings, 2),
            "stockout_penalty_mitigated": round(stockout_penalty_mitigated, 2),
            "net_financial_benefit": round(net_financial_benefit, 2),
            "affected_warehouses_count": len(affected_warehouses),
            "transfers_count": len(df_transfers),
            "transfer_log": df_transfers,
            "simulated_assumptions": {
                "holding_cost_per_unit_month": self.holding_cost_per_unit,
                "stockout_penalty_per_unit": self.stockout_penalty_per_unit,
                "intra_region_freight_cost": self.intra_cost,
                "inter_region_freight_cost": self.inter_cost
            }
        }
