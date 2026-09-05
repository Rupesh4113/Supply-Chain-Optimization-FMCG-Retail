"""
Data Generation Module for FMCG / Retail Supply Chain Optimization.

Simulates high-fidelity operational records for 60 warehouses across 5 regions
over 24 months (1,440 warehouse-month observations) adhering to FMCG inventory mechanics.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import os


# Predefined master list of 60 illustrative distribution facilities across India
WAREHOUSE_CATALOG: List[Dict[str, str]] = [
    # North Region (14 Facilities)
    {"warehouse_id": "WH-01", "name": "Hub North-Central — Delhi/NCR", "region": "North", "city": "Delhi/NCR", "type": "Mega Hub", "capacity": 220000},
    {"warehouse_id": "WH-02", "name": "Depot North-A — Noida Hub", "region": "North", "city": "Noida", "type": "Regional DC", "capacity": 95000},
    {"warehouse_id": "WH-03", "name": "Depot North-B — Gurgaon Metro", "region": "North", "city": "Gurugram", "type": "Urban Depot", "capacity": 45000},
    {"warehouse_id": "WH-04", "name": "Facility North-1 — Chandigarh Gateway", "region": "North", "city": "Chandigarh", "type": "Regional DC", "capacity": 85000},
    {"warehouse_id": "WH-05", "name": "Depot North-2 — Ludhiana Express", "region": "North", "city": "Ludhiana", "type": "Urban Depot", "capacity": 40000},
    {"warehouse_id": "WH-06", "name": "Hub North-3 — Jaipur Logistics Park", "region": "North", "city": "Jaipur", "type": "Regional DC", "capacity": 110000},
    {"warehouse_id": "WH-07", "name": "Depot North-4 — Jodhpur Transit", "region": "North", "city": "Jodhpur", "type": "Urban Depot", "capacity": 35000},
    {"warehouse_id": "WH-08", "name": "Hub North-5 — Lucknow Central", "region": "North", "city": "Lucknow", "type": "Regional DC", "capacity": 90000},
    {"warehouse_id": "WH-09", "name": "Depot North-6 — Kanpur Freight Center", "region": "North", "city": "Kanpur", "type": "Urban Depot", "capacity": 48000},
    {"warehouse_id": "WH-10", "name": "Depot North-7 — Varanasi Terminal", "region": "North", "city": "Varanasi", "type": "Urban Depot", "capacity": 38000},
    {"warehouse_id": "WH-11", "name": "Facility North-8 — Agra Express", "region": "North", "city": "Agra", "type": "Urban Depot", "capacity": 42000},
    {"warehouse_id": "WH-12", "name": "Depot North-9 — Dehradun Valley", "region": "North", "city": "Dehradun", "type": "Urban Depot", "capacity": 32000},
    {"warehouse_id": "WH-13", "name": "Depot North-10 — Amritsar Border Hub", "region": "North", "city": "Amritsar", "type": "Urban Depot", "capacity": 36000},
    {"warehouse_id": "WH-14", "name": "Facility North-11 — Ghaziabad Depot", "region": "North", "city": "Ghaziabad", "type": "Regional DC", "capacity": 78000},

    # West Region (14 Facilities)
    {"warehouse_id": "WH-15", "name": "Hub West-Central — Mumbai Port Mega", "region": "West", "city": "Mumbai", "type": "Mega Hub", "capacity": 250000},
    {"warehouse_id": "WH-16", "name": "Depot West-1 — Bhiwandi Logistics Hub", "region": "West", "city": "Bhiwandi", "type": "Mega Hub", "capacity": 210000},
    {"warehouse_id": "WH-17", "name": "Depot West-2 — Navi Mumbai Metro", "region": "West", "city": "Navi Mumbai", "type": "Urban Depot", "capacity": 55000},
    {"warehouse_id": "WH-18", "name": "Facility West-3 — Pune Auto Corridor", "region": "West", "city": "Pune", "type": "Regional DC", "capacity": 120000},
    {"warehouse_id": "WH-19", "name": "Depot West-4 — Chakan Industrial DC", "region": "West", "city": "Pune", "type": "Regional DC", "capacity": 85000},
    {"warehouse_id": "WH-20", "name": "Hub West-5 — Ahmedabad Sanand Park", "region": "West", "city": "Ahmedabad", "type": "Mega Hub", "capacity": 175000},
    {"warehouse_id": "WH-21", "name": "Depot West-6 — Surat Textile DC", "region": "West", "city": "Surat", "type": "Regional DC", "capacity": 92000},
    {"warehouse_id": "WH-22", "name": "Facility West-7 — Vadodara Depot", "region": "West", "city": "Vadodara", "type": "Regional DC", "capacity": 76000},
    {"warehouse_id": "WH-23", "name": "Depot West-8 — Rajkot Saurashtra", "region": "West", "city": "Rajkot", "type": "Urban Depot", "capacity": 42000},
    {"warehouse_id": "WH-24", "name": "Depot West-9 — Nashik Agri Hub", "region": "West", "city": "Nashik", "type": "Urban Depot", "capacity": 48000},
    {"warehouse_id": "WH-25", "name": "Facility West-10 — Aurangabad Center", "region": "West", "city": "Aurangabad", "type": "Urban Depot", "capacity": 45000},
    {"warehouse_id": "WH-26", "name": "Depot West-11 — Kolhapur Transit", "region": "West", "city": "Kolhapur", "type": "Urban Depot", "capacity": 38000},
    {"warehouse_id": "WH-27", "name": "Depot West-12 — Goa Coastal DC", "region": "West", "city": "Panaji", "type": "Urban Depot", "capacity": 34000},
    {"warehouse_id": "WH-28", "name": "Hub West-13 — Thane Express Depot", "region": "West", "city": "Thane", "type": "Regional DC", "capacity": 88000},

    # South Region (14 Facilities)
    {"warehouse_id": "WH-29", "name": "Hub South-1 — Bengaluru Electronic City", "region": "South", "city": "Bengaluru", "type": "Mega Hub", "capacity": 240000},
    {"warehouse_id": "WH-30", "name": "Depot South-2 — Hosur Road Corridor", "region": "South", "city": "Bengaluru", "type": "Regional DC", "capacity": 130000},
    {"warehouse_id": "WH-31", "name": "Facility South-3 — Whitefield Urban DC", "region": "South", "city": "Bengaluru", "type": "Urban Depot", "capacity": 52000},
    {"warehouse_id": "WH-32", "name": "Hub South-4 — Chennai Port Mega Hub", "region": "South", "city": "Chennai", "type": "Mega Hub", "capacity": 210000},
    {"warehouse_id": "WH-33", "name": "Depot South-5 — Sriperumbudur Terminal", "region": "South", "city": "Chennai", "type": "Regional DC", "capacity": 98000},
    {"warehouse_id": "WH-34", "name": "Depot South-6 — Coimbatore Industrial Hub", "region": "South", "city": "Coimbatore", "type": "Regional DC", "capacity": 75000},
    {"warehouse_id": "WH-35", "name": "Facility South-7 — Madurai South Gateway", "region": "South", "city": "Madurai", "type": "Urban Depot", "capacity": 44000},
    {"warehouse_id": "WH-36", "name": "Hub South-8 — Hyderabad Shamshabad DC", "region": "South", "city": "Hyderabad", "type": "Mega Hub", "capacity": 190000},
    {"warehouse_id": "WH-37", "name": "Depot South-9 — Secunderabad Express", "region": "South", "city": "Hyderabad", "type": "Regional DC", "capacity": 82000},
    {"warehouse_id": "WH-38", "name": "Facility South-10 — Vijayawada River Hub", "region": "South", "city": "Vijayawada", "type": "Regional DC", "capacity": 65000},
    {"warehouse_id": "WH-39", "name": "Depot South-11 — Visakhapatnam Port DC", "region": "South", "city": "Visakhapatnam", "type": "Regional DC", "capacity": 72000},
    {"warehouse_id": "WH-40", "name": "Depot South-12 — Kochi Maritime Terminal", "region": "South", "city": "Kochi", "type": "Regional DC", "capacity": 68000},
    {"warehouse_id": "WH-41", "name": "Facility South-13 — Kozhikode Depot", "region": "South", "city": "Kozhikode", "type": "Urban Depot", "capacity": 38000},
    {"warehouse_id": "WH-42", "name": "Depot South-14 — Mysuru Heritage DC", "region": "South", "city": "Mysuru", "type": "Urban Depot", "capacity": 40000},

    # East Region (10 Facilities)
    {"warehouse_id": "WH-43", "name": "Hub East-1 — Kolkata Dankuni Mega Hub", "region": "East", "city": "Kolkata", "type": "Mega Hub", "capacity": 205000},
    {"warehouse_id": "WH-44", "name": "Depot East-2 — Howrah Freight Center", "region": "East", "city": "Kolkata", "type": "Regional DC", "capacity": 88000},
    {"warehouse_id": "WH-45", "name": "Facility East-3 — Siliguri North Corridor", "region": "East", "city": "Siliguri", "type": "Regional DC", "capacity": 64000},
    {"warehouse_id": "WH-46", "name": "Depot East-4 — Patna Ganga Terminal", "region": "East", "city": "Patna", "type": "Regional DC", "capacity": 82000},
    {"warehouse_id": "WH-47", "name": "Depot East-5 — Ranchi Chotanagpur DC", "region": "East", "city": "Ranchi", "type": "Urban Depot", "capacity": 46000},
    {"warehouse_id": "WH-48", "name": "Facility East-6 — Jamshedpur Steel Hub", "region": "East", "city": "Jamshedpur", "type": "Urban Depot", "capacity": 45000},
    {"warehouse_id": "WH-49", "name": "Depot East-7 — Bhubaneswar Gateway", "region": "East", "city": "Bhubaneswar", "type": "Regional DC", "capacity": 74000},
    {"warehouse_id": "WH-50", "name": "Depot East-8 — Cuttack Logistics Depot", "region": "East", "city": "Cuttack", "type": "Urban Depot", "capacity": 38000},
    {"warehouse_id": "WH-51", "name": "Facility East-9 — Guwahati Northeast Hub", "region": "East", "city": "Guwahati", "type": "Regional DC", "capacity": 85000},
    {"warehouse_id": "WH-52", "name": "Depot East-10 — Durgapur Industrial Hub", "region": "East", "city": "Durgapur", "type": "Urban Depot", "capacity": 42000},

    # Central Region (8 Facilities)
    {"warehouse_id": "WH-53", "name": "Hub Central-1 — Nagpur MIHAN Multimodal", "region": "Central", "city": "Nagpur", "type": "Mega Hub", "capacity": 230000},
    {"warehouse_id": "WH-54", "name": "Depot Central-2 — Nagpur Wardha Road DC", "region": "Central", "city": "Nagpur", "type": "Regional DC", "capacity": 88000},
    {"warehouse_id": "WH-55", "name": "Hub Central-3 — Indore Pithampur Park", "region": "Central", "city": "Indore", "type": "Regional DC", "capacity": 105000},
    {"warehouse_id": "WH-56", "name": "Depot Central-4 — Bhopal Central DC", "region": "Central", "city": "Bhopal", "type": "Regional DC", "capacity": 72000},
    {"warehouse_id": "WH-57", "name": "Facility Central-5 — Jabalpur Rail Terminal", "region": "Central", "city": "Jabalpur", "type": "Urban Depot", "capacity": 48000},
    {"warehouse_id": "WH-58", "name": "Depot Central-6 — Raipur Logistics Center", "region": "Central", "city": "Raipur", "type": "Regional DC", "capacity": 68000},
    {"warehouse_id": "WH-59", "name": "Facility Central-7 — Bilaspur Transit Depot", "region": "Central", "city": "Bilaspur", "type": "Urban Depot", "capacity": 36000},
    {"warehouse_id": "WH-60", "name": "Depot Central-8 — Gwalior Express Gateway", "region": "Central", "city": "Gwalior", "type": "Urban Depot", "capacity": 40000},
]


def generate_supply_chain_data(
    n_warehouses: int = 60,
    n_months: int = 24,
    start_year: int = 2023,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic FMCG warehouse operational dataset.

    Args:
        n_warehouses: Number of warehouse facilities (default: 60)
        n_months: Number of continuous monthly periods (default: 24)
        start_year: Starting year for timeline (default: 2023)
        random_seed: Seed for full reproducibility (default: 42)

    Returns:
        pd.DataFrame containing 1,440 operational records across all required operational metrics.
    """
    np.random.seed(random_seed)
    warehouses = WAREHOUSE_CATALOG[:n_warehouses]
    
    # Generate timeline of months (e.g. 2023-01 through 2024-12)
    dates = pd.date_range(
        start=f"{start_year}-01-01", 
        periods=n_months, 
        freq="MS"
    ).strftime("%Y-%m").tolist()
    
    records = []

    # Indian retail seasonality curve:
    # Strong peaks in festive Q4 (Oct-Nov: Diwali, Puja), moderate mid-year summer peak (Apr-May)
    monthly_seasonality = {
        1: 0.94,  2: 0.90,  3: 1.02,  4: 1.12,  5: 1.15,  6: 1.00,
        7: 0.92,  8: 0.96,  9: 1.08, 10: 1.34, 11: 1.38, 12: 1.16
    }
    
    # Regional baseline growth momentum
    region_momentum_base = {
        "North": 0.08, "West": 0.12, "South": 0.14, "East": 0.05, "Central": 0.09
    }

    # Assign persistent structural traits per warehouse (e.g., vendor performance, operational efficiency)
    wh_profiles = {}
    for wh in warehouses:
        wh_type = wh["type"]
        cap = wh["capacity"]
        
        # Base demand scales with warehouse capacity
        if wh_type == "Mega Hub":
            base_demand = cap * np.random.uniform(0.38, 0.48)
            sku_cnt = int(np.random.uniform(650, 1150))
            base_lead = np.random.uniform(14, 26)
        elif wh_type == "Regional DC":
            base_demand = cap * np.random.uniform(0.35, 0.46)
            sku_cnt = int(np.random.uniform(300, 700))
            base_lead = np.random.uniform(8, 18)
        else: # Urban Depot
            base_demand = cap * np.random.uniform(0.32, 0.44)
            sku_cnt = int(np.random.uniform(100, 350))
            base_lead = np.random.uniform(4, 12)
            
        # Specific structural bias to simulate operational archetypes
        # 25% chronically high-velocity/strained, 25% sluggish/overstocked, 50% well-balanced
        wh_bias_rand = np.random.rand()
        if wh_bias_rand < 0.28:
            archetype = "strained_velocity"
            supplier_rel_mean = np.random.uniform(0.72, 0.84)
            lead_var_base = np.random.uniform(4.5, 9.0)
            target_safety_days = np.random.uniform(10, 18) # often under-buffered
        elif wh_bias_rand > 0.72:
            archetype = "sluggish_buffer"
            supplier_rel_mean = np.random.uniform(0.88, 0.97)
            lead_var_base = np.random.uniform(1.2, 3.2)
            target_safety_days = np.random.uniform(32, 50) # heavily over-buffered
        else:
            archetype = "balanced_flow"
            supplier_rel_mean = np.random.uniform(0.85, 0.94)
            lead_var_base = np.random.uniform(2.0, 4.5)
            target_safety_days = np.random.uniform(20, 30)

        wh_profiles[wh["warehouse_id"]] = {
            "base_demand": base_demand,
            "sku_cnt": sku_cnt,
            "base_lead": base_lead,
            "lead_var_base": lead_var_base,
            "supplier_rel_mean": supplier_rel_mean,
            "target_safety_days": target_safety_days,
            "archetype": archetype
        }

    # Simulate dynamic monthly transitions for each warehouse
    for wh in warehouses:
        wh_id = wh["warehouse_id"]
        prof = wh_profiles[wh_id]
        cap = wh["capacity"]
        reg = wh["region"]
        
        # Initial inventory position
        if prof["archetype"] == "sluggish_buffer":
            current_inv = cap * np.random.uniform(0.65, 0.85)
        elif prof["archetype"] == "strained_velocity":
            current_inv = cap * np.random.uniform(0.22, 0.38)
        else:
            current_inv = cap * np.random.uniform(0.40, 0.58)

        for m_idx, date_str in enumerate(dates):
            month_int = int(date_str.split("-")[1])
            year_int = int(date_str.split("-")[0])
            
            # Seasonality & Momentum
            season_idx = monthly_seasonality[month_int] + np.random.normal(0, 0.03)
            momentum = region_momentum_base[reg] + np.random.normal(0, 0.05) + (0.04 if year_int == 2024 else 0.0)
            
            # Monthly Demand
            # Base + seasonality + regional trend + random operational shock
            demand_noise = np.random.normal(1.0, 0.08)
            monthly_demand = prof["base_demand"] * season_idx * (1.0 + momentum * (m_idx / 24.0)) * demand_noise
            monthly_demand = max(1500.0, round(monthly_demand, 2))
            
            # Lead time and variability
            lead_time_days = round(prof["base_lead"] + np.random.normal(0, 1.2), 1)
            lead_time_days = max(2.5, min(45.0, lead_time_days))
            
            lead_time_var = round(prof["lead_var_base"] * np.random.uniform(0.85, 1.25), 2)
            lead_time_var = max(0.5, lead_time_var)
            
            transit_delay = round(max(0.0, np.random.exponential(scale=lead_time_var * 0.4)), 1)
            supplier_reliability = round(min(0.99, max(0.55, np.random.normal(prof["supplier_rel_mean"], 0.04))), 3)
            
            # Calculated Safety Stock based on lead time volatility & service factor
            daily_demand = monthly_demand / 30.0
            # Safety stock = Z (1.65 for 95% service) * sqrt(L * sigma_D^2 + D^2 * sigma_L^2)
            demand_std = daily_demand * np.random.uniform(0.12, 0.25)
            safety_stock = 1.65 * np.sqrt(lead_time_days * (demand_std**2) + (daily_demand**2) * (lead_time_var**2))
            safety_stock = round(max(300.0, safety_stock), 2)
            
            # Reorder point = (Lead Time * Daily Demand) + Safety Stock
            reorder_point = round((lead_time_days * daily_demand) + safety_stock, 2)
            
            # Monthly Supply Inbound
            # Under strained velocity, supplier under-delivers or delays
            if prof["archetype"] == "strained_velocity":
                supply_ratio = np.random.uniform(0.70, 0.94) * supplier_reliability
            elif prof["archetype"] == "sluggish_buffer":
                supply_ratio = np.random.uniform(1.05, 1.28)
            else:
                supply_ratio = np.random.uniform(0.92, 1.08)
                
            monthly_supply = round(monthly_demand * supply_ratio, 2)
            
            # Update inventory balance
            # New Inventory = Prior Inventory + Inbound Supply - Actual Fulfilled Demand
            potential_available = current_inv + monthly_supply
            
            if potential_available < monthly_demand:
                # Stockout occurred!
                fulfilled_demand = potential_available
                unfulfilled = monthly_demand - potential_available
                current_inv = max(0.0, np.random.uniform(0, 0.05 * safety_stock))
                stockout_incidents = int(np.random.randint(3, 11) + (unfulfilled / (daily_demand + 1e-5)) * 0.15)
                stockout_incidents = min(15, max(1, stockout_incidents))
                fulfillment_rate = round(fulfilled_demand / monthly_demand, 3)
            else:
                fulfilled_demand = monthly_demand
                current_inv = round(potential_available - fulfilled_demand, 2)
                stockout_incidents = 0 if current_inv > safety_stock else int(np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05]))
                fulfillment_rate = round(min(0.999, np.random.uniform(0.95, 0.998)), 3)
                
            # Capacity utilization and overstock
            capacity_utilization = round(current_inv / cap, 3)
            overstock_quantity = round(max(0.0, current_inv - (reorder_point + safety_stock * 1.5)), 2)
            
            # Inventory Turnover (Annualized: (Demand * 12) / Average Inventory)
            inv_turnover = round((monthly_demand * 12.0) / max(current_inv, 100.0), 2)
            inv_turnover = min(25.0, max(0.8, inv_turnover))
            
            # Operational metrics
            order_volume = int(round(monthly_demand / np.random.uniform(22.0, 38.0)))
            hist_demand_var = round(float(np.var(np.random.normal(monthly_demand, monthly_demand * 0.15, 6))), 2)
            
            # Target Classification Generation (Deterministic operational principles + realistic noise)
            dsr = monthly_demand / max(monthly_supply, 1.0)
            ss_coverage = current_inv / max(safety_stock, 1.0)
            
            # Score-based probabilistic assignment
            shortage_score = (
                1.4 * max(0.0, dsr - 1.05) +
                1.8 * max(0.0, 0.85 - ss_coverage) +
                0.8 * (lead_time_var / 5.0) +
                0.7 * (1.0 - supplier_reliability) +
                0.5 * (stockout_incidents / 5.0)
            )
            
            overstock_score = (
                1.3 * max(0.0, 0.95 - dsr) +
                1.6 * max(0.0, ss_coverage - 1.6) +
                1.0 * max(0.0, capacity_utilization - 0.75) +
                0.8 * max(0.0, 4.0 - inv_turnover) / 4.0
            )
            
            # Class assignment
            if shortage_score > 0.85 and shortage_score > overstock_score:
                risk_label = "Understock / Shortage"
            elif overstock_score > 0.80 and overstock_score > shortage_score:
                risk_label = "Overstock"
            else:
                risk_label = "Balanced"
                
            # Add ~6% realistic operational boundary noise (representing shrinkage or phantom inventory)
            if np.random.rand() < 0.06:
                risk_label = np.random.choice(
                    ["Understock / Shortage", "Balanced", "Overstock"],
                    p=[0.30, 0.40, 0.30]
                )

            record = {
                "record_id": f"REC-{date_str}-{wh_id}",
                "date": date_str,
                "warehouse_id": wh_id,
                "warehouse_name": wh["name"],
                "region": reg,
                "city": wh["city"],
                "warehouse_type": wh["type"],
                "monthly_demand": monthly_demand,
                "monthly_supply": monthly_supply,
                "current_inventory": round(current_inv, 2),
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "lead_time_days": lead_time_days,
                "lead_time_variability": lead_time_var,
                "order_volume": order_volume,
                "fulfillment_rate": fulfillment_rate,
                "stockout_incidents": stockout_incidents,
                "overstock_quantity": overstock_quantity,
                "warehouse_capacity": cap,
                "capacity_utilization": capacity_utilization,
                "inventory_turnover": inv_turnover,
                "historical_demand_variance": hist_demand_var,
                "seasonal_demand_index": round(season_idx, 3),
                "supplier_reliability": supplier_reliability,
                "transit_delay_days": transit_delay,
                "sku_count": prof["sku_cnt"],
                "regional_consumption_momentum": round(momentum, 3),
                "inventory_imbalance_risk": risk_label
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


def save_raw_dataset(
    output_path: str = "data/raw/fmcg_supply_chain_raw.csv",
    n_warehouses: int = 60,
    n_months: int = 24,
    random_seed: int = 42
) -> pd.DataFrame:
    """Generate and serialize the raw operational dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_supply_chain_data(
        n_warehouses=n_warehouses,
        n_months=n_months,
        random_seed=random_seed
    )
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df_raw = save_raw_dataset()
    print(f"Dataset generated successfully with shape: {df_raw.shape}")
    print("Class distribution:")
    print(df_raw["inventory_imbalance_risk"].value_counts(normalize=True).round(3))
