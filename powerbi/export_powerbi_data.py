"""
Power BI Data Exporter & Star Schema Transformer for Churn_Modelling Dataset
Transforms the audited banking customer churn dataset into optimized Star Schema Fact & Dimension tables.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import BankDataLoader
from src.feature_engineering import BankFeatureEngineer
from src.modeling import BankChurnModeler
from src.business_insights import BankBusinessInsights

def export_star_schema():
    print("=" * 70)
    print("[Power BI Exporter] Building Star Schema Tables for Churn_Modelling Dataset...")
    print("=" * 70)

    output_dir = os.path.join(BASE_DIR, "powerbi", "data")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data & run features
    raw_path = os.path.join(BASE_DIR, "data", "raw", "bank_customer_churn_raw.csv")
    loader = BankDataLoader(raw_path)
    df_raw = loader.load_or_create()

    fe = BankFeatureEngineer()
    df_feat = fe.create_features(df_raw)

    # 2. Train model & generate churn scores
    X_train, X_test, y_train, y_test, _ = fe.prepare_train_test_data(df_raw)
    modeler = BankChurnModeler()
    modeler.train_and_cross_validate(X_train, y_train)
    champion = modeler.tune_champion_hyperparameters(X_train, y_train)

    drop_cols = ["RowNumber", "CustomerId", "Surname", "Exited"]
    all_X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])
    churn_probs = champion.predict_proba(all_X)[:, 1]

    insights = BankBusinessInsights()
    df_scored = insights.assign_risk_tiers(df_feat, churn_probs)

    # -------------------------------------------------------------
    # Dimension 1: Dim_Geography
    # -------------------------------------------------------------
    dim_geo = pd.DataFrame([
        {"GeographyKey": 1, "Geography": "France", "CountryCode": "FRA", "Region": "Western Europe", "Latitude": 46.2276, "Longitude": 2.2137},
        {"GeographyKey": 2, "Geography": "Germany", "CountryCode": "DEU", "Region": "Central Europe", "Latitude": 51.1657, "Longitude": 10.4515},
        {"GeographyKey": 3, "Geography": "Spain", "CountryCode": "ESP", "Region": "Southern Europe", "Latitude": 40.4637, "Longitude": -3.7492}
    ])
    geo_map = {"France": 1, "Germany": 2, "Spain": 3}
    dim_geo.to_csv(os.path.join(output_dir, "Dim_Geography.csv"), index=False)
    print(f"[OK] Generated: Dim_Geography.csv ({len(dim_geo)} records)")

    # -------------------------------------------------------------
    # Dimension 2: Dim_Demographics
    # -------------------------------------------------------------
    dim_demo_records = []
    demo_key = 1
    demo_lookup = {}

    genders = ["Male", "Female"]
    age_groups = ["Under30", "Age30_45", "Age46_60", "Over60"]
    credit_tiers = ["Poor", "Fair", "Good", "VeryGood", "Exceptional"]

    for g in genders:
        for ag in age_groups:
            for ct in credit_tiers:
                key_combo = (g, ag, ct)
                gen_desc = "Gen Z / Young Adult" if ag == "Under30" else ("Millennial / Prime" if ag == "Age30_45" else ("Gen X / Peak Wealth" if ag == "Age46_60" else "Boomer / Senior"))
                dim_demo_records.append({
                    "DemographicKey": demo_key,
                    "Gender": g,
                    "AgeGroup": ag,
                    "GenerationalCohort": gen_desc,
                    "CreditScoreTier": ct,
                    "AgeGroupSort": 1 if ag == "Under30" else (2 if ag == "Age30_45" else (3 if ag == "Age46_60" else 4))
                })
                demo_lookup[key_combo] = demo_key
                demo_key += 1

    dim_demographics = pd.DataFrame(dim_demo_records)
    dim_demographics.to_csv(os.path.join(output_dir, "Dim_Demographics.csv"), index=False)
    print(f"[OK] Generated: Dim_Demographics.csv ({len(dim_demographics)} records)")

    # -------------------------------------------------------------
    # Dimension 3: Dim_Products
    # -------------------------------------------------------------
    dim_prod_records = []
    prod_key = 1
    prod_lookup = {}
    product_counts = [1, 2, 3, 4]
    has_cards = [0, 1]

    for p in product_counts:
        for hc in has_cards:
            key_combo = (p, hc)
            dim_prod_records.append({
                "ProductKey": prod_key,
                "NumOfProducts": p,
                "HasCreditCard": "Yes" if hc == 1 else "No",
                "HasCreditCardCode": hc,
                "ProductBundleTier": "Single Product" if p == 1 else ("Standard Bundle (2)" if p == 2 else "Multi-Product (3+)")
            })
            prod_lookup[key_combo] = prod_key
            prod_key += 1

    dim_products = pd.DataFrame(dim_prod_records)
    dim_products.to_csv(os.path.join(output_dir, "Dim_Products.csv"), index=False)
    print(f"[OK] Generated: Dim_Products.csv ({len(dim_products)} records)")

    # -------------------------------------------------------------
    # Dimension 4: Dim_Risk_Tiers
    # -------------------------------------------------------------
    dim_risk = pd.DataFrame([
        {
            "RiskTierKey": 1,
            "RiskTier": "High Risk",
            "ProbabilityThreshold": ">= 65%",
            "SortOrder": 1,
            "ColorHex": "#EF4444",
            "RecommendedAction": "Immediate VIP Relationship Manager Outreach & Wealth Advisory Consultation"
        },
        {
            "RiskTierKey": 2,
            "RiskTier": "Medium Risk",
            "ProbabilityThreshold": "35% - 65%",
            "SortOrder": 2,
            "ColorHex": "#F59E0B",
            "RecommendedAction": "Targeted Digital Product Consolidation & Fee Rebate Incentives"
        },
        {
            "RiskTierKey": 3,
            "RiskTier": "Low Risk",
            "ProbabilityThreshold": "< 35%",
            "SortOrder": 3,
            "ColorHex": "#10B981",
            "RecommendedAction": "Standard Relationship Care & Cross-sell"
        }
    ])
    dim_risk.to_csv(os.path.join(output_dir, "Dim_Risk_Tiers.csv"), index=False)
    print(f"[OK] Generated: Dim_Risk_Tiers.csv ({len(dim_risk)} records)")

    # -------------------------------------------------------------
    # Fact Table: Fact_Customer_Churn
    # -------------------------------------------------------------
    fact_df = df_scored.copy()
    
    fact_df["GeographyKey"] = fact_df["Geography"].map(geo_map)
    fact_df["DemographicKey"] = fact_df.apply(
        lambda r: demo_lookup.get((r["Gender"], r["AgeGroup"], r["CreditScoreTier"]), 1), axis=1
    )
    fact_df["ProductKey"] = fact_df.apply(
        lambda r: prod_lookup.get((r["NumOfProducts"], r["HasCrCard"]), 1), axis=1
    )
    fact_df["RiskTierKey"] = fact_df["RiskTier"].map({"High Risk": 1, "Medium Risk": 2, "Low Risk": 3})

    fact_cols = [
        "CustomerId",
        "GeographyKey",
        "DemographicKey",
        "ProductKey",
        "RiskTierKey",
        "Age",
        "CreditScore",
        "Tenure",
        "Balance",
        "EstimatedSalary",
        "IsActiveMember",
        "HasCrCard",
        "Exited",
        "ChurnProbability",
        "EstimatedAnnualValue",
        "ValueAtRisk",
        "BalanceSalaryRatio",
        "TenureAgeRatio",
        "EngagementScore",
        "BalancePerProduct"
    ]
    fact_table = fact_df[fact_cols]
    fact_table.to_csv(os.path.join(output_dir, "Fact_Customer_Churn.csv"), index=False)
    print(f"[OK] Generated: Fact_Customer_Churn.csv ({len(fact_table)} records)")

    print("=" * 70)
    print(f"[SUCCESS] All Star Schema tables exported to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    export_star_schema()
