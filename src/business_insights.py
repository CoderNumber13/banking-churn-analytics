"""
Business Impact Analysis, Customer Risk Tiering & Retention Campaign ROI Simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class BankBusinessInsights:
    """
    Translates machine learning churn probabilities into financial risk quantification,
    actionable customer segmentation, and strategic campaign ROI modeling.
    """

    def __init__(self, high_risk_cutoff: float = 0.65, medium_risk_cutoff: float = 0.35):
        self.high_risk_cutoff = high_risk_cutoff
        self.medium_risk_cutoff = medium_risk_cutoff

    def assign_risk_tiers(self, df_with_features: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
        """
        Appends churn probabilities, risk tiers, and calculated annual customer value.
        """
        df_scored = df_with_features.copy()
        df_scored["ChurnProbability"] = np.round(churn_probs, 4)

        conditions = [
            df_scored["ChurnProbability"] >= self.high_risk_cutoff,
            (df_scored["ChurnProbability"] >= self.medium_risk_cutoff) & (df_scored["ChurnProbability"] < self.high_risk_cutoff),
            df_scored["ChurnProbability"] < self.medium_risk_cutoff
        ]
        tier_labels = ["High Risk", "Medium Risk", "Low Risk"]
        df_scored["RiskTier"] = np.select(conditions, tier_labels, default="Low Risk")

        # Estimated Annual Banking Value: 3% Net Interest Margin on Balance + $150 Annual Fee Revenue
        has_card = df_scored["HasCrCard"] if "HasCrCard" in df_scored.columns else 0
        df_scored["EstimatedAnnualValue"] = np.round(
            (df_scored["Balance"] * 0.03) + (df_scored["NumOfProducts"] * 75.0) + (has_card * 50.0),
            2
        )
        
        # Expected Annual Value at Risk = ChurnProbability * EstimatedAnnualValue
        df_scored["ValueAtRisk"] = np.round(df_scored["ChurnProbability"] * df_scored["EstimatedAnnualValue"], 2)

        return df_scored

    def calculate_tier_financial_summary(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates customer counts, deposit balances at risk, and average demographics by risk tier.
        """
        summary = df_scored.groupby("RiskTier").agg(
            Total_Customers=("CustomerId", "count") if "CustomerId" in df_scored.columns else ("ChurnProbability", "count"),
            Avg_Churn_Probability=("ChurnProbability", "mean"),
            Total_Balance_At_Risk=("Balance", "sum"),
            Avg_Balance=("Balance", "mean"),
            Total_Annual_Value_At_Risk=("ValueAtRisk", "sum"),
            Avg_Age=("Age", "mean"),
            Inactive_Rate=("IsActiveMember", lambda x: (x == 0).mean())
        ).reset_index()

        summary["Pct_Total_Customers"] = np.round((summary["Total_Customers"] / len(df_scored)) * 100, 1)
        summary["Avg_Churn_Probability"] = np.round(summary["Avg_Churn_Probability"] * 100, 1)
        summary["Total_Balance_At_Risk"] = np.round(summary["Total_Balance_At_Risk"], 2)
        summary["Total_Annual_Value_At_Risk"] = np.round(summary["Total_Annual_Value_At_Risk"], 2)
        summary["Avg_Balance"] = np.round(summary["Avg_Balance"], 2)
        summary["Avg_Age"] = np.round(summary["Avg_Age"], 1)
        summary["Inactive_Rate"] = np.round(summary["Inactive_Rate"] * 100, 1)

        tier_order = {"High Risk": 1, "Medium Risk": 2, "Low Risk": 3}
        summary["sort_key"] = summary["RiskTier"].map(tier_order)
        summary = summary.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

        return summary

    def simulate_retention_campaign(
        self,
        df_scored: pd.DataFrame,
        target_tiers: List[str] = ["High Risk", "Medium Risk"],
        intervention_cost_per_customer: float = 120.0,
        campaign_save_rate: float = 0.28
    ) -> Dict[str, Any]:
        """
        Simulates the financial ROI of a proactive retention intervention campaign.
        """
        target_cohort = df_scored[df_scored["RiskTier"].isin(target_tiers)]
        targeted_customers = len(target_cohort)
        
        total_campaign_cost = targeted_customers * intervention_cost_per_customer
        expected_churners = target_cohort["ChurnProbability"].sum()
        customers_saved = expected_churners * campaign_save_rate
        
        gross_value_saved = target_cohort["ValueAtRisk"].sum() * campaign_save_rate
        net_financial_gain = gross_value_saved - total_campaign_cost
        roi_percentage = (net_financial_gain / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0.0

        return {
            "targeted_tiers": target_tiers,
            "targeted_customers_count": int(targeted_customers),
            "intervention_cost_per_customer": float(intervention_cost_per_customer),
            "total_campaign_cost": float(np.round(total_campaign_cost, 2)),
            "assumed_save_rate": float(campaign_save_rate),
            "expected_customers_saved": int(np.round(customers_saved)),
            "gross_annual_value_saved": float(np.round(gross_value_saved, 2)),
            "net_financial_benefit": float(np.round(net_financial_gain, 2)),
            "retention_roi_percent": float(np.round(roi_percentage, 1))
        }

    def generate_strategic_playbooks(self) -> List[Dict[str, str]]:
        """Returns structured strategic retention action items tailored to data drivers."""
        return [
            {
                "Segment": "German High-Deposit Accounts (Age 45-60)",
                "Driver": "Competitive wealth alternatives, high account balances, zero-fee expectations",
                "Action_Plan": "Deploy dedicated Relationship Managers; offer preferential 0.50% APY savings booster and wealth advisory consultation.",
                "Expected_Impact": "18-25% reduction in high-balance deposit attrition."
            },
            {
                "Segment": "Multi-Product Inactive Customers (3+ Products)",
                "Driver": "Product fragmentation, orphaned cards, lack of primary banking engagement",
                "Action_Plan": "Launch automated digital product consolidation workflow with fee-rebate incentive upon 3 monthly mobile logins.",
                "Expected_Impact": "30% increase in digital engagement; 15% reduction in product closure."
            },
            {
                "Segment": "Mature Age Cohort (Age 50-65) with High Balances",
                "Driver": "Retirement asset reallocation and competitive rate shopping",
                "Action_Plan": "Proactive outreach by Senior Wealth Planners with tailored fixed-income and estate advisory offerings.",
                "Expected_Impact": "22% defection mitigation among mature depositors."
            }
        ]
