"""
Exploratory Data Analysis, Data Quality Profiling & Statistical Hypothesis Testing
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional

class BankChurnEDA:
    """
    Automates data quality auditing, bivariate analysis, and statistical hypothesis
    testing for banking customer churn.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = "Exited"):
        self.df = df.copy()
        self.target_col = target_col

    def perform_quality_audit(self) -> Dict[str, Any]:
        """Performs comprehensive data quality and integrity checks."""
        total_records = len(self.df)
        duplicates = self.df["CustomerId"].duplicated().sum() if "CustomerId" in self.df.columns else 0
        missing_summary = self.df.isnull().sum().to_dict()
        data_types = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for c in ["CustomerId", "RowNumber", self.target_col]:
            if c in numeric_cols:
                numeric_cols.remove(c)
            
        outlier_summary = {}
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_summary[col] = {
                "outlier_count": int(outliers),
                "outlier_pct": float(np.round((outliers / total_records) * 100, 2)),
                "lower_bound": float(np.round(lower_bound, 2)),
                "upper_bound": float(np.round(upper_bound, 2))
            }

        return {
            "total_records": total_records,
            "total_columns": self.df.shape[1],
            "duplicate_customer_ids": int(duplicates),
            "missing_values": missing_summary,
            "data_types": data_types,
            "outliers_iqr": outlier_summary,
            "churn_rate": float(self.df[self.target_col].mean())
        }

    def analyze_categorical_features(self, cat_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Runs Chi-Square tests of independence and calculates Cramer's V
        for categorical features against customer churn.
        """
        if cat_cols is None:
            candidates = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember", "CardType", "SatisfactionScore", "Complain"]
            cat_cols = [c for c in candidates if c in self.df.columns]
        
        results = {}
        for col in cat_cols:
            if col not in self.df.columns:
                continue
            
            contingency_tab = pd.crosstab(self.df[col], self.df[self.target_col])
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency_tab)
            
            n = contingency_tab.sum().sum()
            min_dim = min(contingency_tab.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
            
            churn_rates = self.df.groupby(col)[self.target_col].agg(["count", "mean"]).rename(
                columns={"count": "total_customers", "mean": "churn_rate"}
            ).to_dict(orient="index")

            results[col] = {
                "chi2_statistic": float(np.round(chi2, 4)),
                "p_value": float(p_val),
                "degrees_of_freedom": int(dof),
                "cramers_v": float(np.round(cramers_v, 4)),
                "is_statistically_significant": bool(p_val < 0.05),
                "breakdown": churn_rates
            }

        return results

    def analyze_numerical_features(self, num_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Performs 2-sample T-test and Mann-Whitney U test on numerical features
        between churned and retained customer segments.
        """
        if num_cols is None:
            candidates = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary", "PointEarned"]
            num_cols = [c for c in candidates if c in self.df.columns]

        churned = self.df[self.df[self.target_col] == 1]
        retained = self.df[self.df[self.target_col] == 0]

        results = {}
        for col in num_cols:
            if col not in self.df.columns:
                continue
                
            c_vals = churned[col].dropna()
            r_vals = retained[col].dropna()

            u_stat, p_mwu = stats.mannwhitneyu(c_vals, r_vals, alternative="two-sided")
            t_stat, p_ttest = stats.ttest_ind(c_vals, r_vals, equal_var=False)

            n1, n2 = len(c_vals), len(r_vals)
            s1, s2 = np.var(c_vals, ddof=1), np.var(r_vals, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            cohens_d = (c_vals.mean() - r_vals.mean()) / pooled_std if pooled_std > 0 else 0.0

            results[col] = {
                "churned_mean": float(np.round(c_vals.mean(), 2)),
                "churned_median": float(np.round(c_vals.median(), 2)),
                "churned_std": float(np.round(c_vals.std(), 2)),
                "retained_mean": float(np.round(r_vals.mean(), 2)),
                "retained_median": float(np.round(r_vals.median(), 2)),
                "retained_std": float(np.round(r_vals.std(), 2)),
                "mann_whitney_u": float(np.round(u_stat, 2)),
                "mwu_p_value": float(p_mwu),
                "t_statistic": float(np.round(t_stat, 4)),
                "ttest_p_value": float(p_ttest),
                "cohens_d": float(np.round(cohens_d, 4)),
                "is_statistically_significant": bool(p_mwu < 0.05)
            }

        return results

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Returns Pearson correlation matrix for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        drop_cols = ["CustomerId", "RowNumber"]
        numeric_df = numeric_df.drop(columns=[c for c in drop_cols if c in numeric_df.columns])
        return numeric_df.corr().round(3)

    def print_executive_eda_summary(self) -> None:
        """Prints a human-readable EDA summary to standard output."""
        audit = self.perform_quality_audit()
        print("\n" + "=" * 60)
        print("BANKING CUSTOMER CHURN: DATA AUDIT & STATISTICAL SUMMARY")
        print("=" * 60)
        print(f"Total Customer Records: {audit['total_records']:,}")
        print(f"Overall Churn Rate:     {audit['churn_rate']:.2%}")
        print(f"Duplicate Customer IDs: {audit['duplicate_customer_ids']}")
        
        print("\n--- Key Categorical Drivers (Chi-Square & Cramer's V) ---")
        cat_analysis = self.analyze_categorical_features()
        for col, stat in sorted(cat_analysis.items(), key=lambda x: x[1]['cramers_v'], reverse=True):
            sig = "[Statistically Significant]" if stat["is_statistically_significant"] else "[Not Significant]"
            print(f"- {col:<18} | Cramer's V: {stat['cramers_v']:.4f} | p-val: {stat['p_value']:.2e} | {sig}")

        print("\n--- Key Numerical Drivers (Cohen's d & Mann-Whitney U) ---")
        num_analysis = self.analyze_numerical_features()
        for col, stat in sorted(num_analysis.items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True):
            sig = "[Statistically Significant]" if stat["is_statistically_significant"] else "[Not Significant]"
            print(f"- {col:<18} | Cohen's d: {stat['cohens_d']:>+6.3f} | Churned: {stat['churned_mean']:>9.2f} vs Retained: {stat['retained_mean']:>9.2f} | {sig}")
        print("=" * 60)


if __name__ == "__main__":
    from src.data_loader import BankDataLoader
    loader = BankDataLoader("data/raw/bank_customer_churn_raw.csv")
    df_raw = loader.load_or_create()
    eda_engine = BankChurnEDA(df_raw)
    eda_engine.print_executive_eda_summary()
