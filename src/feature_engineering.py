"""
Feature Engineering and Data Transformation Pipeline for Banking Customer Churn
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

class BankFeatureEngineer:
    """
    Transforms raw banking customer attributes into domain-specific predictive features,
    and manages sklearn preprocessing transformers.
    """

    def __init__(self, target_col: str = "Exited"):
        self.target_col = target_col
        self.preprocessor: Optional[ColumnTransformer] = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers financial, behavioral, and demographic features from the raw dataset.
        """
        df_feat = df.copy()

        # 1. Financial Ratio: Balance-to-Salary Ratio
        df_feat["BalanceSalaryRatio"] = np.round(df_feat["Balance"] / (df_feat["EstimatedSalary"] + 1.0), 4)

        # 2. Tenure-to-Age Ratio (Loyalty intensity relative to customer lifespan)
        df_feat["TenureAgeRatio"] = np.round(df_feat["Tenure"] / df_feat["Age"], 4)

        # 3. Product Adoption Density
        df_feat["ProductTenureRatio"] = np.round(df_feat["NumOfProducts"] / (df_feat["Tenure"] + 1.0), 4)

        # 4. Zero Balance Flag (Differentiates transaction-only vs asset-holding customers)
        df_feat["IsZeroBalance"] = (df_feat["Balance"] == 0).astype(int)

        # 5. Age Group Segmentation
        df_feat["AgeGroup"] = pd.cut(
            df_feat["Age"],
            bins=[0, 30, 45, 60, 120],
            labels=["Under30", "Age30_45", "Age46_60", "Over60"],
            right=False
        ).astype(str)

        # 6. Credit Score Tiers (FICO standard brackets)
        df_feat["CreditScoreTier"] = pd.cut(
            df_feat["CreditScore"],
            bins=[0, 580, 670, 740, 800, 1000],
            labels=["Poor", "Fair", "Good", "VeryGood", "Exceptional"],
            right=False
        ).astype(str)

        # 7. Customer Engagement Index (Composite metric)
        df_feat["EngagementScore"] = np.round(
            (df_feat["IsActiveMember"] * 2.0)
            + (df_feat["HasCrCard"] * 1.0)
            + (df_feat["Tenure"] * 0.15),
            2
        )

        # 8. High Defection Risk Cohort Flag (Interaction: Inactive + Mature Age + High Balance)
        df_feat["HighRiskDemographic"] = (
            (df_feat["Age"] >= 45) &
            (df_feat["IsActiveMember"] == 0) &
            (df_feat["Balance"] > 80000)
        ).astype(int)

        # 9. Multi-Product Warning Flag (Customers with 3 or 4 products have unusually high churn)
        df_feat["MultiProductWarning"] = (df_feat["NumOfProducts"] >= 3).astype(int)

        # 10. Balance per Product
        df_feat["BalancePerProduct"] = np.round(df_feat["Balance"] / df_feat["NumOfProducts"], 2)

        return df_feat

    def build_preprocessor(
        self,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        binary_cols: Optional[List[str]] = None
    ) -> ColumnTransformer:
        """
        Constructs a Scikit-Learn ColumnTransformer for robust categorical encoding
        and numerical scaling.
        """
        if categorical_cols is None:
            categorical_cols = ["Geography", "Gender", "AgeGroup", "CreditScoreTier"]
        
        if numeric_cols is None:
            numeric_cols = [
                "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "EstimatedSalary", "BalanceSalaryRatio", "TenureAgeRatio",
                "ProductTenureRatio", "EngagementScore", "BalancePerProduct"
            ]

        if binary_cols is None:
            binary_cols = ["HasCrCard", "IsActiveMember", "IsZeroBalance", "HighRiskDemographic", "MultiProductWarning"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(), numeric_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
                ("bin", "passthrough", binary_cols)
            ]
        )
        self.preprocessor = preprocessor
        return preprocessor

    def prepare_train_test_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Prepares engineered features and performs stratified train/test split.
        """
        df_featured = self.create_features(df)

        drop_cols = ["RowNumber", "CustomerId", "Surname", self.target_col]
        X = df_featured.drop(columns=[c for c in drop_cols if c in df_featured.columns])
        y = df_featured[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test, list(X.columns)


if __name__ == "__main__":
    from src.data_loader import BankDataLoader
    loader = BankDataLoader("data/raw/bank_customer_churn_raw.csv")
    df_raw = loader.load_or_create()
    
    fe = BankFeatureEngineer()
    X_train, X_test, y_train, y_test, feat_cols = fe.prepare_train_test_data(df_raw)
    print(f"[FeatureEngineer] Total Predictors: {len(feat_cols)}")
    print(f"[FeatureEngineer] X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[FeatureEngineer] Train Churn: {y_train.mean():.2%}, Test Churn: {y_test.mean():.2%}")
