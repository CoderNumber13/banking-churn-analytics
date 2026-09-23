"""
Data Loader and Schema Validator for Retail Banking Customer Churn
"""

import os
import pandas as pd
from typing import Optional

class BankDataLoader:
    """
    Handles loading, schema validation, and data verification
    for retail banking customer churn analysis.
    """

    EXPECTED_COLUMNS = [
        "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
        "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary", "Exited"
    ]

    def __init__(self, raw_data_path: Optional[str] = None):
        self.raw_data_path = raw_data_path

    def load_or_create(self, force_generate: bool = False) -> pd.DataFrame:
        """
        Loads the customer churn dataset from disk and validates schema.
        """
        if not self.raw_data_path or not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.raw_data_path}")

        print(f"[DataLoader] Loading customer dataset from: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        self.validate_schema(df)
        return df

    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> bool:
        """Validates that all expected columns exist."""
        missing = [col for col in cls.EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Schema validation failed! Missing columns: {missing}")
        print(f"[DataLoader] Schema validation passed. Records: {len(df):,}, Columns: {df.shape[1]}")
        return True


if __name__ == "__main__":
    loader = BankDataLoader("data/raw/bank_customer_churn_raw.csv")
    df = loader.load_or_create()
    print("Columns:", df.columns.tolist())
    print(f"Overall Churn Rate: {df['Exited'].mean():.2%}")
