"""
Model Evaluation, Threshold Optimization, Calibration & Feature Importance
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score, brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

class BankChurnEvaluator:
    """
    Computes statistical and business-aligned evaluation metrics,
    optimizes decision thresholds, and measures feature importance.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate_predictions(
        y_true: pd.Series,
        y_probs: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Calculates all core performance metrics at a specified decision threshold.
        """
        y_pred = (y_probs >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fpr, tpr, roc_thresh = roc_curve(y_true, y_probs)
        precision_curve, recall_curve, pr_thresh = precision_recall_curve(y_true, y_probs)

        metrics = {
            "threshold": float(threshold),
            "roc_auc": float(roc_auc_score(y_true, y_probs)),
            "pr_auc": float(average_precision_score(y_true, y_probs)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "brier_score": float(brier_score_loss(y_true, y_probs)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            },
            "pr_curve": {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist()
            }
        }
        return metrics

    @staticmethod
    def optimize_decision_threshold(
        y_true: pd.Series,
        y_probs: np.ndarray,
        cost_false_negative: float = 2000.0,  # Lost customer lifetime value (CLV)
        cost_false_positive: float = 150.0,   # Cost of retention campaign offer
        cost_true_positive: float = 250.0     # Cost of successful retention package
    ) -> Tuple[float, pd.DataFrame]:
        """
        Finds the profit-maximizing probability threshold balancing false positive
        retention outreach costs against catastrophic false negative customer loss.
        """
        thresholds = np.linspace(0.05, 0.95, 91)
        records = []

        total_customers = len(y_true)
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Expected business loss calculation
            # FN: Lost customer value
            # FP: Wasted campaign budget
            # TP: Retention cost on customer who would have churned
            business_loss = (fn * cost_false_negative) + (fp * cost_false_positive) + (tp * cost_true_positive)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)

            records.append({
                "threshold": round(t, 2),
                "business_loss": business_loss,
                "f1_score": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            })

        df_thresh = pd.DataFrame(records)
        optimal_row = df_thresh.loc[df_thresh["business_loss"].idxmin()]
        optimal_threshold = float(optimal_row["threshold"])

        return optimal_threshold, df_thresh

    @staticmethod
    def compute_permutation_importances(
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Calculates permutation feature importances for the full sklearn pipeline.
        """
        print("[Evaluator] Computing Permutation Feature Importances on test set...")
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance_Mean": np.round(result.importances_mean, 5),
            "Importance_Std": np.round(result.importances_std, 5)
        }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

        return importance_df


if __name__ == "__main__":
    from src.data_loader import BankDataLoader
    from src.feature_engineering import BankFeatureEngineer
    from src.modeling import BankChurnModeler

    loader = BankDataLoader("data/raw/bank_customer_churn_raw.csv")
    df_raw = loader.load_or_create()
    fe = BankFeatureEngineer()
    X_train, X_test, y_train, y_test, _ = fe.prepare_train_test_data(df_raw)
    
    modeler = BankChurnModeler()
    modeler.train_and_cross_validate(X_train, y_train)
    
    champion = modeler.best_model
    y_probs = champion.predict_proba(X_test)[:, 1]
    
    evaluator = BankChurnEvaluator()
    metrics = evaluator.evaluate_predictions(y_test, y_probs)
    print(f"\nChampion Test ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
    
    opt_t, _ = evaluator.optimize_decision_threshold(y_test, y_probs)
    print(f"Optimal Business Threshold: {opt_t:.2f}")
