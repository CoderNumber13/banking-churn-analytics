"""
Predictive Modeling, Cross-Validation, and Hyperparameter Optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from src.feature_engineering import BankFeatureEngineer

class BankChurnModeler:
    """
    Manages end-to-end model training, cross-validation, hyperparameter tuning,
    and model comparison for banking churn prediction.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_engineer = BankFeatureEngineer()
        self.models: Dict[str, Pipeline] = {}
        self.best_model: Optional[Pipeline] = None
        self.cv_results: Dict[str, Dict[str, float]] = {}

    def get_candidate_models(self) -> Dict[str, Any]:
        """Returns candidate classifier instances configured with class-weight handling."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "Gradient Boosting": HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_depth=6,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=self.random_state
            )
        }

    def train_and_cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Builds full pipelines (preprocessor + model) and runs Stratified 5-Fold Cross-Validation.
        """
        candidates = self.get_candidate_models()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scoring = ["roc_auc", "average_precision", "f1", "precision", "recall", "accuracy"]

        print(f"[Modeler] Running {n_splits}-Fold Stratified Cross-Validation on {len(candidates)} candidate models...")

        for name, clf in candidates.items():
            preprocessor = self.feature_engineer.build_preprocessor()
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])

            cv_res = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
            
            summary = {
                "roc_auc_mean": float(np.mean(cv_res["test_roc_auc"])),
                "roc_auc_std": float(np.std(cv_res["test_roc_auc"])),
                "pr_auc_mean": float(np.mean(cv_res["test_average_precision"])),
                "pr_auc_std": float(np.std(cv_res["test_average_precision"])),
                "f1_mean": float(np.mean(cv_res["test_f1"])),
                "f1_std": float(np.std(cv_res["test_f1"])),
                "recall_mean": float(np.mean(cv_res["test_recall"])),
                "precision_mean": float(np.mean(cv_res["test_precision"])),
                "accuracy_mean": float(np.mean(cv_res["test_accuracy"]))
            }
            self.cv_results[name] = summary

            # Fit on full training set
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline

            print(f"  [OK] {name:<22} | ROC-AUC: {summary['roc_auc_mean']:.4f} (+/-{summary['roc_auc_std']:.4f}) | PR-AUC: {summary['pr_auc_mean']:.4f} | F1: {summary['f1_mean']:.4f}")

        # Determine champion model based on ROC-AUC
        best_name = max(self.cv_results.keys(), key=lambda k: self.cv_results[k]["roc_auc_mean"])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        print(f"[Modeler] Champion model selected: '{best_name}' (ROC-AUC: {self.cv_results[best_name]['roc_auc_mean']:.4f})")

        return self.cv_results

    def tune_champion_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Pipeline:
        """
        Executes Grid Search Cross-Validation to fine-tune the Gradient Boosting champion model.
        """
        print("[Modeler] Fine-tuning champion Gradient Boosting model with GridSearchCV...")
        
        preprocessor = self.feature_engineer.build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(class_weight="balanced", random_state=self.random_state))
        ])

        param_grid = {
            "classifier__learning_rate": [0.05, 0.08, 0.12],
            "classifier__max_iter": [150, 250],
            "classifier__max_depth": [5, 7, 9],
            "classifier__min_samples_leaf": [15, 25],
            "classifier__l2_regularization": [0.0, 1.0]
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        print(f"[Modeler] Best Hyperparameters: {grid_search.best_params_}")
        print(f"[Modeler] Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        self.best_model = grid_search.best_estimator_
        self.models["Tuned Gradient Boosting"] = self.best_model
        return self.best_model


if __name__ == "__main__":
    from src.data_loader import BankDataLoader
    loader = BankDataLoader("data/raw/bank_customer_churn_raw.csv")
    df_raw = loader.load_or_create()
    
    fe = BankFeatureEngineer()
    X_train, X_test, y_train, y_test, _ = fe.prepare_train_test_data(df_raw)
    
    modeler = BankChurnModeler()
    modeler.train_and_cross_validate(X_train, y_train)
    modeler.tune_champion_hyperparameters(X_train, y_train)
