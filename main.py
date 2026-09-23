"""
Master Execution Pipeline: End-to-End Banking Customer Churn Analytics
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Set utf-8 stdout if possible on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.data_loader import BankDataLoader
from src.eda import BankChurnEDA
from src.feature_engineering import BankFeatureEngineer
from src.modeling import BankChurnModeler
from src.evaluation import BankChurnEvaluator
from src.business_insights import BankBusinessInsights
from src.visualization import BankVisualizer
from src.reports import BankReportGenerator

def run_pipeline():
    print("=" * 80)
    print("[START] END-TO-END BANKING CHURN ANALYTICS & MODELING PIPELINE")
    print("=" * 80)

    # Paths
    raw_path = os.path.join(BASE_DIR, "data", "raw", "bank_customer_churn_raw.csv")
    clean_path = os.path.join(BASE_DIR, "data", "processed", "bank_customer_churn_clean.csv")
    feat_path = os.path.join(BASE_DIR, "data", "processed", "bank_customer_churn_features.csv")
    fig_dir = os.path.join(BASE_DIR, "reports", "figures")
    html_report_path = os.path.join(BASE_DIR, "reports", "churn_executive_summary.html")

    os.makedirs(os.path.join(BASE_DIR, "data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

    # -------------------------------------------------------------
    # 1. Data Ingestion & Quality Audit
    # -------------------------------------------------------------
    print("\n[Step 1/6] Ingesting and Auditing Customer Records...")
    loader = BankDataLoader(raw_path)
    df_raw = loader.load_or_create(force_generate=True)
    
    eda = BankChurnEDA(df_raw)
    eda_summary = eda.perform_quality_audit()
    eda.print_executive_eda_summary()
    
    # Save clean dataset
    df_raw.to_csv(clean_path, index=False)
    print(f"[OK] Clean dataset saved to: {clean_path}")

    # -------------------------------------------------------------
    # 2. Exploratory Visualizations & Statistical Exhibits
    # -------------------------------------------------------------
    print("\n[Step 2/6] Generating Statistical & Exploratory Visualizations...")
    viz = BankVisualizer(fig_dir)
    viz.plot_demographic_churn_breakdowns(df_raw, "01_demographic_breakdowns.png")
    viz.plot_financial_distributions(df_raw, "02_financial_distributions.png")
    
    corr_df = eda.get_correlation_matrix()
    viz.plot_correlation_matrix(corr_df, "03_correlation_matrix.png")
    print("[OK] Static high-res figures saved to reports/figures/")

    # -------------------------------------------------------------
    # 3. Feature Engineering & Preprocessing Pipeline
    # -------------------------------------------------------------
    print("\n[Step 3/6] Engineering Domain Financial & Behavioral Features...")
    fe = BankFeatureEngineer()
    df_featured = fe.create_features(df_raw)
    df_featured.to_csv(feat_path, index=False)
    print(f"[OK] Feature-engineered dataset saved to: {feat_path}")

    X_train, X_test, y_train, y_test, feat_cols = fe.prepare_train_test_data(df_raw)
    print(f"  Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")
    print(f"  Engineered Predictor Count: {len(feat_cols)}")

    # -------------------------------------------------------------
    # 4. Predictive Modeling, Cross-Validation & Hyperparameter Tuning
    # -------------------------------------------------------------
    print("\n[Step 4/6] Training & Benchmarking Machine Learning Algorithms...")
    modeler = BankChurnModeler(random_state=42)
    cv_results = modeler.train_and_cross_validate(X_train, y_train, n_splits=5)
    
    # Fine-tune champion
    tuned_champion = modeler.tune_champion_hyperparameters(X_train, y_train)

    # -------------------------------------------------------------
    # 5. Model Evaluation, Calibration & Explainability (XAI)
    # -------------------------------------------------------------
    print("\n[Step 5/6] Evaluating Champion Model on Holdout Test Set & Computing XAI...")
    evaluator = BankChurnEvaluator()
    
    # Collect curves for comparison
    roc_data = {}
    pr_data = {}
    for name, pipe in modeler.models.items():
        probs = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluator.evaluate_predictions(y_test, probs)
        roc_data[name] = (metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], metrics["roc_auc"])
        pr_data[name] = (metrics["pr_curve"]["precision"], metrics["pr_curve"]["recall"], metrics["pr_auc"])

    # Champion predictions
    champion_probs = tuned_champion.predict_proba(X_test)[:, 1]
    test_metrics = evaluator.evaluate_predictions(y_test, champion_probs)
    
    # Optimal threshold
    opt_thresh, thresh_df = evaluator.optimize_decision_threshold(y_test, champion_probs)
    test_metrics["threshold"] = opt_thresh

    print(f"\n--- Champion Test Set Evaluation ---")
    print(f"- ROC-AUC Score:      {test_metrics['roc_auc']:.4f}")
    print(f"- PR-AUC Score:       {test_metrics['pr_auc']:.4f}")
    print(f"- F1-Score:           {test_metrics['f1_score']:.4f}")
    print(f"- Recall:             {test_metrics['recall']:.4f}")
    print(f"- Precision:          {test_metrics['precision']:.4f}")
    print(f"- Optimal Threshold:  {opt_thresh:.2f}")

    viz.plot_model_comparison(cv_results, roc_data, pr_data, "04_model_evaluation.png")

    # Permutation Feature Importance
    importance_df = evaluator.compute_permutation_importances(tuned_champion, X_test, y_test)
    viz.plot_feature_importance(importance_df, top_n=10, filename="05_feature_importance.png")

    # -------------------------------------------------------------
    # 6. Business Impact, Risk Tiering & Executive Reporting
    # -------------------------------------------------------------
    print("\n[Step 6/6] Quantifying Financial Impact & Generating Executive Report...")
    # Score full dataset for global risk tiering
    all_X = df_featured.drop(columns=["CustomerId", "Surname", "Exited"])
    all_probs = tuned_champion.predict_proba(all_X)[:, 1]
    
    insights = BankBusinessInsights()
    df_scored = insights.assign_risk_tiers(df_featured, all_probs)
    tier_summary = insights.calculate_tier_financial_summary(df_scored)
    roi_sim = insights.simulate_retention_campaign(df_scored)
    playbooks = insights.generate_strategic_playbooks()

    viz.plot_risk_tiers(tier_summary, "06_risk_tier_analysis.png")

    # Generate standalone executive HTML dashboard
    report_gen = BankReportGenerator(html_report_path)
    report_gen.generate_html_report(
        eda_summary=eda_summary,
        model_results=cv_results,
        test_metrics=test_metrics,
        tier_summary=tier_summary,
        roi_sim=roi_sim,
        playbooks=playbooks,
        top_features=importance_df
    )

    print("\n" + "=" * 80)
    print("[SUCCESS] END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"- Processed Clean Data:  {clean_path}")
    print(f"- Processed Features:    {feat_path}")
    print(f"- Visualization Figures: {fig_dir}")
    print(f"- Executive HTML Report: {html_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
