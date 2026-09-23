"""
Automated Jupyter Notebook Builder for Banking Customer Churn Project
"""

import os
import nbformat as nbf

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook_01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 01. Data Ingestion, Profiling & Quality Audit
## Banking Customer Churn Analytics & Retention Pipeline

### Executive Summary & Purpose
This notebook performs the foundational data ingestion, schema verification, data quality auditing, missing value profiling, and outlier identification on the 10,000-customer retail banking dataset (`Churn_Modelling.csv`).

---
### Dataset Schema Overview
- **Identifiers**: `RowNumber`, `CustomerId`, `Surname`
- **Demographics**: `Geography` (France, Germany, Spain), `Gender`, `Age`
- **Account & Financial Attributes**: `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- **Target Variable**: `Exited` (1 = Churned, 0 = Retained)
"""),
        nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.abspath(".."))
from src.data_loader import BankDataLoader
from src.eda import BankChurnEDA

# Set visual styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
"""),
        nbf.v4.new_markdown_cell("## 1. Load & Validate Raw Banking Data"),
        nbf.v4.new_code_cell("""# Load raw dataset
data_path = os.path.abspath("../data/raw/bank_customer_churn_raw.csv")
loader = BankDataLoader(data_path)
df = loader.load_or_create()

print(f"Dataset Shape: {df.shape}")
df.head()
"""),
        nbf.v4.new_markdown_cell("## 2. Data Health & Quality Profiling"),
        nbf.v4.new_code_cell("""eda_engine = BankChurnEDA(df)
audit_results = eda_engine.perform_quality_audit()

print(f"Total Audited Records: {audit_results['total_records']:,}")
print(f"Duplicate Customer IDs: {audit_results['duplicate_customer_ids']}")
print(f"Overall Churn Rate:     {audit_results['churn_rate']:.2%}")
print("\\nMissing Value Counts by Column:")
print(pd.Series(audit_results['missing_values']))
"""),
        nbf.v4.new_markdown_cell("## 3. Summary Statistics & Data Types"),
        nbf.v4.new_code_cell("""# Numerical features descriptive statistics
df.describe().T.round(2)
"""),
        nbf.v4.new_markdown_cell("## 4. Outlier Detection (Interquartile Range - IQR)"),
        nbf.v4.new_code_cell("""outliers_df = pd.DataFrame(audit_results['outliers_iqr']).T
outliers_df
"""),
        nbf.v4.new_markdown_cell("## 5. Export Cleaned Dataset for Downstream Analysis"),
        nbf.v4.new_code_cell("""processed_path = os.path.abspath("../data/processed/bank_customer_churn_clean.csv")
os.makedirs(os.path.dirname(processed_path), exist_ok=True)
df.to_csv(processed_path, index=False)
print(f"Successfully saved clean dataset to: {processed_path}")
""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "01_data_cleaning_and_quality_audit.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Created] {path}")

def create_notebook_02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 02. Exploratory Data Analysis & Statistical Hypothesis Testing
## Banking Customer Churn Analytics

### Objectives
1. Perform bivariate and multivariate exploratory analysis across demographic and financial dimensions.
2. Conduct formal statistical hypothesis tests:
   - **Chi-Square Tests of Independence** & **Cramer's V** for categorical features (`Geography`, `Gender`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`).
   - **Mann-Whitney U** and **Independent Two-Sample T-Tests** with **Cohen's d** effect sizes for numerical features (`CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`).
3. Identify core behavioral and structural drivers behind customer attrition.
"""),
        nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(".."))
from src.eda import BankChurnEDA
from src.visualization import BankVisualizer

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/processed/bank_customer_churn_clean.csv")
print(f"Loaded Clean Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
"""),
        nbf.v4.new_markdown_cell("## 1. Overall Churn Distribution"),
        nbf.v4.new_code_cell("""churn_counts = df['Exited'].value_counts()
churn_pct = df['Exited'].value_counts(normalize=True) * 100

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=churn_counts.index, y=churn_counts.values, hue=churn_counts.index, palette=["#2B6CB0", "#E53E3E"], legend=False, ax=ax[0])
ax[0].set_title("Customer Volume by Churn Status", fontweight="bold")
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(["Retained (0)", "Churned (1)"])
ax[0].set_ylabel("Customer Count")

ax[1].pie(churn_pct, labels=["Retained", "Churned"], autopct="%1.1f%%", colors=["#2B6CB0", "#E53E3E"], explode=(0, 0.08), startangle=140)
ax[1].set_title("Overall Churn Proportion", fontweight="bold")
plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("## 2. Categorical Drivers & Statistical Hypothesis Testing (Chi-Square & Cramer's V)"),
        nbf.v4.new_code_cell("""eda = BankChurnEDA(df)
cat_results = eda.analyze_categorical_features()

summary_records = []
for col, res in cat_results.items():
    summary_records.append({
        "Feature": col,
        "Cramer_V": res["cramers_v"],
        "Chi2_Stat": res["chi2_statistic"],
        "p_value": res["p_value"],
        "Statistically_Significant": res["is_statistically_significant"]
    })

pd.DataFrame(summary_records).sort_values(by="Cramer_V", ascending=False)
"""),
        nbf.v4.new_markdown_cell("## 3. Visualizing Key Categorical Drivers"),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Geography
geo_df = df.groupby("Geography", as_index=False)["Exited"].mean()
sns.barplot(data=geo_df, x="Geography", y="Exited", hue="Geography", palette="Blues_r", legend=False, ax=axes[0, 0])
axes[0, 0].set_title("Churn Rate by Geography (Germany has ~2x churn)", fontweight="bold")
axes[0, 0].set_ylabel("Churn Rate")

# Products
prod_df = df.groupby("NumOfProducts", as_index=False)["Exited"].mean()
sns.barplot(data=prod_df, x="NumOfProducts", y="Exited", hue="NumOfProducts", palette="Reds", legend=False, ax=axes[0, 1])
axes[0, 1].set_title("Churn Rate by Number of Products (Severe defection with 3+ products)", fontweight="bold")
axes[0, 1].set_ylabel("Churn Rate")

# IsActiveMember
act_df = df.groupby("IsActiveMember", as_index=False)["Exited"].mean()
sns.barplot(data=act_df, x="IsActiveMember", y="Exited", hue="IsActiveMember", palette="Set2", legend=False, ax=axes[1, 0])
axes[1, 0].set_title("Churn Rate by Active Membership Status", fontweight="bold")
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(["Inactive (0)", "Active (1)"])
axes[1, 0].set_ylabel("Churn Rate")

# Gender
gen_df = df.groupby("Gender", as_index=False)["Exited"].mean()
sns.barplot(data=gen_df, x="Gender", y="Exited", hue="Gender", palette="coolwarm", legend=False, ax=axes[1, 1])
axes[1, 1].set_title("Churn Rate by Customer Gender", fontweight="bold")
axes[1, 1].set_ylabel("Churn Rate")

plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("## 4. Numerical Drivers & Hypothesis Testing (Mann-Whitney U & Cohen's d)"),
        nbf.v4.new_code_cell("""num_results = eda.analyze_numerical_features()
num_summary = []
for col, res in num_results.items():
    num_summary.append({
        "Feature": col,
        "Cohen_d": res["cohens_d"],
        "Churned_Mean": res["churned_mean"],
        "Retained_Mean": res["retained_mean"],
        "MWU_p_val": res["mwu_p_value"],
        "Statistically_Significant": res["is_statistically_significant"]
    })

pd.DataFrame(num_summary).sort_values(by="Cohen_d", key=abs, ascending=False)
"""),
        nbf.v4.new_markdown_cell("## 5. Numerical Feature Distributions (KDE Density Plots)"),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
palette = {0: "#2B6CB0", 1: "#E53E3E"}

# Age
sns.kdeplot(data=df, x="Age", hue="Exited", common_norm=False, fill=True, palette=palette, alpha=0.35, ax=axes[0])
axes[0].set_title("Age Distribution (Peak defection between 45-60)", fontweight="bold")

# Balance
sns.kdeplot(data=df[df["Balance"] > 0], x="Balance", hue="Exited", common_norm=False, fill=True, palette=palette, alpha=0.35, ax=axes[1])
axes[1].set_title("Account Balance Distribution (> $0)", fontweight="bold")

# Credit Score
sns.kdeplot(data=df, x="CreditScore", hue="Exited", common_norm=False, fill=True, palette=palette, alpha=0.35, ax=axes[2])
axes[2].set_title("Credit Score Distribution", fontweight="bold")

plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("## 6. Correlation Heatmap & Feature Interdependence"),
        nbf.v4.new_code_cell("""corr_mat = eda.get_correlation_matrix()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix of Banking Features", fontweight="bold", pad=12)
plt.tight_layout()
plt.show()
""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "02_exploratory_data_analysis.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Created] {path}")

def create_notebook_03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 03. Feature Engineering & Preprocessing Pipeline
## Banking Customer Churn Analytics

### Objectives
1. Engineer domain-specific financial, demographic, and behavioral features (`BalanceSalaryRatio`, `TenureAgeRatio`, `EngagementScore`, `HighRiskDemographic`).
2. Build a reusable, leak-free Scikit-Learn `ColumnTransformer` preprocessing pipeline.
3. Perform stratified train/test split and inspect transformed predictor matrices.
"""),
        nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(".."))
from src.feature_engineering import BankFeatureEngineer

df = pd.read_csv("../data/processed/bank_customer_churn_clean.csv")
print(f"Loaded Raw Dataset: {df.shape}")
"""),
        nbf.v4.new_markdown_cell("## 1. Domain Feature Engineering"),
        nbf.v4.new_code_cell("""fe = BankFeatureEngineer()
df_featured = fe.create_features(df)

engineered_cols = [
    "BalanceSalaryRatio", "TenureAgeRatio", "ProductTenureRatio",
    "IsZeroBalance", "AgeGroup", "CreditScoreTier",
    "EngagementScore", "HighRiskDemographic", "MultiProductWarning", "BalancePerProduct"
]

print("Sample of newly engineered features:")
df_featured[engineered_cols].head(6)
"""),
        nbf.v4.new_markdown_cell("## 2. Analyzing Engineered Features vs Churn"),
        nbf.v4.new_code_cell("""print("--- Churn Rate by Generational Age Group ---")
print(df_featured.groupby("AgeGroup")["Exited"].agg(["count", "mean"]).rename(columns={"mean": "ChurnRate"}).round(3))

print("\\n--- Churn Rate by High Risk Demographic Flag ---")
print(df_featured.groupby("HighRiskDemographic")["Exited"].agg(["count", "mean"]).rename(columns={"mean": "ChurnRate"}).round(3))
"""),
        nbf.v4.new_markdown_cell("## 3. Building Scikit-Learn Preprocessing Pipeline"),
        nbf.v4.new_code_cell("""preprocessor = fe.build_preprocessor()
print("ColumnTransformer Architecture:")
print(preprocessor)
"""),
        nbf.v4.new_markdown_cell("## 4. Stratified Train/Test Split"),
        nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test, feat_cols = fe.prepare_train_test_data(df, test_size=0.20, random_state=42)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"Train Churn Rate: {y_train.mean():.2%}")
print(f"Test Churn Rate:  {y_test.mean():.2%}")
"""),
        nbf.v4.new_markdown_cell("## 5. Save Feature-Engineered Dataset"),
        nbf.v4.new_code_cell("""feat_path = os.path.abspath("../data/processed/bank_customer_churn_features.csv")
df_featured.to_csv(feat_path, index=False)
print(f"Saved feature dataset to: {feat_path}")
""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "03_feature_engineering_and_selection.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Created] {path}")

def create_notebook_04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 04. Predictive Modeling, Evaluation & Explainability
## Banking Customer Churn Analytics

### Objectives
1. Train and benchmark candidate algorithms with 5-Fold Stratified Cross-Validation:
   - **Logistic Regression** (L2 regularized baseline)
   - **Random Forest Classifier**
   - **HistGradientBoosting Classifier**
2. Fine-tune champion hyperparameters using `GridSearchCV`.
3. Evaluate holdout test performance (ROC-AUC, PR-AUC, Confusion Matrix, Decision Threshold Tuning).
4. Perform Permutation Feature Importance for Explainable AI (XAI).
"""),
        nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(".."))
from src.feature_engineering import BankFeatureEngineer
from src.modeling import BankChurnModeler
from src.evaluation import BankChurnEvaluator

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/processed/bank_customer_churn_clean.csv")
fe = BankFeatureEngineer()
X_train, X_test, y_train, y_test, feat_cols = fe.prepare_train_test_data(df)
"""),
        nbf.v4.new_markdown_cell("## 1. Candidate Algorithm Cross-Validation Benchmark"),
        nbf.v4.new_code_cell("""modeler = BankChurnModeler(random_state=42)
cv_results = modeler.train_and_cross_validate(X_train, y_train, n_splits=5)

benchmark_df = pd.DataFrame(cv_results).T
benchmark_df[["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "f1_mean", "recall_mean", "accuracy_mean"]].round(4)
"""),
        nbf.v4.new_markdown_cell("## 2. Hyperparameter Fine-Tuning"),
        nbf.v4.new_code_cell("""champion_model = modeler.tune_champion_hyperparameters(X_train, y_train)
"""),
        nbf.v4.new_markdown_cell("## 3. Holdout Test Set Evaluation & ROC / PR Curves"),
        nbf.v4.new_code_cell("""evaluator = BankChurnEvaluator()
y_probs = champion_model.predict_proba(X_test)[:, 1]

test_metrics = evaluator.evaluate_predictions(y_test, y_probs)
print(f"Test Set ROC-AUC: {test_metrics['roc_auc']:.4f}")
print(f"Test Set PR-AUC:  {test_metrics['pr_auc']:.4f}")
print(f"Test Set F1:      {test_metrics['f1_score']:.4f}")
print(f"Test Set Recall:  {test_metrics['recall']:.4f}")
print(f"Test Set Precision: {test_metrics['precision']:.4f}")
"""),
        nbf.v4.new_markdown_cell("## 4. Decision Threshold Optimization for Business Profit"),
        nbf.v4.new_code_cell("""opt_thresh, thresh_df = evaluator.optimize_decision_threshold(
    y_test, y_probs,
    cost_false_negative=2000.0,
    cost_false_positive=150.0,
    cost_true_positive=250.0
)

print(f"Optimal Business Decision Threshold: {opt_thresh:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].plot(thresh_df["threshold"], thresh_df["business_loss"], color="#E53E3E", lw=2.5)
ax[0].axvline(opt_thresh, color="black", linestyle="--", label=f"Optimal ({opt_thresh:.2f})")
ax[0].set_title("Total Expected Business Loss vs Decision Threshold", fontweight="bold")
ax[0].set_xlabel("Probability Threshold")
ax[0].set_ylabel("Expected Business Loss ($)")
ax[0].legend()

ax[1].plot(thresh_df["threshold"], thresh_df["f1_score"], label="F1-Score", color="#2B6CB0", lw=2)
ax[1].plot(thresh_df["threshold"], thresh_df["recall"], label="Recall", color="#38A169", lw=2)
ax[1].plot(thresh_df["threshold"], thresh_df["precision"], label="Precision", color="#DD6B20", lw=2)
ax[1].set_title("Metric Tradeoffs Across Thresholds", fontweight="bold")
ax[1].set_xlabel("Probability Threshold")
ax[1].legend()
plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("## 5. Permutation Feature Importance (Explainable AI - XAI)"),
        nbf.v4.new_code_cell("""importance_df = evaluator.compute_permutation_importances(champion_model, X_test, y_test)

plt.figure(figsize=(11, 6))
top_10 = importance_df.head(10).sort_values(by="Importance_Mean", ascending=True)
plt.barh(top_10["Feature"], top_10["Importance_Mean"], xerr=top_10["Importance_Std"], color="#3182CE", capsize=4)
plt.title("Top 10 Feature Importances (Mean ROC-AUC Drop)", fontweight="bold", pad=12)
plt.xlabel("Importance Mean")
plt.tight_layout()
plt.show()
""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "04_predictive_modeling_and_evaluation.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Created] {path}")

def create_notebook_05():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 05. Business Impact, Risk Tiering & Retention Strategies
## Banking Customer Churn Analytics

### Objectives
1. Segment customers into actionable Churn Risk Tiers (**High**, **Medium**, **Low**).
2. Calculate total deposit balances and customer lifetime value at risk ($).
3. Simulate proactive retention campaign ROI (cost vs saved value).
4. Review targeted strategic retention playbooks.
"""),
        nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(".."))
from src.feature_engineering import BankFeatureEngineer
from src.modeling import BankChurnModeler
from src.business_insights import BankBusinessInsights

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/processed/bank_customer_churn_clean.csv")
fe = BankFeatureEngineer()
df_featured = fe.create_features(df)

X_train, X_test, y_train, y_test, _ = fe.prepare_train_test_data(df)
modeler = BankChurnModeler()
modeler.train_and_cross_validate(X_train, y_train)
champion = modeler.tune_champion_hyperparameters(X_train, y_train)

drop_cols = ["RowNumber", "CustomerId", "Surname", "Exited"]
all_X = df_featured.drop(columns=[c for c in drop_cols if c in df_featured.columns])
churn_probs = champion.predict_proba(all_X)[:, 1]
"""),
        nbf.v4.new_markdown_cell("## 1. Assign Risk Tiers & Quantify Revenue at Risk"),
        nbf.v4.new_code_cell("""insights = BankBusinessInsights(high_risk_cutoff=0.65, medium_risk_cutoff=0.35)
df_scored = insights.assign_risk_tiers(df_featured, churn_probs)

tier_summary = insights.calculate_tier_financial_summary(df_scored)
tier_summary
"""),
        nbf.v4.new_markdown_cell("## 2. Visualizing Financial Exposure Across Risk Tiers"),
        nbf.v4.new_code_cell("""fig, ax = plt.subplots(1, 2, figsize=(16, 5))
palette = ["#E53E3E", "#DD6B20", "#38A169"]

sns.barplot(data=tier_summary, x="RiskTier", y="Total_Customers", hue="RiskTier", palette=palette, legend=False, ax=ax[0])
ax[0].set_title("Customer Volume by Risk Tier", fontweight="bold")
ax[0].set_ylabel("Total Customers")

sns.barplot(data=tier_summary, x="RiskTier", y="Total_Annual_Value_At_Risk", hue="RiskTier", palette=palette, legend=False, ax=ax[1])
ax[1].set_title("Annual Expected Value at Risk ($)", fontweight="bold")
ax[1].set_ylabel("Revenue at Risk ($)")

plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("## 3. Retention Campaign ROI Simulator"),
        nbf.v4.new_code_cell("""sim_results = insights.simulate_retention_campaign(
    df_scored,
    target_tiers=["High Risk", "Medium Risk"],
    intervention_cost_per_customer=120.0,
    campaign_save_rate=0.28
)

print("--- PROACTIVE RETENTION CAMPAIGN FINANCIAL MODEL ---")
for k, v in sim_results.items():
    print(f"- {k:<30}: {v}")
"""),
        nbf.v4.new_markdown_cell("## 4. Strategic Retention Playbooks"),
        nbf.v4.new_code_cell("""playbooks = insights.generate_strategic_playbooks()
for idx, p in enumerate(playbooks, 1):
    print(f"\\n{'='*70}\\n[PLAYBOOK {idx}] Segment: {p['Segment']}\\n{'='*70}")
    print(f"- Core Drivers:    {p['Driver']}")
    print(f"- Strategic Action:{p['Action_Plan']}")
    print(f"- Projected Impact:{p['Expected_Impact']}")
""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "05_business_insights_and_retention_strategies.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Created] {path}")

def main():
    create_notebook_01()
    create_notebook_02()
    create_notebook_03()
    create_notebook_04()
    create_notebook_05()
    print("\\n[SUCCESS] All 5 Jupyter Notebooks built successfully.")

if __name__ == "__main__":
    main()
