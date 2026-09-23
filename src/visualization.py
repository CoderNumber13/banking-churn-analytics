"""
Publication-Quality Visualization Engine (Matplotlib, Seaborn, and Plotly)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold"
})

CHURN_PALETTE = {0: "#2B6CB0", 1: "#E53E3E"}

class BankVisualizer:
    """
    Renders publication-ready static charts and figures.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_demographic_churn_breakdowns(self, df: pd.DataFrame, filename: str = "01_demographic_breakdowns.png") -> str:
        """Plots churn rate breakdowns across Geography, Products, Activity & Gender."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        
        # 1. Geography
        geo_churn = df.groupby("Geography", as_index=False)["Exited"].mean()
        sns.barplot(data=geo_churn, x="Geography", y="Exited", hue="Geography", palette="Blues_r", legend=False, ax=axes[0, 0])
        axes[0, 0].set_title("Churn Rate by Geography (Germany ~2x)", pad=12)
        axes[0, 0].set_ylabel("Churn Rate")
        axes[0, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for p in axes[0, 0].patches:
            axes[0, 0].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", fontweight="bold")

        # 2. Number of Products
        prod_churn = df.groupby("NumOfProducts", as_index=False)["Exited"].mean()
        sns.barplot(data=prod_churn, x="NumOfProducts", y="Exited", hue="NumOfProducts", palette="Reds", legend=False, ax=axes[0, 1])
        axes[0, 1].set_title("Churn Rate by Number of Products", pad=12)
        axes[0, 1].set_ylabel("Churn Rate")
        axes[0, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for p in axes[0, 1].patches:
            axes[0, 1].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", fontweight="bold")

        # 3. Active Membership vs Gender
        sns.barplot(data=df, x="IsActiveMember", y="Exited", hue="Gender", palette="Set2", ax=axes[1, 0])
        axes[1, 0].set_title("Churn Rate by Activity Status & Gender", pad=12)
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_xticklabels(["Inactive (0)", "Active (1)"])
        axes[1, 0].set_ylabel("Churn Rate")
        axes[1, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

        # 4. Tenure distribution
        tenure_churn = df.groupby("Tenure", as_index=False)["Exited"].mean()
        sns.barplot(data=tenure_churn, x="Tenure", y="Exited", hue="Tenure", palette="crest", legend=False, ax=axes[1, 1])
        axes[1, 1].set_title("Churn Rate across Relationship Tenure (Years)", pad=12)
        axes[1, 1].set_ylabel("Churn Rate")
        axes[1, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

        plt.suptitle("Banking Customer Demographics & Behavioral Churn Drivers", y=1.02, fontsize=17, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_financial_distributions(self, df: pd.DataFrame, filename: str = "02_financial_distributions.png") -> str:
        """Plots distribution of Age, Balance, and Credit Score across churned vs retained."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.kdeplot(data=df, x="Age", hue="Exited", common_norm=False, fill=True, palette=CHURN_PALETTE, alpha=0.4, ax=axes[0])
        axes[0].set_title("Age Distribution by Churn Status")
        axes[0].legend(["Churned (1)", "Retained (0)"], loc="upper right")

        non_zero_bal = df[df["Balance"] > 0]
        sns.kdeplot(data=non_zero_bal, x="Balance", hue="Exited", common_norm=False, fill=True, palette=CHURN_PALETTE, alpha=0.4, ax=axes[1])
        axes[1].set_title("Account Balance Distribution (> $0)")
        axes[1].legend(["Churned (1)", "Retained (0)"], loc="upper right")

        sns.kdeplot(data=df, x="CreditScore", hue="Exited", common_norm=False, fill=True, palette=CHURN_PALETTE, alpha=0.4, ax=axes[2])
        axes[2].set_title("Credit Score Distribution")
        axes[2].legend(["Churned (1)", "Retained (0)"], loc="upper left")

        plt.suptitle("Financial & Demographic Density Distributions", y=1.05, fontsize=16, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_correlation_matrix(self, corr_df: pd.DataFrame, filename: str = "03_correlation_matrix.png") -> str:
        """Generates an annotated correlation heatmap."""
        plt.figure(figsize=(11, 9))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            corr_df,
            mask=mask,
            cmap=cmap,
            vmax=0.6,
            vmin=-0.6,
            center=0,
            square=True,
            linewidths=.5,
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": .8}
        )
        plt.title("Correlation Matrix of Customer & Financial Attributes", pad=15, fontsize=15, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_model_comparison(
        self,
        cv_results: Dict[str, Dict[str, float]],
        roc_data: Dict[str, Tuple[List[float], List[float], float]],
        pr_data: Dict[str, Tuple[List[float], List[float], float]],
        filename: str = "04_model_evaluation.png"
    ) -> str:
        """Plots cross-validation benchmark metrics, ROC curves, and Precision-Recall curves."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        models = list(cv_results.keys())
        roc_means = [cv_results[m]["roc_auc_mean"] for m in models]
        roc_stds = [cv_results[m]["roc_auc_std"] for m in models]
        pr_means = [cv_results[m]["pr_auc_mean"] for m in models]

        x = np.arange(len(models))
        width = 0.35

        axes[0].bar(x - width/2, roc_means, width, yerr=roc_stds, label="ROC-AUC", color="#2B6CB0", capsize=5)
        axes[0].bar(x + width/2, pr_means, width, label="PR-AUC", color="#DD6B20")

        axes[0].set_title("5-Fold Cross-Validation Metrics")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=15, ha="right")
        axes[0].set_ylim(0.4, 1.0)
        axes[0].legend(loc="lower right")

        for name, (fpr, tpr, auc_score) in roc_data.items():
            axes[1].plot(fpr, tpr, lw=2.5, label=f"{name} (AUC = {auc_score:.3f})")
        axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
        axes[1].set_title("Receiver Operating Characteristic (ROC) Curves")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right")

        for name, (precision, recall, pr_auc) in pr_data.items():
            axes[2].plot(recall, precision, lw=2.5, label=f"{name} (PR-AUC = {pr_auc:.3f})")
        axes[2].set_title("Precision-Recall Curves")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].legend(loc="lower left")

        plt.suptitle("Predictive Model Performance Comparison", y=1.03, fontsize=16, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 10, filename: str = "05_feature_importance.png") -> str:
        """Plots top permutation feature importances."""
        plt.figure(figsize=(11, 6))
        top_df = importance_df.head(top_n).sort_values(by="Importance_Mean", ascending=True)

        bars = plt.barh(top_df["Feature"], top_df["Importance_Mean"], xerr=top_df["Importance_Std"], color="#3182CE", capsize=4)
        plt.title(f"Top {top_n} Permutation Feature Importances (ROC-AUC Impact)", pad=15, fontsize=14, fontweight="bold")
        plt.xlabel("Mean Decrease in ROC-AUC")
        
        for bar in bars:
            val = bar.get_width()
            plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_risk_tiers(self, tier_summary: pd.DataFrame, filename: str = "06_risk_tier_analysis.png") -> str:
        """Plots customer volume vs value at risk across risk tiers."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        colors = ["#E53E3E", "#DD6B20", "#38A169"]

        sns.barplot(data=tier_summary, x="RiskTier", y="Total_Customers", hue="RiskTier", palette=colors, legend=False, ax=axes[0])
        axes[0].set_title("Customer Volume by Churn Risk Tier")
        axes[0].set_ylabel("Number of Customers")
        for p in axes[0].patches:
            axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", fontweight="bold")

        sns.barplot(data=tier_summary, x="RiskTier", y="Total_Annual_Value_At_Risk", hue="RiskTier", palette=colors, legend=False, ax=axes[1])
        axes[1].set_title("Annual Expected Revenue at Risk ($)")
        axes[1].set_ylabel("Revenue at Risk ($)")
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
        for p in axes[1].patches:
            axes[1].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", fontweight="bold")

        plt.suptitle("Customer Segmentation & Financial Exposure Summary", y=1.03, fontsize=16, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
