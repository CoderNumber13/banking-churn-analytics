"""
Executive HTML Report Generator for Banking Customer Churn Analytics
"""

import os
import pandas as pd
from typing import Dict, Any, List

class BankReportGenerator:
    """
    Assembles all metrics, statistical analyses, predictive model outputs,
    and business strategies into a standalone, modern interactive HTML executive dashboard.
    """

    def __init__(self, output_path: str = "reports/churn_executive_summary.html"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def generate_html_report(
        self,
        eda_summary: Dict[str, Any],
        model_results: Dict[str, Dict[str, float]],
        test_metrics: Dict[str, Any],
        tier_summary: pd.DataFrame,
        roi_sim: Dict[str, Any],
        playbooks: List[Dict[str, str]],
        top_features: pd.DataFrame
    ) -> str:
        """Generates a self-contained, responsive, executive HTML dashboard."""

        tier_rows_html = ""
        for _, row in tier_summary.iterrows():
            badge_class = "badge-danger" if row["RiskTier"] == "High Risk" else ("badge-warning" if row["RiskTier"] == "Medium Risk" else "badge-success")
            tier_rows_html += f"""
            <tr>
                <td><span class="badge {badge_class}">{row['RiskTier']}</span></td>
                <td><strong>{int(row['Total_Customers']):,}</strong> ({row['Pct_Total_Customers']}%)</td>
                <td>{row['Avg_Churn_Probability']:.1f}%</td>
                <td>${row['Total_Balance_At_Risk']:,.2f}</td>
                <td><strong>${row['Total_Annual_Value_At_Risk']:,.2f}</strong></td>
                <td>{row['Avg_Age']:.1f} yrs</td>
                <td>{row['Inactive_Rate']:.1f}%</td>
            </tr>
            """

        playbook_cards_html = ""
        for p in playbooks:
            playbook_cards_html += f"""
            <div class="playbook-card">
                <div class="playbook-header">
                    <span class="playbook-title">{p['Segment']}</span>
                    <span class="playbook-tag">Action Priority: High</span>
                </div>
                <div class="playbook-body">
                    <p><strong>Primary Drivers:</strong> {p['Driver']}</p>
                    <p><strong>Retention Action:</strong> {p['Action_Plan']}</p>
                    <div class="playbook-impact">
                        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z"/></svg>
                        <strong>Projected Impact:</strong> {p['Expected_Impact']}
                    </div>
                </div>
            </div>
            """

        features_rows_html = ""
        for idx, row in top_features.head(8).iterrows():
            pct_bar = min(100, int((row['Importance_Mean'] / max(top_features['Importance_Mean'].max(), 0.0001)) * 100))
            features_rows_html += f"""
            <tr>
                <td><strong>{idx+1}. {row['Feature']}</strong></td>
                <td>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {pct_bar}%;"></div>
                    </div>
                </td>
                <td><code>{row['Importance_Mean']:.4f}</code></td>
                <td>+/-{row['Importance_Std']:.4f}</td>
            </tr>
            """

        models_rows_html = ""
        for name, metrics in model_results.items():
            models_rows_html += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td><code>{metrics['roc_auc_mean']:.4f} +/-{metrics['roc_auc_std']:.4f}</code></td>
                <td><code>{metrics['pr_auc_mean']:.4f} +/-{metrics['pr_auc_std']:.4f}</code></td>
                <td><code>{metrics['f1_mean']:.4f}</code></td>
                <td><code>{metrics['recall_mean']:.4f}</code></td>
                <td><code>{metrics['accuracy_mean']:.4f}</code></td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Churn Analytics & Retention Intelligence Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0b1120;
            --bg-secondary: #131d35;
            --bg-card: rgba(26, 38, 68, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(66, 153, 225, 0.3);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background: var(--bg-primary);
            color: var(--text-main);
            font-family: var(--font-main);
            line-height: 1.6;
            padding: 40px 24px;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.10) 0px, transparent 50%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1320px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 36px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 20px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
        }}

        .badge-live {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            background: #34d399;
            border-radius: 50%;
            box-shadow: 0 0 8px #34d399;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.3); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.8; }}
        }}

        h1 {{
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 8px;
        }}

        .subtitle {{
            color: var(--text-muted);
            font-size: 1.05rem;
            margin-top: 4px;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 36px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-3px);
            border-color: var(--border-hover);
        }}

        .kpi-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #fff;
        }}

        .kpi-change {{
            font-size: 0.82rem;
            margin-top: 6px;
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        .text-emerald {{ color: #34d399; }}
        .text-rose {{ color: #f87171; }}
        .text-cyan {{ color: #38bdf8; }}
        .text-amber {{ color: #fbbf24; }}

        .section-title {{
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .section-title svg {{
            color: var(--accent-cyan);
        }}

        .grid-2col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
            gap: 24px;
            margin-bottom: 36px;
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 26px;
            backdrop-filter: blur(12px);
        }}

        .table-responsive {{
            overflow-x: auto;
            margin-top: 14px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            text-align: left;
        }}

        th {{
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            font-weight: 600;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.5px;
        }}

        td {{
            padding: 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-main);
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        code {{
            font-family: var(--font-mono);
            background: rgba(15, 23, 42, 0.6);
            padding: 3px 8px;
            border-radius: 6px;
            color: #38bdf8;
            font-size: 0.85rem;
            border: 1px solid rgba(56, 189, 248, 0.15);
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .badge-danger {{
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        .badge-warning {{
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}

        .badge-success {{
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .bar-container {{
            width: 140px;
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 99px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #06b6d4);
            border-radius: 99px;
        }}

        .playbook-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            margin-top: 16px;
        }}

        .playbook-card {{
            background: rgba(19, 29, 53, 0.85);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 14px;
            padding: 22px;
        }}

        .playbook-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}

        .playbook-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #fff;
        }}

        .playbook-tag {{
            font-size: 0.72rem;
            font-weight: 700;
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}

        .playbook-body p {{
            font-size: 0.88rem;
            color: #cbd5e1;
            margin-bottom: 10px;
        }}

        .playbook-impact {{
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.25);
            padding: 10px 14px;
            border-radius: 8px;
            color: #34d399;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 14px;
        }}

        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 20px;
            margin-top: 18px;
        }}

        .gallery-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            overflow: hidden;
            transition: transform 0.2s ease;
        }}

        .gallery-card:hover {{
            transform: translateY(-3px);
            border-color: var(--border-hover);
        }}

        .gallery-card img {{
            width: 100%;
            height: 240px;
            object-fit: cover;
            display: block;
            background: #0f172a;
        }}

        .gallery-info {{
            padding: 16px;
        }}

        .gallery-info h4 {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 4px;
        }}

        .gallery-info p {{
            font-size: 0.82rem;
            color: var(--text-muted);
        }}

        footer {{
            margin-top: 60px;
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            padding-top: 24px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div>
                <span class="badge-live">
                    <span class="pulse-dot"></span>
                    Production Analytics Engine v1.0
                </span>
                <h1>Banking Customer Churn & Retention Intelligence</h1>
                <p class="subtitle">End-to-End Predictive Analytics, Risk Quantification & Retention Strategy Optimization</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.85rem; color: var(--text-muted);">Audited Customer Records</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #fff;">{eda_summary['total_records']:,} Accounts</div>
            </div>
        </header>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Baseline Churn Rate</div>
                <div class="kpi-value text-rose">{eda_summary['churn_rate']:.1%}</div>
                <div class="kpi-change text-rose">
                    <span>Portfolio attrition rate</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Champion Model ROC-AUC</div>
                <div class="kpi-value text-cyan">{test_metrics['roc_auc']:.3f}</div>
                <div class="kpi-change text-emerald">
                    <span>▲ Tuned Gradient Boosting</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Annual Value at Risk</div>
                <div class="kpi-value text-amber">${tier_summary['Total_Annual_Value_At_Risk'].sum():,.0f}</div>
                <div class="kpi-change text-amber">
                    <span>${tier_summary['Total_Balance_At_Risk'].sum():,.0f} Total Balances</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Campaign Projected ROI</div>
                <div class="kpi-value text-emerald">{roi_sim['retention_roi_percent']:.0f}%</div>
                <div class="kpi-change text-emerald">
                    <span>+${roi_sim['net_financial_benefit']:,.0f} Net Gain</span>
                </div>
            </div>
        </div>

        <!-- 2 Column Section: Model Evaluation & Feature Importance -->
        <div class="grid-2col">
            <!-- Model Performance -->
            <div class="card">
                <h3 class="section-title">
                    <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16"><path d="M4 11H2v3h2v-3zm5-4H7v7h2V7zm5-5h-2v12h2V2zm-2-1a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1h-2zM7 6a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1H7zM2 10a1 1 0 0 0-1 1v3a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-3a1 1 0 0 0-1-1H2z"/></svg>
                    Predictive Model Benchmark (5-Fold CV)
                </h3>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th>ROC-AUC</th>
                                <th>PR-AUC</th>
                                <th>F1-Score</th>
                                <th>Recall</th>
                                <th>Accuracy</th>
                            </tr>
                        </thead>
                        <tbody>
                            {models_rows_html}
                        </tbody>
                    </table>
                </div>
                <div style="margin-top: 16px; font-size: 0.85rem; color: var(--text-muted);">
                    Optimal Decision Threshold calibrated to <strong>{test_metrics['threshold']:.2f}</strong> to maximize net financial retention margin.
                </div>
            </div>

            <!-- Top Permutation Drivers -->
            <div class="card">
                <h3 class="section-title">
                    <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16"><path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/><path d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z"/></svg>
                    Top Churn Predictors (Feature Importance)
                </h3>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Relative Impact</th>
                                <th>Importance</th>
                                <th>Std Dev</th>
                            </tr>
                        </thead>
                        <tbody>
                            {features_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Risk Tier Segmentation -->
        <div class="card" style="margin-bottom: 36px;">
            <h3 class="section-title">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16"><path d="M7 14s-1 0-1-1 1-4 5-4 5 3 5 4-1 1-1 1H7zm4-6a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/><path fill-rule="evenodd" d="M5.216 14A2.238 2.238 0 0 1 5 13c0-1.355.68-2.75 1.936-3.72A6.325 6.325 0 0 0 5 9c-4 0-5 3-5 4s1 1 1 1h4.216z"/><path d="M4.5 8a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z"/></svg>
                Customer Risk Tiering & Financial Exposure
            </h3>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Risk Tier</th>
                            <th>Customer Volume</th>
                            <th>Avg Churn Risk</th>
                            <th>Total Deposits at Risk</th>
                            <th>Annual Value at Risk</th>
                            <th>Avg Age</th>
                            <th>Inactive Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tier_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Strategic Playbooks -->
        <div class="card" style="margin-bottom: 36px;">
            <h3 class="section-title">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16"><path d="M14.5 3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5v-9a.5.5 0 0 1 .5-.5h13zm-13-1A1.5 1.5 0 0 0 0 3.5v9A1.5 1.5 0 0 0 1.5 14h13a1.5 1.5 0 0 0 1.5-1.5v-9A1.5 1.5 0 0 0 14.5 2h-13z"/><path d="M3 5.5a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9a.5.5 0 0 1-.5-.5zM3 8a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9A.5.5 0 0 1 3 8zm0 2.5a.5.5 0 0 1 .5-.5h6a.5.5 0 0 1 0 1h-6a.5.5 0 0 1-.5-.5z"/></svg>
                Actionable Retention Playbooks & Operational Interventions
            </h3>
            <div class="playbook-grid">
                {playbook_cards_html}
            </div>
        </div>

        <!-- Visual Analytics Gallery -->
        <div class="card">
            <h3 class="section-title">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16"><path d="M6.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z"/><path d="M2.002 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2h-12zm12 1a1 1 0 0 1 1 1v6.5l-3.777-1.947a.5.5 0 0 0-.577.093l-3.71 3.71-2.66-1.772a.5.5 0 0 0-.63.062L1.002 12V3a1 1 0 0 1 1-1h12z"/></svg>
                Analytical Visualizations & Statistical Exhibits
            </h3>
            <div class="gallery-grid">
                <div class="gallery-card">
                    <img src="figures/01_demographic_breakdowns.png" alt="Demographics Churn Drivers">
                    <div class="gallery-info">
                        <h4>Demographic & Behavioral Churn Breakdowns</h4>
                        <p>Geography (Germany 2x rate), Product fragmentation, and Activity status.</p>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="figures/02_financial_distributions.png" alt="Financial Density">
                    <div class="gallery-info">
                        <h4>Financial Density Distributions</h4>
                        <p>Age shift towards 45-60, balance clusters, and credit score density.</p>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="figures/03_correlation_matrix.png" alt="Correlation Matrix">
                    <div class="gallery-info">
                        <h4>Correlation Matrix & Multicollinearity</h4>
                        <p>Pearson and Spearman association matrix for numerical banking attributes.</p>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="figures/04_model_evaluation.png" alt="Model Evaluation">
                    <div class="gallery-info">
                        <h4>ROC & PR Curves Benchmarking</h4>
                        <p>Comparative curves across Gradient Boosting, Random Forest, and Logistic Regression.</p>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="figures/05_feature_importance.png" alt="Feature Importance">
                    <div class="gallery-info">
                        <h4>Permutation Feature Importance</h4>
                        <p>Global rank-ordering of feature contribution to test set ROC-AUC.</p>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="figures/06_risk_tier_analysis.png" alt="Risk Tiers">
                    <div class="gallery-info">
                        <h4>Risk Tiering & Revenue Exposure</h4>
                        <p>High, Medium, and Low risk customer volume vs dollar balance at risk.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer>
            End-to-End Data Analytics Project • Banking Customer Churn & Retention Modeling • Generated with Antigravity
        </footer>
    </div>
</body>
</html>
"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[ReportGenerator] Standalone Executive HTML Dashboard compiled: {self.output_path}")
        return self.output_path
