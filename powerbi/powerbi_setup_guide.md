# 📊 Power BI Setup & Implementation Guide: Banking Churn Analytics

This guide provides step-by-step instructions to assemble the 3-page executive Power BI Dashboard in Power BI Desktop using the provided Star Schema datasets and DAX measures.

---

## 1. 🏗️ Star Schema Data Model Architecture

The data model uses a clean **Star Schema** with one central fact table and 4 dimension tables linked via **$1:\text{Many}$ single-direction relationships**:

```
           ┌──────────────────────┐
           │    Dim_Geography     │
           │  (GeographyKey [1])  │
           └──────────┬───────────┘
                      │ 1
                      │
                      │ *
┌─────────────────────┼─────────────────────┐
│  Dim_Demographics   │   Dim_Products      │
│(DemographicKey [1]) │ (ProductKey [1])    │
└──────────┬──────────┴───────────┬─────────┘
           │ 1                    │ 1
           │                      │
           │ *                    │ *
   ┌───────┴──────────────────────┴───────┐
   │         Fact_Customer_Churn          │
   │      - CustomerId (PK)               │
   │      - GeographyKey (FK)             │
   │      - DemographicKey (FK)           │
   │      - ProductKey (FK)               │
   │      - RiskTierKey (FK)              │
   │      - Balances, Scores, ChurnProb   │
   └──────────────────┬───────────────────┘
                      │ *
                      │
                      │ 1
           ┌──────────┴───────────┐
           │    Dim_Risk_Tiers    │
           │  (RiskTierKey [1])   │
           └──────────────────────┘
```

---

## 2. 📥 Loading Data into Power BI Desktop

1. Open **Power BI Desktop**.
2. Click **Get Data** > **Text/CSV**.
3. Load the 5 CSV files located at:
   `C:\Users\asus\.gemini\antigravity-ide\scratch\banking-churn-analytics\powerbi\data\`
   - `Fact_Customer_Churn.csv`
   - `Dim_Geography.csv`
   - `Dim_Demographics.csv`
   - `Dim_Products.csv`
   - `Dim_Risk_Tiers.csv`
4. In the **Model View**, verify the relationships:
   - `Dim_Geography[GeographyKey]` $\rightarrow$ `Fact_Customer_Churn[GeographyKey]` ($1 \to *$)
   - `Dim_Demographics[DemographicKey]` $\rightarrow$ `Fact_Customer_Churn[DemographicKey]` ($1 \to *$)
   - `Dim_Products[ProductKey]` $\rightarrow$ `Fact_Customer_Churn[ProductKey]` ($1 \to *$)
   - `Dim_Risk_Tiers[RiskTierKey]` $\rightarrow$ `Fact_Customer_Churn[RiskTierKey]` ($1 \to *$)
5. Create a new measure table `_Measures` and paste the measures from [`dax_measures.dax`](file:///C:/Users/asus/.gemini/antigravity-ide/scratch/banking-churn-analytics/powerbi/dax_measures.dax).

---

## 3. 📑 Report Page Blueprints

### Page 1: 🌐 Executive Portfolio Overview & Churn Intelligence
* **Target Audience**: Chief Risk Officer (CRO), Head of Retail Banking
* **Canvas Settings**: $16:9$ (1280 x 720 px), Dark Slate Theme (`#0F172A`)

#### Visual Layout:
1. **Top Header & Global Slicers**:
   - **Slicers**: `Dim_Geography[Geography]`, `Dim_Demographics[Gender]`, `Dim_Risk_Tiers[RiskTier]`
2. **KPI Multi-Row Cards (Top Strip)**:
   - Card 1: `[Total Customers]` (Format: `#,##0`)
   - Card 2: `[Churn Rate %]` (Format: `0.0%`, Conditional Formatting: Red if $>20\%$)
   - Card 3: `[Total Balances ($)]` (Format: `$#,##0.0M`)
   - Card 4: `[Total Annual Value at Risk ($)]` (Format: `$#,##0.0M`)
   - Card 5: `[Active Member %]` (Format: `0.0%`)
3. **Left Visual**: **Map / Country Churn Comparison**:
   - Visual: Filled Map or Clustered Column Chart
   - X-Axis: `Dim_Geography[Geography]`
   - Y-Axis: `[Churn Rate %]`, `[Total Customers]`
4. **Center Visual**: **Churn Breakdown by Product Count & Activity**:
   - Visual: 100% Stacked Bar Chart
   - X-Axis: `Dim_Products[NumOfProducts]`
   - Legend: `Fact_Customer_Churn[Exited]`
5. **Right Visual**: **Customer Risk Distribution Donut Chart**:
   - Visual: Donut Chart
   - Category: `Dim_Risk_Tiers[RiskTier]`
   - Values: `[Total Customers]`

---

### Page 2: 👥 Demographic & Behavioral Vulnerability
* **Target Audience**: Customer Experience (CX) & Product Managers

#### Visual Layout:
1. **Visual 1**: **Generational Cohort & Age Breakdown**:
   - Visual: Clustered Column & Line Chart
   - X-Axis: `Dim_Demographics[AgeGroup]` (Sorted by `AgeGroupSort`)
   - Columns: `[Total Customers]`, `[Total Balances ($)]`
   - Line: `[Churn Rate %]`
2. **Visual 2**: **Customer Complaints Impact Matrix**:
   - Visual: Treemap
   - Category: `Fact_Customer_Churn[Complain]` $\rightarrow$ `Dim_Demographics[Gender]`
   - Values: `[Total Customers]`
   - Tooltips: `[Complaint Churn Rate %]`
3. **Visual 3**: **Financial Wealth Tier Scatter Plot**:
   - Visual: Scatter Plot
   - X-Axis: `Fact_Customer_Churn[Balance]`
   - Y-Axis: `Fact_Customer_Churn[EstimatedSalary]`
   - Legend: `Dim_Risk_Tiers[RiskTier]`
   - Size: `Fact_Customer_Churn[ValueAtRisk]`

---

### Page 3: 🎯 Predictive Risk & What-If Retention Simulator
* **Target Audience**: Marketing Operations & Customer Retention Leads

#### Visual Layout:
1. **What-If Parameter Controls**:
   - Numeric Range Slicer 1: `Intervention Cost ($)` ($50 - $300, Step $10, Default $120)
   - Numeric Range Slicer 2: `Assumed Save Rate (%)` (10% - 60%, Step 2%, Default 28%)
2. **Dynamic Financial ROI KPI Cards**:
   - `[Campaign Targeted Customers]` (Count)
   - `[Campaign Total Budget ($)]` ($)
   - `[Campaign Retained Customers Count]` (Count)
   - `[Campaign Net Financial Benefit ($)]` ($)
   - `[Campaign Net ROI %]` (%)
3. **High-Risk Action Table / Drillthrough**:
   - Table Visual: `CustomerId`, `Geography`, `Age`, `Balance`, `ChurnProbability`, `ValueAtRisk`, `RecommendedAction`
   - Sort by: `ValueAtRisk` Descending.
