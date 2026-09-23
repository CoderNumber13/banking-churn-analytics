/* ==============================================================================
   POWER QUERY (M) ETL TRANSFORMATION SCRIPTS
   Copy and paste these queries into Power BI Desktop > Advanced Editor
   ============================================================================== */

// ------------------------------------------------------------------------------
// QUERY 1: Dim_Geography
// ------------------------------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\Users\asus\.gemini\antigravity-ide\scratch\banking-churn-analytics\powerbi\data\Dim_Geography.csv"), [Delimiter=",", Columns=6, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{
        {"GeographyKey", Int64.Type},
        {"Geography", type text},
        {"CountryCode", type text},
        {"Region", type text},
        {"Latitude", type number},
        {"Longitude", type number}
    })
in
    #"Changed Type"


// ------------------------------------------------------------------------------
// QUERY 2: Dim_Demographics
// ------------------------------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\Users\asus\.gemini\antigravity-ide\scratch\banking-churn-analytics\powerbi\data\Dim_Demographics.csv"), [Delimiter=",", Columns=6, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{
        {"DemographicKey", Int64.Type},
        {"Gender", type text},
        {"AgeGroup", type text},
        {"GenerationalCohort", type text},
        {"CreditScoreTier", type text},
        {"AgeGroupSort", Int64.Type}
    })
in
    #"Changed Type"


// ------------------------------------------------------------------------------
// QUERY 3: Dim_Products
// ------------------------------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\Users\asus\.gemini\antigravity-ide\scratch\banking-churn-analytics\powerbi\data\Dim_Products.csv"), [Delimiter=",", Columns=6, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{
        {"ProductKey", Int64.Type},
        {"NumOfProducts", Int64.Type},
        {"CardType", type text},
        {"HasCreditCard", type text},
        {"HasCreditCardCode", Int64.Type},
        {"ProductBundleTier", type text}
    })
in
    #"Changed Type"


// ------------------------------------------------------------------------------
// QUERY 4: Dim_Risk_Tiers
// ------------------------------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\Users\asus\.gemini\antigravity-ide\scratch\banking-churn-analytics\powerbi\data\Dim_Risk_Tiers.csv"), [Delimiter=",", Columns=6, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{
        {"RiskTierKey", Int64.Type},
        {"RiskTier", type text},
        {"ProbabilityThreshold", type text},
        {"SortOrder", Int64.Type},
        {"ColorHex", type text},
        {"RecommendedAction", type text}
    })
in
    #"Changed Type"


// ------------------------------------------------------------------------------
// QUERY 5: Fact_Customer_Churn
// ------------------------------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\Users\asus\.gemini\antigravity-ide\scratch\banking-churn-analytics\powerbi\data\Fact_Customer_Churn.csv"), [Delimiter=",", Columns=21, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{
        {"CustomerId", Int64.Type},
        {"GeographyKey", Int64.Type},
        {"DemographicKey", Int64.Type},
        {"ProductKey", Int64.Type},
        {"RiskTierKey", Int64.Type},
        {"Age", Int64.Type},
        {"CreditScore", Int64.Type},
        {"Tenure", Int64.Type},
        {"Balance", type number},
        {"EstimatedSalary", type number},
        {"SatisfactionScore", Int64.Type},
        {"PointEarned", Int64.Type},
        {"IsActiveMember", Int64.Type},
        {"Complain", Int64.Type},
        {"Exited", Int64.Type},
        {"ChurnProbability", type number},
        {"EstimatedAnnualValue", type number},
        {"ValueAtRisk", type number},
        {"BalanceSalaryRatio", type number},
        {"TenureAgeRatio", type number},
        {"EngagementScore", type number}
    })
in
    #"Changed Type"
