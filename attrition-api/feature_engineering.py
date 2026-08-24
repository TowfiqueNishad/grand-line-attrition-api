import pandas as pd

def add_hr_features(X):
    X = X.copy()
    X["YearsSincePromotionRatio"] = X["YearsSinceLastPromotion"] / (X["YearsAtCompany"] + 1)
    X["CurrentRoleRatio"] = X["YearsInCurrentRole"] / (X["YearsAtCompany"] + 1)
    X["ManagerTenureRatio"] = X["YearsWithCurrManager"] / (X["YearsAtCompany"] + 1)
    X["CompanyExperienceRatio"] = X["YearsAtCompany"] / (X["TotalWorkingYears"] + 1)
    X["IncomePerJobLevel"] = X["MonthlyIncome"] / (X["JobLevel"] + 1)
    X["IncomePerYearExperience"] = X["MonthlyIncome"] / (X["TotalWorkingYears"] + 1)
    X["YearsAtCompanyRatioToAge"] = X["YearsAtCompany"] / (X["Age"] + 1)
    X["JobLevelPerYear"] = X["JobLevel"] / (X["TotalWorkingYears"] + 1)

    satisfaction_cols = ["EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction",
                          "RelationshipSatisfaction", "WorkLifeBalance"]
    X["SatisfactionAvg"] = X[satisfaction_cols].mean(axis=1)
    X["WorkPressureScore"] = (
        X["OverTime"].map({"Yes": 1, "No": 0})
        + (4 - X["WorkLifeBalance"]) / 4
        + (4 - X["JobSatisfaction"]) / 4
    )
    X["EarlyCareer"] = ((X["Age"] < 30) & (X["TotalWorkingYears"] < 5)).astype(int)
    X["FrequentJobChanges"] = ((X["NumCompaniesWorked"] >= 3) & (X["TotalWorkingYears"] <= 10)).astype(int)
    X["OverTime_JobSatisfaction"] = X["OverTime"].astype(str) + "_" + X["JobSatisfaction"].astype(str)
    X["OverTime_WorkLifeBalance"] = X["OverTime"].astype(str) + "_" + X["WorkLifeBalance"].astype(str)
    X["OverTime_JobLevel"] = X["OverTime"].astype(str) + "_" + X["JobLevel"].astype(str)
    X["Marital_OverTime"] = X["MaritalStatus"].astype(str) + "_" + X["OverTime"].astype(str)
    X["JobRole_OverTime"] = X["JobRole"].astype(str) + "_" + X["OverTime"].astype(str)
    return X
