def rule_based_check(row):
    issues = []

    if not (0 <= row["age"] <= 120):
        issues.append("Age outside physiological limits")

    if not (80 <= row["resting_bp"] <= 200):
        issues.append("Blood pressure outside safe range")

    if not (100 <= row["cholesterol"] <= 350):
        issues.append("Cholesterol outside safe range")

    if not (60 <= row["max_heart_rate"] <= 220):
        issues.append("Heart rate outside physiological limits")

    if not (0 <= row["oldpeak"] <= 6):
        issues.append("Oldpeak value abnormal")

    return ("INVALID", issues) if issues else ("VALID", [])
