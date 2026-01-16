DL_THRESHOLD = 0.8
RISK_RULE, RISK_DL, RISK_ML = 0.6, 0.3, 0.2

def compute_verdict(rule_v, ml_v, dl_score):
    dl_v = "NORMAL" if dl_score < DL_THRESHOLD else "ANOMALOUS"

    risk = (
        (RISK_RULE if rule_v == "INVALID" else 0) +
        (RISK_DL if dl_v == "ANOMALOUS" else 0) +
        (RISK_ML if ml_v == "ANOMALOUS" else 0)
    )

    if risk <= 0.2:
        return "✅ PLAUSIBLE – SAFE TO USE", "Data is physiologically reliable."
    elif risk <= 0.4:
        return "⚠ PLAUSIBLE WITH CAUTION", "Borderline physiological patterns detected."
    elif risk <= 0.6:
        return "⚠ REVIEW REQUIRED", "Multiple inconsistencies detected."
    else:
        return "❌ IMPLAUSIBLE – DO NOT USE", "Physiological safety violated."
