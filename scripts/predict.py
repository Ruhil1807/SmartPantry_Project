def predict_spoilage(expires_in_days):
    if expires_in_days <= 0:
        return "Expired"
    elif expires_in_days <= 2:
        return "Expiring Soon"
    else:
        return "Fresh"
