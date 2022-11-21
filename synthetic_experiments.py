import dcor


# dcorr
def get_dcorr(model_preds, cf):
    # dcorr between learned features and protected variable
    assert len(model_preds.shape) == 1
    assert len(cf.shape) == 1
    return dcor.distance_correlation_sqr(model_preds.cpu(), cf.cpu())
