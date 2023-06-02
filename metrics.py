
from sklearn.metrics import roc_auc_score

class AUC:
    """
    Computes area under ROC curve
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, target, pred): return roc_auc_score(target, pred)
    
def get_evaluation_metric(config):return AUC()