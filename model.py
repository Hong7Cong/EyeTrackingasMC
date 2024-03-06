from sklearn.base import BaseEstimator, ClassifierMixin

def get_train_test_data_3state(ctmc_cvi_models, ctmc_ctrl_models, train_ids):
    featureofinterest = [1,2,4,6,8,9]#[0,1,2,3,5,6,7,8,9,10,11,12]
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for k in ctmc_cvi_models.keys():
        if(k.split('_')[0] in train_ids):
            X_train.append(ctmc_cvi_models[k].matrix.flatten()[featureofinterest].tolist())
            y_train.append(1)
        else:
            X_test.append(ctmc_cvi_models[k].matrix.flatten()[featureofinterest].tolist())
            y_test.append(1)

    for k in ctmc_ctrl_models.keys():
        if(k.split('_')[0] in train_ids):
            X_train.append(ctmc_ctrl_models[k].matrix.flatten()[featureofinterest].tolist())
            y_train.append(0)
        else:
            X_test.append(ctmc_ctrl_models[k].matrix.flatten()[featureofinterest].tolist())
            y_test.append(0)
    return X_train, y_train, X_test, y_test

class dummiesclf(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass
    
    def predict(self, X, y=None):
        ret = np.ones(len(X))
        return ret
    
    def predict_proba(self, X, y=None):
        ret = np.stack([np.zeros(len(X)), np.ones(len(X))], axis=1)
        return ret
    
class meanclf(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass
    
    def predict(self, X, y=None):
        ret = np.ones(len(X))
        return ret
    
    def predict_proba(self, X, y=None):
        ret = np.stack([np.zeros(len(X)), np.ones(len(X))], axis=1)
        return ret