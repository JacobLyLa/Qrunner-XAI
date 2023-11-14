import warnings

import torch
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, r2_score

warnings.filterwarnings('ignore', category=ConvergenceWarning)

def train_binary(train_acts, train_values, test_acts, test_values, max_iter, k_fold, num_reg):
    reg = linear_model.LogisticRegressionCV(max_iter=max_iter, cv=k_fold, Cs=num_reg)
    reg.fit(train_acts, train_values)
    pred = reg.predict(test_acts)
    score = accuracy_score(test_values, pred)
    return reg, 2*score-1

def train_non_binary(train_acts, train_values, test_acts, test_values, max_iter, k_fold, num_reg):
    reg = linear_model.LassoCV(max_iter=max_iter, cv=k_fold, n_alphas=num_reg)
    reg.fit(train_acts, train_values)
    pred = reg.predict(test_acts)
    score = r2_score(test_values, pred)
    return reg, score

def train_probe(model, concept, layer, max_iter, k_fold, num_reg):
    _, train_acts_dict = model(torch.tensor(concept.train_obs), return_acts=True)
    _, test_acts_dict = model(torch.tensor(concept.test_obs), return_acts=True)
    train_values = concept.train_values
    test_values = concept.test_values

    train_acts = train_acts_dict[str(layer)].detach().numpy()
    test_acts = test_acts_dict[str(layer)].detach().numpy()
    train_acts = train_acts.reshape(len(train_acts), -1)
    test_acts = test_acts.reshape(len(test_acts), -1)

    if concept.binary:
        probe, score = train_binary(train_acts, train_values, test_acts, test_values, max_iter, k_fold, num_reg)
    else:
        probe, score = train_non_binary(train_acts, train_values, test_acts, test_values, max_iter, k_fold, num_reg)
    return probe, score