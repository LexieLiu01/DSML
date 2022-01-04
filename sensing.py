import os
import numpy as np
import pandas as pd
import pickle

from fancyimpute import SoftImpute, KNN, SimpleFill, IterativeImputer
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

from measures import *
from paras import *



class vk_sensing():
  def __init__(self, method, **kwargs):
    self.clf = None
    self.method = method
    if method == "SoftImpute":
      self.clf = SoftImpute(**kwargs)
    elif method == "KNN":
      self.clf = KNN(**kwargs)
    elif method == "Naive":
      self.clf = SimpleFill()
    elif method == 'II':
      raise ('NOT TESTED')
      self.clf = IterativeImputer(min_value = 0)
    else:
      raise("Not Implemented method")

  def fit_transform(self, X_train):
    # print (X_train, np.isnan(X_train).all())
    assert(self.clf is not None)
    X_est = None
    if np.isnan(X_train).any():
      if np.isnan(X_train).all():
        X_est = np.zeros_like(X_train)
      else:
        # print (np.isnan(self.clf.fit_transform(X_train)).any())
        X_est = massage_imputed_matrix(self.clf.fit_transform(X_train))
    else:
        X_est = X_train
    assert (not np.isnan(X_est).any())
    return X_est

  def CVfit(self,X, val_ratio = 0.2):
    mask = np.invert(np.isnan(X))
    sample_mask = np.random.rand(*X.shape) < val_ratio
    X_train = X.copy()
    X_train[mask & (~sample_mask)] = np.nan
    X_val = X.copy()
    X_val[mask & (sample_mask)] = np.nan
    assert (np.sum(~np.isnan(X_val)) > 0)
    assert (np.sum(~np.isnan(X_train)) > 0)
    # print (np.sum(~np.isnan(X_val)), np.sum(~np.isnan(X_train)))
    cur_best_err = np.inf
    cur_best_k = None
    cur_best_thres = None
    for k in GLOB_IMPUTE_K_SWEEP:
      for thres in GLOB_IMPUTE_thres_SWEEP:
        print(k, thres)
        clf = construct_low_rank_imputer(self.method, k, thres)
        if np.isnan(X_train).any():
          if np.isnan(X_train).all():
            X_est = np.zeros_like(X_train)
          else:
            X_est = massage_imputed_matrix(clf.fit_transform(X_train))
        else:
          X_est = X_train
        err = SMAPE1(X_est, X_val)
      # print (k, err, RMSN(X_est, X_val))
      if err < cur_best_err:
        cur_best_err = err
        cur_best_k = k
        cur_best_thres = thres
    assert(cur_best_k is not None)
    assert(cur_best_thres is not None)
    # if cur_best_k is None:
    #   cur_best_k = 1
    # print (cur_best_k)
    self.clf = construct_low_rank_imputer(self.method, cur_best_k, cur_best_thres)

  # def transform(self, X):
  #   assert(self.clf is not None)
  #   clf.transform(X)


def construct_low_rank_imputer(method, k, thres):
  clf = None
  if method == "SoftImpute":
    clf = SoftImpute(max_rank = k, shrinkage_value = thres, max_iters = 300, verbose = False)
  elif method == "KNN":
    clf = KNN(k = k, verbose = False)
  elif method == 'II':
    clf = IterativeImputer(min_value = 0)
  else:
    raise("Not implemented")
  return clf

def massage_imputed_matrix(X, eps = 1e-3):
  new_X = X.copy()
  for i in range(X.shape[0]):
    tmp = X[i]
    if np.sum(tmp > eps) > 0:
      available = np.nanmean(tmp[tmp > eps])
    else:
      available = 0
    for j in range(X.shape[1]):
      if X[i,j] > eps:
        available = X[i,j]
      else:
        new_X[i,j] = available
  return new_X
