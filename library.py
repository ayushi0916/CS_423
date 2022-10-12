import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column  #column to focus on
    self.dummy_na = dummy_na
    self.drop_first = drop_first
  
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X
    
  def transform(self, X):
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?
    X_ = X.copy()
    return pd.get_dummies(X_,prefix= self.target_column,
                          prefix_sep='_',
                          columns=[self.target_column],
                          dummy_na = self.dummy_na,
                          drop_first = self.drop_first)
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    self.column_list = column_list
    self.action = action

  def transform(self, X):
    col_list_rem = set(self.column_list)-set(X.columns.to_list())
    X_ = X.copy()

    #Assertion for the action keep and keeping the columns
    if self.action == 'keep':
      assert col_list_rem==set(), f'{self.__class__.__name__} does not contain the column "{col_list_rem}" to be kept.'
      return X_[self.column_list]
    
    #Warning for the action drop and dropping the columns
    elif self.action=='drop':
      if col_list_rem!=set():
        print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {col_list_rem}.\n")
        return X_.drop(columns=self.column_list,errors='ignore')
      elif col_list_rem==set():
        return X_.drop(columns=self.column_list,errors='ignore')

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self,target_column):
    self.target_column = target_column  #column to focus on
  
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X
    
  def transform(self, X):
    X_ = X.copy()
    assert isinstance(X_, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X_)} instead.'
    assert self.target_column in X_.columns.to_list(), f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X_[self.target_column].to_list()])

    #your code below
    sigma = X_[self.target_column].std()
    mu = X_[self.target_column].mean()
    X_[self.target_column]=(X_[self.target_column].clip(lower=mu-3*sigma, upper=mu+3*sigma))
    return X_
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)

    return result
  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,target_column,fence):
    self.target_column = target_column  #column to focus on
    self.fence = fence
  
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X
    
  def transform(self, X):
    X_ = X.copy()
    if self.fence == 'inner':
      q1 = X_[self.target_column].quantile(0.25)
      q3 = X_[self.target_column].quantile(0.75)
      iqr = q3-q1
      inner_low = q1-1.5*iqr
      inner_high = q3+1.5*iqr
      X_[self.target_column]=(X_[self.target_column].clip(lower=inner_low, upper=inner_high))
      return X_

    elif self.fence == 'outer':
      q1 = X_[self.target_column].quantile(0.25)
      q3 = X_[self.target_column].quantile(0.75)
      iqr = q3-q1
      outer_low = q1-3*iqr
      outer_high = q3+3*iqr
      X_[self.target_column]=(X_[self.target_column].clip(lower=outer_low, upper=outer_high))
      return X_

    
  def fit_transform(self, X, y = None):
    result = self.transform(X)

    return result
