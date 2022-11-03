import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
  
class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    self.threshold = threshold  #column to focus on
   
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X
    
  def transform(self, X):
    #assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?
    X_ = X.copy()
    df_corr = X_.corr(method='pearson')
    masked_df = df_corr.abs() > self.threshold
    upper_mask = np.triu(masked_df,k=True)
    correlated_columns = [masked_df.columns[j] for i,j in enumerate((np.any(upper_mask == True, axis=0).nonzero()[len(np.any(upper_mask == True, axis=0).nonzero())-1]))]
    new_df = X_.drop(columns=correlated_columns)

    return new_df
  
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

class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass  #takes no 
      #fill in rest below
  
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return X 
    
  def transform(self, X):
    from sklearn.preprocessing import MinMaxScaler
    X_ = X.copy()
    transformer_scaler = MinMaxScaler()
    transformer_result = transformer_scaler.fit_transform(X_)
    new_dataframe = pd.DataFrame(transformer_result,columns=[columns for columns in X_.columns])
    return new_dataframe
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform"):
    #your code
    self.n_neighbors = n_neighbors
    self.weights = weights
  
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return X 
    
  def transform(self, X):
    from sklearn.impute import KNNImputer
    X_ = X.copy()
    imputer = KNNImputer(n_neighbors= self.n_neighbors, weights=self.weights) 
    imputed_data = imputer.fit_transform(X_) 
    new_impdataframe = pd.DataFrame(imputed_data,columns=[column for column in X_.columns])
    return new_impdataframe

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier()
  var = []
  for i in np.arange(1, n):
  
    train_X , test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                    random_state=i, stratify=labels)
    model = KNeighborsClassifier()
    model.fit(train_X, train_y)  #train model
    train_pred = model.predict(train_X)           #predict against training set
    test_pred = model.predict(test_X)             #predict against test set
    train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
    test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
    f1_ratio = test_f1/train_f1          #take the ratio
    var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value
  idx = np.array(abs(var - rs_value)).argmin()
  return idx


titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)


customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)


def dataset_setup(full_table, label_column_name:str, the_transformer, rs, ts=.2):
  #your code below
  titanic_features = full_table.drop(columns=label_column_name)
  labels = full_table[label_column_name].to_list()
  X_train, X_test, y_train, y_test = train_test_split(titanic_features, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  X_train_transformed = titanic_transformer.fit_transform(X_train)
  X_test_transformed = titanic_transformer.fit_transform(X_test)
  x_trained_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy

def titanic_setup(titanic_table, transformer=titanic_transformer, rs=40, ts=.2):
    x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(titanic_table, 'Survived',
                                                                           titanic_transformer,
                                                                           rs,ts)

    return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy

def customer_setup(customer_table, transformer=customer_transformer, rs=76, ts=.2):
    x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(customer_table, 'Rating',
                                                                           customer_transformer,
                                                                           rs,ts)

    return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy
