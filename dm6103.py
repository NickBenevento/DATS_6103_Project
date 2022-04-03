#%%
import mysql.connector
from mysql.connector import Error
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt

#%%
def dbCon_dsLand(tbname, ind_col_name = ""):
  """ 
  connect to datasci.land database for this class and pull all rows from table
  :param str tbname: table name that exist on the server 
  :param str ind_col_name: optional, name of index column 
  :return: pandas.Dataframe
  """

  df = None # set a global variable to store the dataframe
  hostname = 'pysql.datasci.land'
  dbname = 'datascilanddb0'
  username = '6103_sp22'
  pwd = 'v8rX91jb7s'
  query = 'SELECT * FROM `'+ dbname +'`.`'+ tbname + '`'

  try:
    connection = mysql.connector.connect(host=hostname, database=dbname, user=username, password=pwd)
    if connection.is_connected():
      # optional output
      db_Info = connection.get_server_info()
      print(f'Connected to MySQL Server version {db_Info}')
      cursor = connection.cursor()
      cursor.execute("select database();")
      record = cursor.fetchone()
      print(f"You're connected to database: {record}")
      # read query into dataframe df
      df = pd.read_sql(query, connection, index_col= ind_col_name) if (ind_col_name) else pd.read_sql(query, connection) # tables often have unique Id field
      print(f'Dataframe is loaded.')
      cursor.close()
      connection.close()
      print("MySQL connection is closed")

  except Error as e:
    print(f'Error while connecting to MySQL {e}')
      
  return df

# print("\nFunction dbCon_dsLand loaded. Ready to continue.")

#%%
def api_dsLand(tbname, ind_col_name = ""):
  """ 
  call to api endpoint on datasci.land database to access datasets
  :param str tbname: table name that exist on the server 
  :param str ind_col_name: optional, name of index column 
  :return: pandas.Dataframe
  """
  
  df = None # set a global variable to store the dataframe
  apikey = 'K35wHcKuwXuhHTaz7zY42rCje'
  parameters = {"apikey": apikey, 'table': tbname}
  js = {'error': 'Initialize' }

  try:
    response = requests.get("https://api.datasci.land/endpt.json", params=parameters)
    js = response.json()
  except Error as e:
    print(f'Error while connecting to API {e}. Please contact the administrator.')

  if ('error' in js) : 
    print(f'Error: {js["error"]} Please contact the administrator.') # The json object will have a key named "error" if not successful
    return df
  
  # json object seems okay at this point
  try: df = pd.DataFrame(js) 
  except ValueError: print(f'Value Error while converting json into dataframe. Please contact the administrator.')
  except Error as e: print(f'Error while converting json into dataframe. {e}. Please contact the administrator.')
  
  # df seems load okay at this point. Default values is object/string everywhere.
  # try to convert all possible ones to numeric
  for col in df.columns:
    try: df[col]=pd.to_numeric(df[col])
    except ValueError: pass
    except: pass

  # set index if given
  # if (ind_col_name and ind_col_name in df): df.set_index(ind_col_name, inplace=True)  # if given col_name exist, make it the index.
  try: df.set_index(ind_col_name, inplace=True)
  except ValueError: pass
  except TypeError: pass
  except: pass
  
  print(f'Dataframe from DSLand API is loaded.')
  return df

# print("\nFunction api_dsLand loaded. Ready to continue.")

#%%
# Standard quick checks
def dfChk(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  # print(f'\n{cnt}: dtypes: ')
  # cnt+=1
  # print(dframe.dtypes)

  # try:
  #   print(f'\n{cnt}: columns: ')
  #   cnt+=1
  #   print(dframe.columns)
  # except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

# examples:
# dfChk(df, True)

#%%
"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""
print(__doc__)

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#%%


def plot_classifier(X, y, clf, ax=None, ticks=False, proba=False, lims=None): # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False

    # can abstract some of this into a higher-level function for learners to call
    cs = plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8, proba=proba)
    if proba:
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('probability of red $\Delta$ class', fontsize=20, rotation=270, labelpad=30)
        cbar.ax.tick_params(labelsize=14)
    #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k', linewidth=1)
    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(X0[y==labels[0]], X1[y==labels[0]], cmap=plt.cm.coolwarm, s=60, c='b', marker='o', edgecolors='k')
        ax.scatter(X0[y==labels[1]], X1[y==labels[1]], cmap=plt.cm.coolwarm, s=60, c='r', marker='^', edgecolors='k')
    else:
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k', linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel(data.feature_names[0])
#     ax.set_ylabel(data.feature_names[1])
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())
#     ax.set_title(title)
    if show:
        plt.show()
    else:
        return ax

def plot_4_classifiers(X, y, clfs):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for clf, ax, title in zip(clfs, sub.flatten(), ("(1)","(2)","(3)","(4)")):
        # clf.fit(X, y)
        plot_classifier(X, y, clf, ax, ticks=True)
        ax.set_title(title)
    plt.show()

def plot_classifiers(X, y, clfs):
    titles = []
    for model in clfs:
      titles.append(model.__class__.__name__)

    # Set-up nx2 grid for plotting.
    nrows = int(len(clfs)/2) # assume len is even for now
    fig, sub = plt.subplots(nrows, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for clf, ax, title in zip(clfs, sub.flatten(), titles):
        # clf.fit(X, y)
        plot_classifier(X, y, clf, ax, ticks=True)
        ax.set_title(title)
    plt.show()
    
#%%