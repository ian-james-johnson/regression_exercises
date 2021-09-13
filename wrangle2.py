
 
# Functions to obtain Zillow data from the Codeup Data Science Database: zillow
# This function prepares the data for use in exploration in modeling by dropping nulls
#It returns a pandas dataframe.
#--------------------------------

#This function calls acquire.py to obtain Zillow data from SQL database using a SQL query, 
#and prepares the data be removing null values .  

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  MinMaxScaler


#Custom module: acquire.py
import acquire

#FUNCTION to acquire and prepare Zillow data for exploration and modeling
#------------------------------------------------------------------------
def wrangle_zillow():

    #Use acquire.py to pull data from the Zillow database using a SQL query, create a local csv pandas, and return a pandas DataFrame
    df = acquire.get_zillow_data()

    # Drop all rows with any Null values, assign to df, and verify.
    df = df.dropna()

    #Remove outliers
    df = remove_outliers(df, 2.0, ['calculatedfinishedsquarefeet',
       'taxvaluedollarcnt', 'taxamount','bedroomcnt', 'bathroomcnt'] )

    # Rename
    df = df.rename(columns = { 'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'sq_footage',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'yr_built'
        })   

# get distributions of numeric data
    get_hist(df)
   # get_box(df)

# Return the dataframe to the calling code
    return df


# FUNCTION to remove outliers per John Salas
# ------------------------------------------
#   The value for k is a constant that sets the threshold. 
#   Usually, you’ll see k start at 1.5, or 3 or less, depending on how many outliers you want to keep. 
#   The higher the k, the more outliers you keep. Recommend not going beneath 1.5, but this is worth using, 
#   especially with data w/ extreme high/low values

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


#FUNCTION to get min/max from columns in a dataframe
# --------------------------------------------------
# To call this function use:   df.apply(minMax)  (Outside will have to call as:  df.apply(wrangle.minMax))
# x is a list dataframe or 
def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

#FUNCTION to get a dataframe with the original columns and return the train, validate, test with addition of scaled columns
# ---------------------------------------------------------------------
def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    
    # new column names
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    # Fit the scaler on the train
    scaler.fit(train[columns_to_scale])
    
    # transform train validate and test
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test

###################### Prepare Zillow Data With Split ######################

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''

    # splits df into train_validate and test using train_test_split() 
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
    # splits train_validate into train and validate using train_test_split() 
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, validate, test


################### Plotting Functions #####################################
#Plots a normalized value count as a percent using catplotS
def category_percentages_by_another_category_col(df, category_a, category_b):
    """
    Produces a .catplot with a normalized value count
    """
    (df.groupby(category_b)[category_a].value_counts(normalize=True)
    .rename('percent')
    .reset_index()
    .pipe((sns.catplot, 'data'), x=category_a, y='percent', col=category_b, kind='bar', ))


# FUNCTION to plot a scatterplot passing in two variables
# -------------------------------------------------------
def plot_scatter(a, b):

    ax1 = df.plot.scatter(x=a,y=b,c='Navy')        

    return ax1

# FUNCTION to plot histograms of continuous variables
# 
def get_hist(df):
    ''' Gets histograms of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()    

#FUNCTION to plot boxplots of continuous variables
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()


#FUNCTION that takes in a train, validate, test and returns a Standard scaler with the train_scaled, validate_scaled, and test_scaled
def Standard_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs
    """

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled

#FUNCTION that takes in a train, validate, test and returns a Min Max scaler with the train_scaled, validate_scaled, and test_scaled
def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled