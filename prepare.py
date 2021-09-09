import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd
from acquire import get_telco_data

def telco_split(df):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=123, 
                                            stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=123,
                                       stratify=train_validate.churn)
    return train, validate, test

def prep_telco(df):
    '''
    This function takes in df that was acquired from get_telco_data.
    Then it drops the SQL foreign keys, which are unnecessary:
    payment_type_id, internet_service_type_id, contract_type_id
    '''
    
    # Get rid of any duplicate records
    df = df.drop_duplicates()
    
     # Create a new column to show if payment was automatic
    df['automatic_pmt'] = np.where(df['payment_type'].str.contains("automatic", case=False), 1, 0)
    
    # Create dummies dataframe
    # .get_dummies(column_names,not dropping any of the dummy columns)
    #dummy_features = ['multiple_lines','online_security','online_backup',
    #                  'device_protection','tech_support','streaming_tv',
    #                  'streaming_movies','contract_type','internet_service_type',
    #                  'payment_type']
    #dummy_df = pd.get_dummies(df, columns=dummy_features, drop_first=False)
    
    # Join original df with dummies df
    # .concat([original_df,dummy_df])
    #df = pd.concat([df, dummy_df])
    
    # Drop original columns that we made dummies of
    #df = df.drop(columns=dummy_features)
    
    # Convert total_charges to a numeric data type (is currently string type)
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    # There are 7 null values in total_charges
    # Replace those 7 null values with median value
    df.total_charges.fillna(df.total_charges.median(), inplace=True)
    
    # Replace string values with numbers
    df.replace({'gender':{'Male':1, 'Female':0}}, inplace=True)
    df.replace({'partner':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'dependents':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'phone_service':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'paperless_billing':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'churn':{'Yes':1, 'No':0}}, inplace=True)
    
    df.replace({'multiple_lines':{'Yes':1, 'No':0, 'No phone service':-1}}, inplace=True)
    df.replace({'device_protection':{'Yes':1, 'No':0, 'No internet service':-1}}, inplace=True)
    df.replace({'online_security':{'Yes':1, 'No':0, 'No internet service':-1}}, inplace=True)
    df.replace({'online_backup':{'Yes':1, 'No':0, 'No internet service':-1}}, inplace=True)
    df.replace({'tech_support':{'Yes':1, 'No':0, 'No internet service':-1}}, inplace=True)
    df.replace({'streaming_tv':{'Yes':1, 'No':0, 'No internet service':-1}}, inplace=True)
    df.replace({'streaming_movies':{'Yes':1, 'No':0, 'No internet service':-1}}, inplace=True)
    df.replace({'contract_type':{'Month-to-month':0, 'One year':1, 'Two year':2}}, inplace=True)
    df.replace({'internet_service_type':{'None':0, 'DSL':1, 'Fiber optic':2}}, inplace=True)
    df.replace({'payment_type':{'Mailed check':0, 'Electronic check':1, 'Bank transfer (automatic)':2, 'Credit card (automatic)':3}}, inplace=True)
    
    # Create new feature to combine online utility options
    # These online options did not seem popular during univariate exploration
    #df['online_utility']= df.online_security + df.online_backup + df.tech_support
    
    # Create new feature to combine streaming options, because they are very similar
    #df['streaming']= df.streaming_tv + df.streaming_movies
    
    # Drop the unnecessary colums
    df = df.drop(columns=['customer_id', 'payment_type_id', 'internet_service_type_id', 'contract_type_id'])    
    
    # Split the dataset into train, validate, and test subsets
    train, validate, test = telco_split(df)
    
    return train, validate, test



def scale_telco(train, validate, test):
    # Create the object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit the object
    # Scalers should only be fit to train to prevent data leakage
    scaler.fit(train)

    # Use the object
    # the same object that was fitted to train can be used on validate and test
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    # For some reason, the minmax scaler converted the dataframes into series
    # and renamed the columns 0-6

    # The following lines convert those series back to dataframes 
    # and restore their column names

    train_scaled = pd.DataFrame(train_scaled)
    train_scaled = train_scaled.rename(columns={0:'bedrooms', 1:'bathrooms', 2:'area_sqft', 3:'tax_value', 4:'year_built',5:'tax_amount', 6:'fips'})

    validate_scaled = pd.DataFrame(validate_scaled)
    validate_scaled = validate_scaled.rename(columns={0:'bedrooms', 1:'bathrooms', 2:'area_sqft', 3:'tax_value', 4:'year_built',5:'tax_amount', 6:'fips'})

    test_scaled = pd.DataFrame(test_scaled)
    test_scaled = test_scaled.rename(columns={0:'bedrooms', 1:'bathrooms', 2:'area_sqft', 3:'tax_value', 4:'year_built',5:'tax_amount', 6:'fips'})

    return train_scaled, validate_scaled, test_scaled
