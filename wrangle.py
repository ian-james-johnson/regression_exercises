# this function be used to access the SQL server
# the user, host, and password will come from importing 'env'

import numpy as np
import pandas as pd
import os
from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    '''
    This function creates a connection to the Codeup db.
    It takes db argument as a string name.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function gets new telco data from the Codeup database.
    '''
    sql_query = """
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                JOIN propertylandusetype USING (propertylandusetypeid)
                WHERE propertylandusedesc = 'Single Family Residential';
                """
    # Read in dataframe from Codeup
    df = pd.read_sql(sql_query,get_connection('zillow'))
    return df

def get_zillow_data():
        '''
        This function gets telco data from csv, or otherwise from Codeup database.
        '''
        if os.path.isfile('zillow.csv'):
            df = pd.read_csv('zillow.csv', index_col = 0)
        else:
            df = new_zillow_data()
            df.to_csv('zillow.csv')
        return df