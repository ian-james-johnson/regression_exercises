import numpy as np
import pandas as pd
import os
from env import host, user, password


# this function be used to access the SQL server
# the user, host, and password will come from importing 'env'

def get_connection(db, user=user, host=host, password=password):
    '''
    This function creates a connection to the Codeup db.
    It takes db argument as a string name.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_telco_data():
    '''
    This function gets new telco data from the Codeup database.
    '''
    sql_query = """
                SELECT *
                FROM customers
                JOIN contract_types USING(contract_type_id)
                JOIN internet_service_types USING(internet_service_type_id)
                JOIN payment_types USING(payment_type_id);
                """
    # Read in dataframe from Codeup
    df = pd.read_sql(sql_query,get_connection('telco_churn'))
    return df

def get_telco_data():
        '''
        This function gets telco data from csv, or otherwise from Codeup database.
        '''
        if os.path.isfile('telco.csv'):
            df = pd.read_csv('telco.csv', index_col = 0)
        else:
            df = new_telco_data()
            df.to_csv('telco.csv')
        return df

