import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import datetime

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import seaborn as sns

from os import listdir
from os.path import isfile, join

import requests

def prosper_auth():
    client_id = 'XXX'
    client_secret = 'XXX'
    user = 'XXX'
    password = 'XXX'

    auth_url = "https://api.prosper.com/v1/security/oauth/token"
    auth_payload = "grant_type=password&client_id={}&client_secret={}&username={}&password={}".format(client_id, client_secret, user, password)
    auth_headers = { 'accept': "application/json", 'content-type': "application/x-www-form-urlencoded" }
    auth_response = requests.request("POST", auth_url, data=auth_payload, headers=auth_headers)
    json_auth_response = auth_response.json()

    return json_auth_response

def get_historical_listings(refresh = False):
    listing_pkl = 'historical_listings.pkl'
    listing_path = 'lend2 data/historical_listings'
    listing_files = [f for f in listdir(listing_path) if isfile(join(listing_path, f))]

    if listing_pkl not in listing_files or refresh == True:
        results = []
        for i in listing_files:
            print(i)
            df = pd.read_csv('lend2 data/historical_listings/{}'.format(i), engine='python', encoding='iso-8859-1') # https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas
            results.append(df)

        print('Concat ready (historical listings)..')
        listing_df = pd.concat(results)
        listing_df.to_pickle('lend2 data/historical_listings/{}'.format(listing_pkl))

    elif  listing_pkl in listing_files:
        print('Loading listing pkl..')
        listing_df = pd.read_pickle('lend2 data/historical_listings/{}'.format(listing_pkl))

    # make new columns which will match those of the loan dataset
    listing_df['origination_date'] = listing_df['loan_origination_date']
    listing_df['co_borrower_application'] = listing_df['CoBorrowerApplication']
    listing_df['amount_borrowed'] = listing_df['amount_funded']
    
    # clean df
    remove_cols = list(pd.read_csv('lend2 data/references/remove_cols.csv').columns)
    listing_df['origination_date'] = pd.to_datetime(listing_df['origination_date'], errors='coerce').dt.strftime('%m/%d/%Y')
    listing_df = listing_df[listing_df['origination_date'].notna()]
    listing_df = listing_df.drop(columns = ['loan_origination_date', 'CoBorrowerApplication'] + remove_cols)

    return listing_df

def get_historical_loans(refresh = False):
    loan_pkl = 'loan_df_cleaned.pkl'
    loan_path = 'lend2 data/historical_loans'
    loan_files = [f for f in listdir(loan_path) if isfile(join(loan_path, f))]

    if loan_pkl not in loan_files or refresh == True:
        results = []
        for i in loan_files:
            print(i)
            df = pd.read_csv('lend2 data/historical_loans/{}'.format(i), engine='python', encoding='iso-8859-1') # https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas
            results.append(df)

        print('Concat ready..')
        loan_df = pd.concat(results)
        loan_df.to_pickle('lend2 data/historical_loans/{}'.format(loan_pkl))

    elif  loan_pkl in loan_files:
        print('Loading loan pkl..')
        loan_df = pd.read_pickle('lend2 data/historical_loans/{}'.format(loan_pkl))
    
    loan_df['origination_date'] = pd.to_datetime(loan_df['origination_date'], errors='coerce').dt.strftime('%m/%d/%Y')

    return loan_df

def get_historicals():
    listing_df = get_historical_listings()
    loan_df = get_historical_loans()

    listing_df['classifier'] = (listing_df['borrower_rate'].astype(str) + listing_df['co_borrower_application'].astype(str) + 
                                listing_df['origination_date'].astype(str) + listing_df['prosper_rating'].astype(str) + listing_df['amount_borrowed'].astype(int).astype(str)).astype(str)
    
    loan_df['classifier'] = (loan_df['borrower_rate'].astype(str) + loan_df['co_borrower_application'].astype(str) + 
                            loan_df['origination_date'].astype(str) + loan_df['prosper_rating'].astype(str) + loan_df['amount_borrowed'].astype(int).astype(str)).astype(str)
    
    loan_df = loan_df.drop(columns=['borrower_rate', 'co_borrower_application', 'origination_date', 'prosper_rating', 'amount_borrowed'])
    
    # cols_to_use = list(loan_df.columns.difference(listing_df.columns)) + ['classifier']

    final_df = pd.merge(loan_df, listing_df, on='classifier', how='left')
    final_df = final_df.drop_duplicates(subset=['classifier'], keep=False) # if any duplicates, get rid of all the data
    final_df['loan_default_reason_description'] = final_df['loan_default_reason_description'].fillna('Completed')
    final_df=final_df.dropna(subset=['listing_number'])

    return final_df

def get_active_listings(json_auth_response, refresh = True):
    active_csv = datetime.datetime.today().strftime('%m-%d-%Y') + '.csv'
    active_path = 'lend2 data/active_listings'
    active_files = [f for f in listdir(active_path) if isfile(join(active_path, f))]

    if active_csv not in active_files or refresh == True:

        res_length = 1
        offset = 0
        offset_interval = 25

        max_pages = 10
        max_offset = 25*10

        results = []

        while res_length > 0:
            access_token = json_auth_response['access_token']
            print(access_token)
            
            base_url = 'https://api.prosper.com/listingsvc/v2/listings/?offset={}&limit=25'.format(str(offset))
            base_headers = {'Authorization': 'bearer {}'.format(access_token)}
            base_result = requests.request("GET", base_url, headers=base_headers)
            # print(base_result)


            json_listings = base_result.json()
            res_length = len(json_listings['result'])
            # print(res_length)

            # print(json_listings['result'])

            for i in json_listings['result']:
                for s in i['credit_bureau_values_transunion_indexed']:
                    i[s] = i['credit_bureau_values_transunion_indexed'][s]
                del i['credit_bureau_values_transunion_indexed']

                print(i)
                print('###############')
                results.append(i)

            offset += offset_interval
            print('Loading next page..')

        listings = pd.DataFrame(results)
        listings.to_csv('lend2 data/active_listings/{}'.format(active_csv))
    
    elif active_csv in active_files:
        listings = pd.read_csv('lend2 data/active_listings/{}'.format(active_csv))

    listings['bankcard_utilization'] = listings['bc34s_bankcard_utilization']/100
    listings = listings.fillna(-999999)
    
    return listings

def factorize(dataframe):
    # df[['prosper_rating', 'fico_score', 'employment_status_description', 'borrower_state']] = df[['prosper_rating', 'fico_score', 'employment_status_description', 'borrower_state']].apply(lambda x: pd.factorize(x)[0])
    switch_cols = ['prosper_rating', 'fico_score', 'employment_status_description', 'borrower_state']
    
    for i in switch_cols:
        ref = pd.read_csv('lend2 data/references/{}.csv'.format(i)).to_dict('records')
        for r in ref:
            dataframe.loc[dataframe[i] == r['text'], i] = r['numeric']

    return dataframe

def model(df, active):
    
    if len(active) < 25:
        raise InterruptedError('Less than 25 active listings currently available. Please try again later.')
    

    df = df[['listing_number', 'credit_pull_date', 'listing_status', 'listing_amount', 'amount_funded', 'funding_threshold', 'prosper_rating',
             'estimated_return', 'estimated_loss_rate', 'listing_term', 'scorex', 'fico_score', 'prosper_score', 'listing_category_id', 'borrower_apr',
             'stated_monthly_income', 'income_verifiable', 'employment_status_description', 'months_employed', 'borrower_state',
             'prior_prosper_loans_principal_borrowed', 'prior_prosper_loans_late_cycles', 'prior_prosper_loan_earliest_pay_off',
             'is_homeowner', 'investment_typeid', 'real_estate_payment', 'real_estate_balance', 'bankcard_utilization', 'inquiries_last6_months',
             'prior_prosper_loans_cycles_billed', 'effective_yield', 'loan_default_reason_description']]

    
    active = factorize(dataframe=active)

    df = factorize(dataframe=df)
    
    df.loc[(df['loan_default_reason_description'] == 'Completed')|(df['loan_default_reason_description'] == 'PaidInFull'), 'loan_default_reason_description'] = 1
    df.loc[(df['loan_default_reason_description'] != 1), 'loan_default_reason_description'] = 2

    df = df.fillna(-999999)

    df_complete = df.loc[df['loan_default_reason_description'] == 1]
    df_incomplete = df.loc[df['loan_default_reason_description'] == 2]
    df_incomplete = df_incomplete.loc[df_incomplete.index.repeat(int(len(df_complete)/len(df_incomplete)))].reset_index()

    df = pd.concat([df_incomplete, df_complete])

    x_cols = ['borrower_apr', 'bankcard_utilization', 'funding_threshold', 'employment_status_description', 'fico_score', 'prior_prosper_loans_cycles_billed',]
    X = df[x_cols]
    y = df['loan_default_reason_description']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    standard_scaler = StandardScaler()
    X_train_standard = standard_scaler.fit_transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)


    clf = KNeighborsClassifier(3).fit(X_train_standard, y_train)
    
    y_pred = clf.predict(X_test_standard)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    # print(cm)
    print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(X_test_standard, y_test)))

    y_pred_active = clf.predict(active[x_cols])

    active['y_hats'] = y_pred_active

    return active


def main():
    auth = prosper_auth()

    df = get_historicals()
    active = get_active_listings(refresh=True, json_auth_response=auth)
    model_res = model(df, active)
    print(model_res)

#-------------------------------
#         Main Function
#-------------------------------
if __name__ == '__main__':
    main()
################################