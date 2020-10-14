#import kaggle utilities
#from kaggle.api.kaggle_api_extended import KaggleApi
import os
import datetime
import os.path
from os import path
from datetime import timedelta
import pandas as pd

owd = os.getcwd()

#Check if file/directory exists
def file_check(filepath):
    result = os.path.exists(filepath)
    return result

#Download Dataset from Kaggle
#def dataset_download():

    #Authenticate with API Server
#    api = KaggleApi()
#    api.authenticate()

    #Select kaggle page and specfic file to download
#    page = 'rtatman/188-million-us-wildfires'
#    page_file = 'FPA_FOD_20170508.sqlite'

#    api.dataset_download_files(page, page_file)


    #unzip sqlite file
#    path_to_zip_file = 'FPA_FOD_20170508.sqlite/188-million-us-wildfires.zip'

#    import zipfile
#    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
#        zip_ref.extractall('FPA_FOD_20170508.sqlite/')


#    # Connect to SQLite database and import Fires table
#    import sqlite3
#    import pandas as pd

 #   conn = None
#    conn = sqlite3.connect('FPA_FOD_20170508.sqlite/FPA_FOD_20170508.sqlite')
    # cur = conn.cursor()

#    raw_df = pd.read_sql("""SELECT LATITUDE, LONGITUDE, FIRE_YEAR, DISCOVERY_DOY, STAT_CAUSE_DESCR, FIRE_SIZE_CLASS FROM fires WHERE FIRE_YEAR >2013""", con=conn)
#    conn.close()

#    raw_df.to_csv(r'2014-2015_raw.csv', index = False, header=True)

    # delete downloaded sqlite file/folder
#    import shutil
#    shutil.rmtree(page_file)

# Define test_train_splits
def test_train_split(dataframe):
    """Split dataframe into train and test data. Returns Train 1st, Test 2nd"""
    train = dataframe.sample(frac=0.8, random_state=100)  # random state is a seed value
    train = train.reset_index(drop=True)
    test = dataframe.drop(train.index)
    test = test.reset_index(drop=True)
    return train, test

#fix issue with stat_cause spellings for folder assignment
def string_fixer(word):
    new_word = word.replace(' ', '_')
    new_word = new_word.replace('/', '_')
    return new_word

#Boundaries defined for sentinel API requests (WIDTH & LENGTH ASSIGNED HERE)
def bbox_boundaries(dataframe):
    boundaries = []
    for i in range(len(dataframe)):
        long = dataframe['LONGITUDE'][i]
        lat = dataframe['LATITUDE'][i]
        long_left = long - 0.026
        long_right = long + 0.026
        lat_bottom = lat - 0.018
        lat_top = lat + 0.018
        bound_list = [round(long_left, 5), round(lat_bottom, 5), round(long_right, 5), round(lat_top, 5)]
        boundaries.append(bound_list)
    dataframe['BBOX'] = boundaries
    return dataframe


def add_date(dataframe):
    new_dates = []
    for i in range(len(dataframe)):
        start_date = str(dataframe['FIRE_YEAR'][i]) + '-1-1'
        DOY = dataframe['DISCOVERY_DOY'][i]
        Date = datetime.datetime.strptime(start_date, '%Y-%m-%d')+ timedelta(float(DOY))
        new_dates.append(Date.date())
    dataframe['Date'] = new_dates
    return dataframe

def cause_fixer(dataframe):
    new_strings = []
    for word in dataframe['STAT_CAUSE_DESCR']:
        new_word = string_fixer(word)
        new_strings.append(new_word)
    dataframe['STAT_CAUSE_DESCR'] = new_strings
    return dataframe


def reformatting(dataframe):
    df = bbox_boundaries(dataframe)
    df = add_date(df)
    df = cause_fixer(df)
    return df


def cause_splitter(dataframe, causes):
    for cause in causes:
        os.chdir(owd+'/Data')
        if file_check(cause):
            print("'"+str(cause)+"' "+"folder already exists")
        else:
            os.makedirs(cause)
            print("'" + str(cause) + "' " + "folder created")

        cause_df = dataframe[dataframe['STAT_CAUSE_DESCR']==cause]
        cause_df = cause_df.reset_index(drop=True)
        cause_df_train, cause_df_test = test_train_split(cause_df)

        os.chdir(cause)
        try:
            os.makedirs('train')
            print('Created train folder')
        except:
            print('train folder already created')
        try:
            os.makedirs('test')
            print('Created test folder')
        except:
            print('test folder already created')

        fixed_cause = string_fixer(cause)
        train_path = 'train/'+str(fixed_cause)+'_2014_2015_train.csv'
        test_path = 'test/'+str(fixed_cause)+'_2014_2015_test.csv'
        cause_df_train.to_csv(train_path)
        cause_df_test.to_csv(test_path)
        print('Train and Test data saved in '+str(cause)+' folder')
    return




def main_function():
    owd = os.getcwd()
    downloaded = file_check('2014-2015_raw.csv')
    try:
        os.makedirs('Data')
        print('Data folder created')
    except:
        print('Data folder already exists')
    os.chdir('Data')

    if downloaded:
        print('Reference data table has already been downloaded')
    else:
        print('Error')
        exit()
        #print('Reference data table has not been downloaded yet')
        #print('Downloading now...')
        #dataset_download()
        #print('Reference data table downloaded')

    os.chdir(owd)
    raw_df = pd.read_csv('2014-2015_raw.csv')
    df = reformatting(raw_df)

    list_of_causes = set(df['STAT_CAUSE_DESCR'])

    cause_splitter(df, list_of_causes)
    os.chdir(owd)
    print(' ')
    print('Reference data is ready for Sentinel imagery download')


if __name__ == '__main__':
    main_function()
