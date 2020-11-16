import numpy as np
import pandas as pd


def tf_df(train, mode='train'):

    model_name = train['모델명']
    years = train['연식']

    tf_model_name = []
    tf_years = []

    for name, year in zip(model_name, years):
        tf_name = name.split(' ')[0]
        tf_model_name.append(tf_name)

        tf_year = 2021 - year
        tf_years.append(tf_year)

    distance = train['주행거리']
    distance.head()

    tf_distance=[]
    for d in distance:
        if 'ml' in d:
            if '만' in d:
                val = int(d.split('만')[0])
                val = val*10000*1.609344

        elif 'km' in d:
            if '만' in d:
                val = int(d.split('만')[0])
                val = val * 10000
            elif '천' in d:
                val = int(d.split('천')[0])
                val = val * 1000
            else: 
                val = int(d.split('km')[0])

        tf_distance.append(val)
    len(tf_distance)

    power=train['최대출력(마력)']
    torq=train['최대토크(kgm)']
    fuel=train['연료']
    method=train['구동방식']
    engine=train['기통']
    
    if mode=='train':
        label=train['가격(만원)']

        df = pd.DataFrame({
                             #'model_name': tf_model_name,
                             'label': label,

                             'years': tf_years,
                             'distance': tf_distance,
                             'power':power,
                             'torq': torq,
                             'fuel': fuel,
                             'method': method,
                             'engine': engine

        })
        
    else:
         df = pd.DataFrame({
                             #'model_name': tf_model_name,
                             'years': tf_years,
                             'distance': tf_distance,
                             'power':power,
                             'torq': torq,
                             'fuel': fuel,
                             'method': method,
                             'engine': engine


        })
            
    return df



def get_minmax(df, df_test, categoric_list):
    minmax={}
    for col in categoric_list:
        
        train = df[col]

        
        if col == 'label': 
            minmax[col]=[ np.min(train), np.max(train), np.mean(train)]
            continue
                
        test = df_test[col]
        minmax[col]=[np.min((np.min(test), np.min(train))), np.max((np.max(test), np.max(train))), np.mean(train)]

    return minmax

def categoric_process(df, df_test, onehot_colums):
    for col in onehot_colums:
        train = df[col]
        test = df_test[col]
        
        train_onehot = pd.get_dummies(train, drop_first=True)
        test_onehot = pd.get_dummies(test, drop_first=True)
        
        train_cols = train_onehot.columns
        test_cols = test_onehot.columns
        
        for i, var in enumerate(train_cols):
            if var not in test_cols: 
                test_onehot.insert(i, var, [0 for i in range(len(test_onehot))] )
            
            df[var] = train_onehot[var].to_numpy() 
            df_test[var] = test_onehot[var].to_numpy() 
            
        df = df.drop([col], axis=1)
        df_test = df_test.drop([col], axis=1)

            
    return df, df_test

def numeric_process(df, df_test, categoric_list):
    minmax = get_minmax(df, df_test, categoric_list)
    for col in minmax:
        print(col, minmax[col])
        train = df[col]
        train = train.fillna(minmax[col][2])
        train = (train-minmax[col][0]) / (minmax[col][1]-minmax[col][0])
        df[col]=train
        
        if col=='label': continue
        test = df_test[col]
        test = test.fillna(minmax[col][2])
        test = (test-minmax[col][0]) / (minmax[col][1]-minmax[col][0])
        df_test[col]=test
        
    return df, df_test
