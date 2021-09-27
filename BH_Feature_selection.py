# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 23:12:28 2021

@author: prasa
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def fitness_func(y_actual, y_pred, selected_features):
    acc = accuracy_score(y_actual, y_pred)
    return (acc / (1 + 0.01 * selected_features))
    
def final_accuracy_func(X_train, X_test, y_train, y_test):
    
    n_estimators = [50, 100, 200, 300, 500]
    max_features = [0.1, 0.2, 0.4, 0.6, 0.8]
    max_features.append('auto')
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)

    "Grid"
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth}
    
    clf = RandomForestClassifier()
    scorer = make_scorer(accuracy_score)
    model = RandomizedSearchCV(estimator = clf, 
                               param_distributions = random_grid, 
                               n_iter = 100, 
                               cv = 5, 
                               verbose=1, 
                               random_state=42, 
                               n_jobs = -1,
                               scoring = scorer)

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return acc

def CrossValidation(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    selected_features = X_train.shape[1]
    fit = fitness_func(y_test, y_pred, selected_features)
    return fit

def CheckforNullPopulation(population):
    i = 0
    #newPop = []
    while(i < len(population)):
        if sum(population[i]) == 0:
            population[i] = np.random.randint(low = 0,high = 2,size = (len(population[i])))
            continue
        i += 1
    return population

def preprocess(pre_data):
    X = pre_data.iloc[:,:-1]
    y = pre_data.iloc[:,-1]
    X.fillna(0,inplace=True)
    
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y))
    
    scalar = MinMaxScaler()
    X = pd.DataFrame(scalar.fit_transform(X))
    return X,y

def data_read(dataset_name):
    try:
        if dataset_name != './datasets/HeartEW.csv':
            pre_data = pd.read_csv(dataset_name,sep=',',header=None)
            pre_data.drop(0,axis=0,inplace=True)
        else:
            pre_data = pd.read_csv(dataset_name,sep=' ',header=None)
            pre_data.drop(0,axis=0,inplace=True)
    except:
        pre_data =  pd.read_excel(dataset_name)
        pre_data.reset_index(drop=True,inplace=True)
        pre_data.drop(0,axis=0,inplace=True)
    
    return pre_data

def Separte(population, PopulationSize):
    fitness = []
    for pop in range(PopulationSize):
        X_train_sample = X_train.iloc[:,population[pop]==1]
        X_test_sample = X_test.iloc[:,population[pop]==1]
        fitness.append(CrossValidation(X_train_sample, X_test_sample, y_train, y_test))
    
    
    BH_fitness = max(fitness)
    BH = population[np.argmax(fitness)]
    
    stars = np.delete(population,np.argmax(fitness),axis=0)
    stars_fitness = np.delete(fitness,np.argmax(fitness),axis=0)
    
    return BH, BH_fitness, stars, stars_fitness

def BH_feature_selection(population, PopulationSize, bitSize):
    
    for gen in range(maxiter + 1):
                    
        BH, BH_fitness, stars, stars_fitness = Separte(population, PopulationSize)

        print('Generation',gen, 'Accuracy', BH_fitness*(1+0.01*sum(BH)))
        if gen > maxiter:
            break
        
        event_horizon = BH_fitness / sum(stars_fitness)
            
        for moving in range(len(stars)):
            if (BH_fitness - stars_fitness[moving]) <= event_horizon:
                stars[moving] = np.random.randint(low = 0,high = 2,size = bitSize)
            else:
                index_to_replace = np.random.choice(np.where(BH != stars[moving])[0],
                                                    int(0.25 * sum(BH != stars[moving])),
                                                    replace=False)
            
                stars[moving][index_to_replace] = abs(stars[moving][index_to_replace] - 1)
        
        population = np.append(stars,[BH],axis=0)
        population = CheckforNullPopulation(population)
    return BH, BH_fitness, stars, stars_fitness

PopulationSize = 5
maxiter = 10
dfff = []

filename = [ 'biodeg.csv',
             'BreastEW.csv',
             'Cardiotocography.xls',
             'colon.csv',
             'derm.csv',
             'HeartEW.csv',
             'IonosphereEW.csv',
             'leukemia.csv',
             'spambase.csv',
             'steel-plates-fault_csv.csv',
             'WaveformEW.csv',
             'WineEW.csv']


for data_set in filename:
    print(data_set)
    dfff = []
    for runs in range(2):
        print('Run:', runs)
        pre_data = data_read('./datasets/' + data_set)
        X, y = preprocess(pre_data)
        bitSize = X.shape[1]
        
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 42)
        
        population = np.random.randint(low = 0,high = 2,size = (PopulationSize, bitSize))
        
        BH, BH_fitness, stars, stars_fitness = BH_feature_selection(population, PopulationSize, bitSize)
        
        X_train_sample = X_train.iloc[:,BH==1].copy()
        X_test_sample = X_test.iloc[:,BH==1].copy()
            
        final_BH_accuracy = final_accuracy_func(X_train_sample, X_test_sample, y_train, y_test)
    
        dfff.append([X_train_sample.columns, sum(BH), BH_fitness, final_BH_accuracy])
        
    dataset_df = pd.DataFrame(dfff)
    dataset_df.columns = ['subset','subset_size','fitness','accuracy']
    dataset_df['dataset'] = data_set
    #dataset_df.to_csv('./Results/BH/results_'+ data_set, encoding='utf-8', index=False)
