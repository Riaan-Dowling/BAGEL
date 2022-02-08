import numpy as np
import pandas as pd
import networkx as nx

import os
import joblib

from sklearn import preprocessing #Normalise data [-1, 1]

import plots



def pseudo_data_import(palantir_pseudo_time_plot_FLAG, original_manifold_plot_FLAG):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    tsnedata = os.path.join(THIS_FOLDER, 'sample_tsne.p') 
    tsne = pd.read_pickle(tsnedata)
    pseudo_time  = joblib.load("pseudo_time.pkl") #Calculated pseudo_time
    waypoints  = joblib.load("waypoints.pkl") #Calculated pseudo_time
    c=pseudo_time[tsne.index]

    

    #Pseudo time dataframe

    #[0,1] normalise
    min_max_scaler = preprocessing.MinMaxScaler() #[0,1]
    pseudo_time = min_max_scaler.fit_transform(pseudo_time.values.reshape(-1,1))

    pseudo_time = pseudo_time.tolist()
    pseudo_time = [j for i in pseudo_time for j in i]


    # [-1,1] noramles
    min_max_scaler = preprocessing.MaxAbsScaler()
    tsne_1 = min_max_scaler.fit_transform(tsne['x'].values.reshape(-1,1))
    tsne_2 = min_max_scaler.fit_transform(tsne['y'].values.reshape(-1,1))
    
    tsne_1 = tsne_1.tolist()
    tsne_1 = [j for i in tsne_1 for j in i]

    tsne_2 = tsne_2.tolist()
    tsne_2 = [j for i in tsne_2 for j in i]

    
    d = {'Pseudo_Time_normal': pseudo_time,'tsne_1':tsne_1,'tsne_2': tsne_2}
    pseudo_data = pd.DataFrame(d, index=tsne.index)

    #Link terminal state to normalized manifold
    terminal_states  = joblib.load("terminal_states.pkl")

    wp_data_TSNE_ROW =pseudo_data.loc[terminal_states]

    joblib.dump(wp_data_TSNE_ROW, "wp_data_TSNE_ROW.pkl", compress=3)

    plots.palantir_pseudo_time_plot(pseudo_data, c, palantir_pseudo_time_plot_FLAG)
    plots.original_manifold_plot(pseudo_data,c, original_manifold_plot_FLAG)#Plot original manifold

    pseudo_data = pseudo_data.sort_values('Pseudo_Time_normal')

    pt_samples = len(tsne['x'])
    cell_ID_number = range(pt_samples)
    # Number of cell
    pseudo_data['cell_ID_number'] = cell_ID_number 

    return pseudo_data
