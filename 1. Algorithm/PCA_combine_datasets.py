import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import proj3d

import os
from tqdm import tqdm

import joblib



def filter_counts_data(data, cell_min_molecules=1000, genes_min_cells=10):
    """Remove low molecule count cells and low detection genes

    :param data: Counts matrix: Cells x Genes
    :param cell_min_molecules: Minimum number of molecules per cell
    :param genes_min_cells: Minimum number of cells in which a gene is detected
    :return: Filtered counts matrix
    """

    # Molecule and cell counts
    ms = data.sum(axis=1)
    cs = data.sum()

    # Filter
    return data.loc[ms.index[ms > cell_min_molecules], cs.index[cs > genes_min_cells]]


def _clean_up(df):
    # Data frame = Access a group of rows and columns by label(s) or a boolean array.
    # df = Is the values in the data frame that where the rows> 0 or columns > 0(non zero)
    df = df.loc[df.index[df.sum(axis=1) > 0], :] #rows - >cells
    df = df.loc[:, df.columns[df.sum() > 0]]#colums  -> genes
    return df


def normalize_counts(data):
    """Correct the counts for molecule count variability

    :param data: Counts matrix: Cells x Genes
    :return: Normalized matrix
    """
    ms = data.sum(axis=1)
    norm_df = data.div(ms, axis=0).mul(np.median(ms), axis=0)
    return norm_df


def log_transform(data, pseudo_count=0.1):
    """Log transform the matrix

    :param data: Counts matrix: Cells x Genes
    :return: Log transformed matrix
    """
    return np.log2(data + pseudo_count)

def combine(Main_data_file, Secondary_data_file):
    '''
    -----------------------------------------------------------------
    Load data
    -----------------------------------------------------------------
    '''
    #Main data set
    try:
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        Main_data_file_dir = os.path.join(THIS_FOLDER, Main_data_file) 
        Main_data_file = pd.read_csv(Main_data_file_dir, sep=',', index_col=0)
    except:
        print("ERROR! The MAIN input data set does not exist. Please ensure that the correct file name is specified (Hint: Confirm spelling).")
        os._exit(1)
    
    #Secondary data set
    try:
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        Secondary_data_file_dir = os.path.join(THIS_FOLDER, Secondary_data_file) 
        Secondary_data_file = pd.read_csv(Secondary_data_file_dir, sep=',', index_col=0)
        # Secondary_data_file = Secondary_data_file.head(2000)
        # Secondary_data_file = Secondary_data_file.head(300)
    except:
        print("ERROR! The SECONDARY input data set does not exist. Please ensure that the correct file name is specified (Hint: Confirm spelling).")
        os._exit(1)

    print('Both data sets sucsefully loaded.')
    print('Combining data sets.')

    '''
    -----------------------------------------------------------------
    Select gene columns in common and Missing genese
    -----------------------------------------------------------------
    '''
    
    Genes_Common = Secondary_data_file.loc[:,Secondary_data_file.columns.isin(Main_data_file.columns.tolist())]
    Genes_Missing = Main_data_file.loc[:,~Main_data_file.columns.isin(Genes_Common.columns.tolist())]


    Genes_Missing_columns  = pd.DataFrame(columns = Genes_Missing.columns.tolist() )

    #Persentage gene overlapp
    persentage = (len(Genes_Common.columns) / len(Main_data_file.columns))*100
    print('Gene percentage overlapp: ' + str(persentage) + '%')
    #Append missing gene columns
    Frames = [Genes_Common, Genes_Missing_columns]
    Adjusted_SECONDARY_data_set = pd.concat(Frames, axis=1, sort=False)
    Adjusted_SECONDARY_data_set = Adjusted_SECONDARY_data_set.fillna(0) #Replace NAN with '0'


    '''
    -----------------------------------------------------------------
    Normalise secondary data set
    -----------------------------------------------------------------
    '''
    ms = Adjusted_SECONDARY_data_set.sum(axis=1)
    ms_2 = Main_data_file.sum(axis = 1)
    # Adjusted_SECONDARY_data_set_NORMALIZED = (Adjusted_SECONDARY_data_set - Adjusted_SECONDARY_data_set.min()) / (Adjusted_SECONDARY_data_set.max() - Adjusted_SECONDARY_data_set.min())

    #Multiply with median of Dataset 2
    Adjusted_SECONDARY_data_set_NORMALIZED = Adjusted_SECONDARY_data_set.div(ms, axis=0).mul(np.median(ms_2), axis=0)
    Adjusted_SECONDARY_data_set_NORMALIZED = Adjusted_SECONDARY_data_set.fillna(0) #Remove NAN
 

    #Rerange data in correct column order and MERGE with dataset 1
    Adjusted_SECONDARY_data_set_ORDERD = Adjusted_SECONDARY_data_set_NORMALIZED[Main_data_file.columns]
    Frames = [Main_data_file, Adjusted_SECONDARY_data_set_ORDERD]
    OUTPUT = pd.concat(Frames, axis=0, sort=False)

    #Calculate norm_df for the combined data sets:
    data = _clean_up(OUTPUT)
    filtered_counts = filter_counts_data(data, cell_min_molecules=1000, genes_min_cells=10)
    norm_df = normalize_counts(filtered_counts)
    norm_df = log_transform(norm_df)
    joblib.dump(norm_df, "norm_df.pkl", compress=3)

    #Redfine dataset 2
    new_data_FRAME = pd.DataFrame()

    
    #Set cell ID number to each cell to enable singel cell selection
    pt_samples = len(Adjusted_SECONDARY_data_set_ORDERD.index)
    cell_ID_number = range(pt_samples)
    label = Adjusted_SECONDARY_data_set_ORDERD.index
    d = {'cell_ID_number':cell_ID_number}
    cell_id_df = pd.DataFrame(d)

    Adjusted_SECONDARY_data_set_ORDERD.reset_index(drop=True, inplace=True)
    cell_id_df.reset_index(drop=True, inplace=True)


    #Append column
    Frames = [Adjusted_SECONDARY_data_set_ORDERD, cell_id_df]
    con = pd.concat(Frames, axis=1, sort=False)

    temp_Adjusted_SECONDARY_data_set_ORDERD = pd.DataFrame(con.values, index=label, columns = con.columns )

    #total added cells
    total_Secondary_cells_used = 0
    for l in tqdm(range(len(Adjusted_SECONDARY_data_set_ORDERD.index)), desc ="PCA data set combining."):
        #Select one cell
        select_one = temp_Adjusted_SECONDARY_data_set_ORDERD.head(1)
        #Remove selected cell from data
        temp_Adjusted_SECONDARY_data_set_ORDERD = temp_Adjusted_SECONDARY_data_set_ORDERD[~temp_Adjusted_SECONDARY_data_set_ORDERD['cell_ID_number'].isin(select_one['cell_ID_number'].values)]

        #Delete cell ID column
        del select_one['cell_ID_number']
        
        #Merge 1 cell data to Dataset 1
        Frames = [Main_data_file, select_one]
        data = pd.concat(Frames, axis=0, sort=False) 

        #Preform Palantir data pre-processing
        data = _clean_up(data)
        filtered_counts = filter_counts_data(data, cell_min_molecules=1000, genes_min_cells=10)
        norm_df = normalize_counts(filtered_counts)
        norm_df = log_transform(norm_df)

        #Select PCA
        pca = PCA(n_components=300, svd_solver='randomized')
        pca_projections = pca.fit_transform(norm_df)
        pca_projections = pd.DataFrame(pca_projections, index=norm_df.index)

        computed_PCA_1 = pca_projections.tail(1)

        #Only append if added data is 
        if computed_PCA_1.index == select_one.index:
            #Count how many cells are used
            total_Secondary_cells_used = total_Secondary_cells_used + 1
            #Append one cell data
            if new_data_FRAME.empty:
                new_data_FRAME.reset_index(drop=True, inplace=True)
                new_data_FRAME =computed_PCA_1

            else:
                # new_data_FRAME.reset_index(drop=True, inplace=True)
                Frames = [new_data_FRAME, computed_PCA_1]
                new_data_FRAME = pd.concat(Frames, axis=0, sort=False)#Append new lineage data

    joblib.dump(total_Secondary_cells_used, "total_Secondary_cells_used.pkl", compress=3)


    #Data set 1 calculations
    #Palantir pre-process data
    data = _clean_up(Main_data_file)
    filtered_counts = filter_counts_data(data, cell_min_molecules=1000, genes_min_cells=10)
    norm_df = normalize_counts(filtered_counts)
    norm_df = log_transform(norm_df)

    #Select PCA
    pca = PCA(n_components=300, svd_solver='randomized')
    pca_projections = pca.fit_transform(norm_df)
    pca_projections = pd.DataFrame(pca_projections, index=norm_df.index)

    #obtain indexes
    d = {'index_1': pca_projections.index}
    pca_projections_INDEX = pd.DataFrame(d) 


    d = {'index_1': new_data_FRAME.index}
    new_data_FRAME_INDEX = pd.DataFrame(d)

    # print(new_data_FRAME_INDEX)

    Frames = [pca_projections_INDEX, new_data_FRAME_INDEX]
    Index_recovery = pd.concat(Frames, axis=0, sort=False)#Append new lineage data

    pca_projections.reset_index(drop=True, inplace=True)
    new_data_FRAME.reset_index(drop=True, inplace=True)

    Frames = [pca_projections, new_data_FRAME]
    D1_combine_D2_PCA = pd.concat(Frames, axis=0, sort=False)#Append new lineage data

    D1_combine_D2_PCA.reset_index(drop=True, inplace=True)

    #Export combined data
    EXPORT_D1_combine_D2_PCA = pd.DataFrame(D1_combine_D2_PCA.values, index=Index_recovery.loc[:,'index_1'])
    EXPORT_D1_combine_D2_PCA.to_csv('Combined_PCA.csv')

    

    print('Combining data sets end.')
