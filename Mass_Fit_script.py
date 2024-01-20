import os
import sys
import argparse
import numpy as np
import pandas as pd
import uproot
import pickle
import torch

import argparse

# Argument parser

parser = argparse.ArgumentParser(description='Add MVA output')
parser.add_argument('-n', '--number',   type=int, default = 0, 
                    help='Input file number')
parser.add_argument('-p', '--polarity',  choices=['MagDown', 'MagUp'], type=str, default = "MagDown", 
                    help='Input file polarity')

args = parser.parse_args()

if(__name__ == "__main__"):
    
    ## Load the input file 
    input_dir  = "/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping"
    input_name = f"{args.number}_Data_Xiccpp2LcpKmPipPip_2018_{args.polarity}.root"
    
    print(f"Processing file  {input_dir}/{input_name}")

    # Set up INPUT ROOT FILE for testing:

    # Produce Signal and Background DataFrame (All of Signal and 0,1,2 of MagUp and MagDown of Data) (default PID Cuts have already been applied)

    df_Signal_Xicc = uproot.open('/home/bonacci/Data/Xicc_MC_Train.root:tree').arrays(library = 'pd')
    # df_Signal_Xicc = df_Signal_Xicc.astype('float32')  # Convert to float32

    df_Background_Xicc = uproot.open('/home/bonacci/Data/Xicc_Data_Train.root:tree').arrays(library = 'pd')
    # df_Background_Xicc = df_Background_Xicc.astype('float32')  # Convert to float32

    # Create Training Variables:

    def Create_Train(df):
        
        # 1:
        df["LOG[Xicc_IPCHI2_OWNPV]"] = np.log(df["Xicc_IPCHI2_OWNPV"])
        # 2:
        df["COS^{-1}[Xicc_DIRA_OWNPV]"] = np.arccos(df["Xicc_DIRA_OWNPV"])
        # 3:
        df["LOG[Xicc_FDCHI2_OWNPV]"] = np.log(df["Xicc_FDCHI2_OWNPV"])
        # 4:
        df["Lc_ENDVERTEX_CHI2/NDOF"] = df["Lc_ENDVERTEX_CHI2"] / df["Lc_ENDVERTEX_NDOF"]
        # 5:
        df["Xicc_ENDVERTEX_CHI2/NDOF"] = df["Xicc_ENDVERTEX_CHI2"] / df["Xicc_ENDVERTEX_NDOF"]
        # 6:
        df["Xicc_MassFit_chi2/nDOF"] = df["Xicc_MassFit_chi2"] / df["Xicc_MassFit_nDOF"]
        # 7:
        columns_to_check = ['XiccK_IPCHI2_OWNPV', 'XiccPi1_IPCHI2_OWNPV', 'XiccPi2_IPCHI2_OWNPV', 'Lc_IPCHI2_OWNPV']
        df['Xicc_MIN_IPCHI2_OWNPV'] = np.log(df[columns_to_check]).min(axis=1)
        # 8:
        columns_to_check = ["XiccK_PT", "XiccPi1_PT", "XiccPi2_PT", "Lc_PT"]
        df['Xicc_SUM_PT'] = df[columns_to_check].sum(axis=1)
        # 9:
        columns_to_check = ["XiccK_PT", "XiccPi1_PT", "XiccPi2_PT", "Lc_PT"]
        df['Xicc_MIN_PT'] = df[columns_to_check].min(axis=1)
        # 10:
        columns_to_check = ["LcK_PT", "LcPi_PT", "LcP_PT"]
        df['Lc_MIN_PT'] = df[columns_to_check].min(axis=1)
        return df

    # Prepare data for training:

    # Create Signal and Background training variables:

    df_Signal_Xicc = Create_Train(df_Signal_Xicc)
    df_Background_Xicc = Create_Train(df_Background_Xicc)

    # Training Variables for Xicc:

    training_variables_Xicc = ["LOG[Xicc_IPCHI2_OWNPV]", "COS^{-1}[Xicc_DIRA_OWNPV]", "LOG[Xicc_FDCHI2_OWNPV]", "Lc_ENDVERTEX_CHI2/NDOF", 
                        "Xicc_ENDVERTEX_CHI2/NDOF", "Xicc_MassFit_chi2/nDOF", 'Xicc_MIN_IPCHI2_OWNPV', 'Xicc_SUM_PT', 
                        'Xicc_MIN_PT', 'Lc_MIN_PT']
                        
    # Defining the parameter kept uniform and the other parameters being trained:

    uniform_variables  = ["Xicc_OWNPV_NDOF"]

    df = pd.concat([df_Signal_Xicc, df_Background_Xicc], ignore_index=False)

    # To get Scaling mean and variance:
    X = df[training_variables_Xicc + uniform_variables]

    def Scaler(X, X_mean, X_std):
        x = x = (X - X_mean) / X_std
        return x

    # For Scaling the data:
    X_mean = np.mean(X, axis = 0)
    X_std = np.std(X, axis = 0)


    # Begin using the model on the INPUT ROOT FILES:


    # Loading the pickled File:

    with open('/home/bonacci/Work/classifiers_dict.pkl', 'rb') as f:
        loaded_classifier = pickle.load(f)

    # This classifier didn't test well:
    loaded_classifier.pop('KNN - hep_ml')

    # Branches Used:
    Background_Branches_Xicc = [
        "Xicc_MassFit_chi2",
        "Xicc_MassFit_nDOF",
        "Xicc_M",
        "Xicc_IPCHI2_OWNPV",
        "Xicc_DIRA_OWNPV",
        "Xicc_FDCHI2_OWNPV",
        "Lc_ENDVERTEX_CHI2",
        "Lc_ENDVERTEX_NDOF",
        "Xicc_ENDVERTEX_CHI2",
        "Xicc_ENDVERTEX_NDOF",
        "XiccK_IPCHI2_OWNPV",
        "XiccPi1_IPCHI2_OWNPV",
        "XiccPi2_IPCHI2_OWNPV",
        "Lc_IPCHI2_OWNPV",
        "XiccK_PT",
        "XiccPi1_PT",
        "XiccPi2_PT",
        "Lc_PT",
        "LcK_PT",
        "LcPi_PT",
        "LcP_PT",
        'Xicc_OWNPV_NDOF',
        "Lc_M"
    ]

    filename_Data = f"{input_dir}/{input_name}" + ":Xiccpp2LcpKmPipPip/DecayTree" #INPUT DATA FILE

    Xicc_bgd = uproot.open(filename_Data).arrays(Background_Branches_Xicc, library = 'np')

    # Fix variables:
    Xicc_bgd["Xicc_MassFit_chi2"] = np.concatenate(Xicc_bgd["Xicc_MassFit_chi2"]).ravel()
    Xicc_bgd["Xicc_MassFit_nDOF"] = np.concatenate(Xicc_bgd["Xicc_MassFit_nDOF"]).ravel()

    Xicc_bgd = pd.DataFrame.from_dict(Xicc_bgd)

    # Preparing Training/Testing Signal and Background DataFrame:

    def Prepare_Training(df):
        cuts = (df["Xicc_IPCHI2_OWNPV"]>0) & (df["Xicc_DIRA_OWNPV"]>-1) & (df["Xicc_DIRA_OWNPV"]<1) & (df["Xicc_FDCHI2_OWNPV"]>0) & (df["Lc_ENDVERTEX_NDOF"]!=0) & (df["Xicc_ENDVERTEX_NDOF"]!=0) & (df["Xicc_MassFit_nDOF"]!=0) & (df['XiccK_IPCHI2_OWNPV']>0) & (df['XiccPi1_IPCHI2_OWNPV']>0) & (df['XiccPi2_IPCHI2_OWNPV']>0) & (df['Lc_IPCHI2_OWNPV']>0) & (df['XiccK_IPCHI2_OWNPV']>0)
        return df[cuts].copy()

    Xicc_bgd = Prepare_Training(Xicc_bgd)

    Create_Train(Xicc_bgd)
    # Training Variables for Xicc:

    training_variables_Xicc = ["LOG[Xicc_IPCHI2_OWNPV]", "COS^{-1}[Xicc_DIRA_OWNPV]", "LOG[Xicc_FDCHI2_OWNPV]", "Lc_ENDVERTEX_CHI2/NDOF", 
                        "Xicc_ENDVERTEX_CHI2/NDOF", "Xicc_MassFit_chi2/nDOF", 'Xicc_MIN_IPCHI2_OWNPV', 'Xicc_SUM_PT', 
                        'Xicc_MIN_PT', 'Lc_MIN_PT']
                        
    # Defining the parameter kept uniform and the other parameters being trained:

    uniform_variables  = ["Xicc_OWNPV_NDOF"]
    
    # For Scaling the data:
    x = Scaler(Xicc_bgd[training_variables_Xicc+uniform_variables], X_mean, X_std)


    # Collect classification probabilities:

    for name, clf in loaded_classifier.items():
        if name in ['NN - PyTorch', 'NN_FL - PyTorch']:
            x_test_torch = torch.tensor(x[training_variables_Xicc].to_numpy(), dtype=torch.float32)
            Y_proba = clf(x_test_torch).detach().numpy()
        
        elif name in ['AdaBoost - sklearn', 'Random Forest - sklearn', 'KNN - sklearn']:
            Y_proba = clf.predict_proba(x[training_variables_Xicc])[:, 1]  # Assuming binary classification and you want the proba for class 1

        else:
            Y_proba = clf.predict_proba(x)[:, 1]  # Assuming binary classification and you want the proba for class 1

        # Add predictions to DataFrame:
        Xicc_bgd[f"{name}"] = Y_proba

    # Classifier AdaBoost - sklearn: Mean Cut = 0.94994994994995, Mean FoM = 1.8888339092813748

    # Classifier KNN - hep_ml: Mean Cut = 0.2747474747474748, Mean FoM = 1.2092490660781732

    # Classifier KNNFL - hep_ml: Mean Cut = 0.890909090909091, Mean FoM = 1.6054257816136297

    # Classifier FL - hep_ml: Mean Cut = 0.8868686868686868, Mean FoM = 1.6239895224754892

    # Classifier NN - PyTorch: Mean Cut = 0.8972972972972973, Mean FoM = 1.906554262320239

    # Classifier NN_FL - PyTorch: Mean Cut = 0.9212121212121213, Mean FoM = 1.951426895679742

    def df_to_Root(df, output_filename):
        # Create a new ROOT file
        with uproot.recreate(output_filename) as file:
            # Define the tree structure based on the DataFrame columns
            tree_structure = {col: df[col].dtype.name for col in df.columns}
            
            # Convert the dtypes to string formats that ROOT understands
            for key in tree_structure.keys():
                if "int" in tree_structure[key]:
                    tree_structure[key] = "int64"
                elif "float" in tree_structure[key]:
                    tree_structure[key] = "float64"
                # Add more dtype conversions if needed
            
            # Create a tree
            tree = file.mktree("tree", tree_structure)
            
            # Convert DataFrame to a dictionary of NumPy arrays
            data_dict = {col: df[col].to_numpy() for col in df.columns}
            
            # Extend the tree with the data
            tree.extend(data_dict)

    # Define a dictionary with classifier names as keys and their mean cut values as values
    classifier_cuts = {
        'AdaBoost - sklearn': 0.94994994994995,
        'KNNFL - hep_ml': 0.890909090909091,
        'FL - hep_ml': 0.8868686868686868,
        'NN - PyTorch': 0.8972972972972973,
        'NN_FL - PyTorch': 0.9212121212121213,
    }

    # Iterate over the classifier names and their corresponding cut values
    for classifier_name, cut_value in classifier_cuts.items():
        # Filter the DataFrame based on the cut value for the current classifier
        filtered_data = Xicc_bgd[Xicc_bgd[classifier_name] > cut_value]

        # Sanitize the classifier name and modify the output filename
        sanitized_classifier_name = classifier_name.replace(" ", "_").replace("-", "_")

        ## Create an output file 
        output_dir  = "/home/bonacci/With_MVA"
        output_name = f"{args.number}_Data_Xiccpp2LcpKmPipPip_2018_{args.polarity}_WithMVA_{sanitized_classifier_name}.root"
        print(f"Creating file  {output_dir}/{output_name}")

        # Write the DataFrame to a modified ROOT file
        df_to_Root(filtered_data, f"{output_dir}/{output_name}")  # OUTPUT DATA FILE
