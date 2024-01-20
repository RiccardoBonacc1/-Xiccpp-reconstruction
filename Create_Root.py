import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

# For Training/Testing Classifiers:

def Cut_Root_Files(branch_list, files_list, cut_function, step_size, output_filename):
    # Create a new ROOT file
    file = uproot.recreate(output_filename)
    
    # Define the tree structure based on the branch list
    tree_structure = {branch: 'float64' for branch in branch_list}
    print("Defined tree structure:", tree_structure)
    file.mktree('tree', tree_structure)
    
    # Iterate over the files and branches in batches
    for batch in uproot.iterate(files_list, branch_list, step=step_size, library='np'):

        # Apply cuts:
        batch = cut_function(batch)

        # Create a dictionary for the current batch
        data_dict = {branch: np.array(batch[branch], dtype=np.float64) for branch in branch_list}
        
        # Extend the tree with the data from the batch
        file['tree'].extend(data_dict)
    
    # Close the file after writing
    file.close()



# Signal MC Files (PID corrections have been applied):

# 2450_MC_Xicc_Xicc2LcKPiPi_2018_Down:
filename_MC_0 = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/MC/Stripping/LDST/MC_Xicc2LcKPiPi_2018_MagDown_WithPIDCorr.root'

# 2451_MC_Xicc_Lc2pKPi_2018_Down:
filename_MC_1 = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/MC/Stripping/LDST/MC_Lc2pKPi_2018_MagDown_WithPIDCorr.root'

# 2452_MC_Xicc_Xicc2LcKPiPi_2018_Up:
filename_MC_2 = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/MC/Stripping/LDST/MC_Xicc2LcKPiPi_2018_MagUp_WithPIDCorr.root'

# 2453_MC_Xicc_Lc2pKPi_2018_Up:
filename_MC_3 = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/MC/Stripping/LDST/MC_Lc2pKPi_2018_MagUp_WithPIDCorr.root'

Xicc_filenames_MC = [filename_MC_0, filename_MC_2]

Lc_filenames_MC = [filename_MC_1, filename_MC_3]

# Background Data Files:

# Train/Test Files:

# 0-5 test data files:

# MagDown:

filename_Data_0_Down = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/0_Data_Xiccpp2LcpKmPipPip_2018_MagDown.root'

filename_Data_1_Down = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/1_Data_Xiccpp2LcpKmPipPip_2018_MagDown.root'

filename_Data_2_Down = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/2_Data_Xiccpp2LcpKmPipPip_2018_MagDown.root'

filename_Data_3_Down = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/3_Data_Xiccpp2LcpKmPipPip_2018_MagDown.root'

# filename_Data_4_Down = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/4_Data_Xiccpp2LcpKmPipPip_2018_MagDown.root'

# filename_Data_5_Down = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/5_Data_Xiccpp2LcpKmPipPip_2018_MagDown.root'

# MagDown:

filename_Data_0_Up = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/0_Data_Xiccpp2LcpKmPipPip_2018_MagUp.root'

filename_Data_1_Up = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/1_Data_Xiccpp2LcpKmPipPip_2018_MagUp.root'

filename_Data_2_Up = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/2_Data_Xiccpp2LcpKmPipPip_2018_MagUp.root'

filename_Data_3_Up = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/3_Data_Xiccpp2LcpKmPipPip_2018_MagUp.root'

# filename_Data_4_Up = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/4_Data_Xiccpp2LcpKmPipPip_2018_MagUp.root'

# filename_Data_5_Up = '/home/shared/XiccMultiplicity/Xicc2LcKPiPi/Data/Stripping/5_Data_Xiccpp2LcpKmPipPip_2018_MagUp.root'

filenames_Data = [filename_Data_0_Down, filename_Data_0_Up, filename_Data_1_Down, filename_Data_1_Up, filename_Data_2_Down, filename_Data_2_Up]
                #   , filename_Data_3_Down, filename_Data_3_Up] 
                #   , filename_Data_4_Down, filename_Data_4_Up, 
                #   filename_Data_5_Down, filename_Data_5_Up]


# Filenames:

Xicc_filenames_Data = [item + ':Xiccpp2LcpKmPipPip/DecayTree' for item in filenames_Data]

Lc_filenames_Data = [item + ':Lcp2pKPi/DecayTree' for item in filenames_Data]

def Signal_Xicc_cutting(np_MC_Xicc):

    # Signal Xicc Cuts:

    # Fix variables:
    np_MC_Xicc["Xicc_MassFit_chi2"] = np.concatenate(np_MC_Xicc["Xicc_MassFit_chi2"]).ravel()
    np_MC_Xicc["Xicc_MassFit_nDOF"] = np.concatenate(np_MC_Xicc["Xicc_MassFit_nDOF"]).ravel()

    # Define Xicc subset mask:
    cut_MC_BKGCAT_Xicc = ((np_MC_Xicc['Xicc_BKGCAT']==0) | (np_MC_Xicc['Xicc_BKGCAT']==10))

    cut_MC_PID_Xicc    = (np_MC_Xicc['LcPi_PIDK_corr']<5.0) & \
                    (np_MC_Xicc['LcK_PIDK_corr'] >5.0) & \
                    (np_MC_Xicc['LcP_PIDp_corr']>5.0)& \
                    (np_MC_Xicc['LcP_PIDp_corr'] - np_MC_Xicc['LcP_PIDK_corr']>5.0) & \
                    (np_MC_Xicc['XiccPi1_ProbNNpi_corr']>0.2) & \
                    (np_MC_Xicc['XiccPi2_ProbNNpi_corr']>0.2) & \
                    (np_MC_Xicc['XiccK_ProbNNk_corr']>0.1)

    # Don't take events which are too busy, we implemented a cut-off:
    cut_MC_NDOF_Xicc = (np_MC_Xicc['Xicc_OWNPV_NDOF']<200) 

    # Get rid of events which possess no PID knowledge:
    cut_PID = (np_MC_Xicc['LcPi_PIDK_corr'] > -700) & (np_MC_Xicc['LcK_PIDK_corr'] > -700)

    Signal_Xicc_cut = cut_MC_BKGCAT_Xicc & cut_MC_PID_Xicc & cut_MC_NDOF_Xicc &cut_PID

    Signal_Xicc = {key: np_MC_Xicc[key][Signal_Xicc_cut] for key in np_MC_Xicc.keys()}

    return Signal_Xicc

def Signal_Lc_cutting(np_MC_Lc):

    # Signal Lc Cuts:

    # Define Lc subset mask:
    cut_MC_BKGCAT_Lc = ((np_MC_Lc['Lc_BKGCAT']==0) | (np_MC_Lc['Lc_BKGCAT']==10))

    cut_MC_PID_Lc    = (np_MC_Lc['LcPi_PIDK_corr']<5.0) & \
                    (np_MC_Lc['LcK_PIDK_corr'] >5.0) & \
                    (np_MC_Lc['LcP_PIDp_corr']>5.0)& \
                    (np_MC_Lc['LcP_PIDp_corr'] - np_MC_Lc['LcP_PIDK_corr']>5.0) & \
                    (np_MC_Lc['LcPi_ProbNNpi_corr']>0.2) & \
                    (np_MC_Lc['LcK_ProbNNk_corr']>0.1)

    # Don't take events which are too busy, we implemented a cut-off:
    cut_MC_NDOF_Lc = (np_MC_Lc['Lc_OWNPV_NDOF']<200) 

    cut_PID = (np_MC_Lc['LcPi_PIDK_corr'] > -700) & (np_MC_Lc['LcK_PIDK_corr'] > -700) & (np_MC_Lc['LcP_PIDp_corr'] > -700)

    Signal_Lc_cut = cut_MC_BKGCAT_Lc & cut_MC_PID_Lc & cut_MC_NDOF_Lc & cut_PID

    Signal_Lc = {key: np_MC_Lc[key][Signal_Lc_cut] for key in np_MC_Lc.keys()}

    return Signal_Lc
def Background_Xicc_cutting(np_Data_Xicc):
    
    # Fix data:
    np_Data_Xicc["Xicc_MassFit_chi2"] = np.concatenate(np_Data_Xicc["Xicc_MassFit_chi2"]).ravel()
    np_Data_Xicc["Xicc_MassFit_nDOF"] = np.concatenate(np_Data_Xicc["Xicc_MassFit_nDOF"]).ravel()

    # Take end as background on purpose:
    # Will do this cut later on:
    # cut_Data_mass_Xicc = (np_Data_Xicc['Xicc_M']>3700.0)
    
    # Background_Xicc = {key: np_Data_Xicc[key][cut_Data_mass_Xicc] for key in np_Data_Xicc.keys()}

    # return Background_Xicc

    return np_Data_Xicc
def Background_Lc_cutting(np_Data_Lc):
    
    # Take end as background on purpose:
    # cut_Data_mass_Lc = (np_Data_Lc['Lc_M']>2320)

    # Background_Lc = {key: np_Data_Lc[key][cut_Data_mass_Lc] for key in np_Data_Lc.keys()}

    # return Background_Lc
    return np_Data_Lc
# Selected Branches:

Signal_Branches_Xicc = [
    "Xicc_MassFit_chi2",
    "Xicc_MassFit_nDOF",
    "Xicc_BKGCAT",
    "LcPi_PIDK_corr",
    "LcK_PIDK_corr",
    "LcP_PIDp_corr",
    'LcP_PIDK_corr',
    "XiccPi1_ProbNNpi_corr",
    "XiccPi2_ProbNNpi_corr",
    "XiccK_ProbNNk_corr",
    "Xicc_OWNPV_NDOF",
    "Xicc_IPCHI2_OWNPV",
    "Xicc_DIRA_OWNPV",
    "Xicc_FDCHI2_OWNPV",
    "Lc_ENDVERTEX_CHI2",
    "Lc_ENDVERTEX_NDOF",
    "Xicc_ENDVERTEX_CHI2",
    "Xicc_ENDVERTEX_NDOF",
    "Xicc_MassFit_chi2",
    "Xicc_MassFit_nDOF",
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
    "Xicc_M"
]

Signal_Branches_Lc = [
    "Lc_BKGCAT",
    "LcPi_PIDK_corr",
    "LcK_PIDK_corr",
    "LcP_PIDp_corr",
    "LcP_PIDK_corr",
    'LcPi_ProbNNpi_corr',
    'LcK_ProbNNk_corr',
    "Lc_OWNPV_NDOF",
    "Lc_IPCHI2_OWNPV",
    "Lc_DIRA_OWNPV",
    "Lc_FDCHI2_OWNPV",
    "Lc_ENDVERTEX_CHI2",
    "Lc_ENDVERTEX_NDOF",
    "LcK_IPCHI2_OWNPV",
    "LcPi_IPCHI2_OWNPV",
    "LcP_IPCHI2_OWNPV",
    "LcK_PT",
    "LcPi_PT",
    "LcP_PT",
    "Lc_M"
]

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
    'Xicc_OWNPV_NDOF'
]


Background_Branches_Lc = [
    "Lc_M",
    "Lc_IPCHI2_OWNPV",
    "Lc_DIRA_OWNPV",
    "Lc_FDCHI2_OWNPV",
    "Lc_ENDVERTEX_CHI2",
    "Lc_ENDVERTEX_NDOF",
    "LcK_IPCHI2_OWNPV",
    "LcPi_IPCHI2_OWNPV",
    "LcP_IPCHI2_OWNPV",
    "LcK_PT",
    "LcPi_PT",
    "LcP_PT",
    "Lc_OWNPV_NDOF"
]
def Identity(x):
    return x

# Produce Cut Files:

# Lc data files:
Cut_Root_Files(Signal_Branches_Lc, Lc_filenames_MC, Signal_Lc_cutting, 8000, '/home/bonacci/Data/Lc_MC_Train.root')
Cut_Root_Files(Background_Branches_Lc, Lc_filenames_Data, Background_Lc_cutting, 8000, '/home/bonacci/Data/Lc_Data_Train.root')

# Xicc data files:
Cut_Root_Files(Signal_Branches_Xicc, Xicc_filenames_MC, Signal_Xicc_cutting, 8000, '/home/bonacci/Data/Xicc_MC_Train.root')
Cut_Root_Files(Background_Branches_Xicc, Xicc_filenames_Data, Background_Xicc_cutting, 8000, '/home/bonacci/Data/Xicc_Data_Train.root')