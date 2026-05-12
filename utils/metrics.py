import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy.interpolate import PchipInterpolator

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

roc_storage = {}
base_fpr = np.linspace(0, 1, 100)

years = [1, 2, 3, 4, 5]
baselines = ['CN', 'MCI']
rts = range(1)

def get_bacc(y_true, y_pred, weight):
    """
    Compute the balanced accuracy score.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels of samples.
    y_pred : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class.
    weight : array-like of shape (n_samples,)
        Sample weights.

    Returns:
    --------
    bacc : float
        Balanced accuracy score.
    """
    # Normalize the weight based on the number of samples
    weight *= len(y_true) / np.sum(weight)
    
    # Get a boolean array indicating whether the prediction is correct or not
    correct_preds = (y_true == np.argmax(y_pred, axis=1))
    
    # Calculate the weighted number of correct predictions for each sample
    weighted_correct_preds = weight * correct_preds
    
    # Compute the average of the weighted correct predictions over all samples
    bacc = np.mean(weighted_correct_preds)
    
    # Scale the balance accuracy value to percentage.
    return bacc * 100


def get_rocauc(y_true, y_pred, dx_bl):
    """
    Calculate one-vs-rest the area under the receiver operating characteristic curve (AUC-ROC).
    For CN baseline, the positive class is MCI.
    For MCI baseline, the positive class is Dementia.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities or confidence scores for the positive class.
        dx_bl (int): Baseline diagnosis code. 0 for CN-baseline and 1 for MCI-baseline.

    Returns:
        float: The ROC_AUC score.

    """
    
    if dx_bl == 0:
        y_pred = y_pred[:, 1]  # Get the predicted probabilities the positive class.
        y_true = (y_true == 1).astype(int) # Get the one-vs-rest labels for the postive class.
    elif dx_bl == 1:
        y_pred = y_pred[:, 2]   # Get the predicted probabilities the positive class.
        y_true = (y_true == 2).astype(int)  # Get the one-vs-rest labels for the postive class.
        
    # Calculate the ROC_AUC score.
    roc_auc = roc_auc_score(y_true, y_pred) 
    
    return roc_auc * 100


def get_performance_metrics(df):
    """
    Calculates balanced accuracy (BAcc) and area under the receiver operating characteristic curve (ROCAUC) 
    for each follow-up year and baseline diagnostic group (CN or MCI) as well as the average over all years 
    and diagnostic groups.
    
    Parameters:
    df (pandas.DataFrame): Dataframe containing the true follow-up diagnoses (FollowupDX), predicted probabilities 
                            (Pred), and sample weights (SampleWeight) for each subject.
    
    Returns:
    metrics (dict): Dictionary containing the calculated metrics.
    """
    
    # Calculate metrics.
    metrics = {}

    # Loop over the two baseline diagnostic groups (CN and MCI).
    for dx_bl, dx_bl_name in zip([0, 1], ['CN', 'MCI']):
        # Subset the dataframe to include only subjects with the current baseline diagnosis.
        df_bl = df.loc[df['BaselineDX']==dx_bl]
        
        # Initialize lists to store BAcc and ROCAUC for each follow-up year.
        baccs = []
        rocaucs = []
        
        # Loop over the unique follow-up years.
        for year in np.unique(df['FollowupYear']):
            # Subset the dataframe to include only subjects with the current baseline diagnosis and follow-up year.
            year_df = df_bl.loc[df_bl['FollowupYear']==year]
            
            # Extract the true follow-up diagnoses, predicted probabilities, and sample weights.
            y_true = year_df['FollowupDX'].values
            y_pred = np.vstack(year_df['Pred'].values)
            weight = year_df['SampleWeight'].values
            
            # Calculate the BAcc and ROCAUC for the current year and baseline diagnosis.
            bacc = get_bacc(y_true, y_pred, weight)
            rocauc = get_rocauc(y_true, y_pred, dx_bl)
            
            # Add the BAcc and ROCAUC to the metrics dictionary with appropriate keys.
            metrics['BAcc_'+dx_bl_name+'_'+str(int(year))] = bacc
            metrics['ROCAUC_'+dx_bl_name+'_'+str(int(year))] = rocauc
            
            # Append the BAcc and ROCAUC to the lists.
            baccs.append(bacc)
            rocaucs.append(rocauc)
    
        # Calculate the average BAcc and ROCAUC over all follow-up years for the current baseline diagnosis.
        metrics['Avg_BAcc_'+dx_bl_name] = np.mean(baccs)
        metrics['Avg_ROCAUC_'+dx_bl_name] = np.mean(rocaucs)
    
    # Calculate the average BAcc and ROCAUC over all follow-up years and both baseline diagnostic groups.
    metrics['Avg_BAcc'] = 0.5*(metrics['Avg_BAcc_CN']+metrics['Avg_BAcc_MCI'])
    metrics['Avg_ROCAUC'] = 0.5*(metrics['Avg_ROCAUC_CN']+metrics['Avg_ROCAUC_MCI'])
            
    return metrics


def get_roc_aucs(df):
    """
    Computes the area under the ROC curve (ROCAUC) for each follow-up year and baseline diagnosis (CN or MCI).

    Args:
        df (pandas.DataFrame): DataFrame containing the true follow-up diagnosis (FollowupDX), predicted probabilities (Pred),
            and sample weights (SampleWeight) for each subject at each follow-up year and baseline diagnosis.

    Returns:
        dict: A dictionary containing the ROCAUC values for each follow-up year and baseline diagnosis.
    """

    # Initialize metrics dictionary
    rocaucs = {}

    # Loop through each baseline diagnosis (CN or MCI)
    for dx_bl, dx_bl_name in zip([0, 1], ['CN', 'MCI']):
        # Filter dataframe by baseline diagnosis
        df_bl = df.loc[df['BaselineDX']==dx_bl]
        # Loop through each follow-up year
        for year in np.unique(df['FollowupYear']):
            # Filter dataframe by follow-up year
            year_df = df_bl.loc[df_bl['FollowupYear']==year]
            
            # Extract true follow-up diagnoses, predicted probabilities, and sample weights
            y_true = year_df['FollowupDX'].values
            y_pred = np.vstack(year_df['Pred'].values)
            
            if dx_bl == 0:
                y_pred_positive_class = y_pred[:, 1]
                y_true_binary = (y_true == 1).astype(int)
            elif dx_bl == 1:
                y_pred_positive_class = y_pred[:, 2]
                y_true_binary = (y_true == 2).astype(int)
            
            # Compute ROCAUC
            rocauc = roc_auc_score(y_true_binary, y_pred_positive_class)
            
            # Add ROCAUC value to metrics dictionary
            rocaucs['ROCAUC_'+dx_bl_name+'_'+str(int(year))] = rocauc
            
    return rocaucs
    
def save_plot(fpr, tpr, thresholds, year, rt, dx_bl_name):
    roc_auc = auc(fpr, tpr)
    
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    output_filename = 'roc_auc_curve.png'
    plt.savefig(output_filename)
    """
    
    #new_roc_data = pd.DataFrame({
        #f'{year}_fpr_{dx_bl_name}': fpr,
        #f'{rt}_tpr_{dx_bl_name}': tpr
    #})
    
    new_roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr
    })
    
    roc_storage[(year, dx_bl_name, rt)] = new_roc_data
    
    print('Stored ROC data for Y{} {} RT{}'.format(year, dx_bl_name, rt))
    
    """
    OUTPUT_FILENAME = 'aucs.csv'
    
    print("Writing data for:")
    print("Year: " + str(year))
    print("RT: " + str(rt))
    print("Baseline: " + str(dx_bl_name)) 

    if os.path.exists(OUTPUT_FILENAME):
        #print(f"File '{OUTPUT_FILENAME}' exists. Appending columns...")
        
        # Read the existing data
        existing_df = pd.read_csv(OUTPUT_FILENAME)
        
        # Concatenate the existing columns and the new columns horizontally (axis=1)
        # The column names 'fpr' and 'tpr' will be repeated.
        final_df = pd.concat([existing_df, new_roc_data], axis=1)
    else:
        #print(f"File '{OUTPUT_FILENAME}' not found. Creating new file...")
        final_df = new_roc_data

    # Overwrite the file with the full, merged DataFrame.
    # We use mode='w' (write) and header=True to ensure the full data,
    # including the repeated column titles, is saved correctly.
    final_df.to_csv(OUTPUT_FILENAME, index=False, mode='w', header=True)
    
    #print(f"\nSuccessfully saved plot to {OUTPUT_FILENAME}")
    """
    
def store_mean_rocs():
    averaged_results = {}

    for year in years:
        for baseline in baselines:
            # A. Collect all TPR columns for this Year/Baseline group (across all 20 RTs)
            tpr_collection = []
            for rt in rts:
                # Retrieve the dataframe from storage
                df = roc_storage[(year, baseline, rt)]
                # We only need the TPR column for averaging
                tpr_collection.append(df['tpr'])
                
            # B. Concatenate them side-by-side
            # Result is shape (101 rows, 20 columns)
            combined_tpr = pd.concat(tpr_collection, axis=1)
            
            # C. Calculate the Mean TPR across the columns (axis=1)
            mean_tpr = combined_tpr.mean(axis=1)
            
            # D. Create the final averaged DataFrame
            avg_roc_df = pd.DataFrame({
                'fpr': base_fpr,  # The standard X-axis
                'mean_tpr': mean_tpr
            })
            
            # Store the result
            averaged_results[(year, baseline)] = avg_roc_df
          
    first_key = list(averaged_results.keys())[0]
    
    master_df = pd.DataFrame({
        'FPR': averaged_results[first_key]['fpr']
    })
    
    sorted_keys = sorted(averaged_results.keys())
    
    for year, baseline in sorted_keys:
        # Define a descriptive column name
        col_name = f"TPR_Year{year}_{baseline}"
      
        # Extract the mean_tpr column from the specific dataframe
        tpr_values = averaged_results[(year, baseline)]['mean_tpr']
      
        # Add it to the master dataframe
        master_df[col_name] = tpr_values
      
    master_df.to_csv("aucs.csv", index=False)

def get_roc_curve(y_true, y_pred, dx_bl, year, rt, dx_bl_name):
    """
    Computes the ROC curve for a given diagnosis group (CN or MCI) and a given follow-up year.

    Args:
        y_true (numpy.ndarray): True follow-up diagnosis for each subject.
        y_pred (numpy.ndarray): Predicted probability of each subject belonging to each follow-up diagnosis group.
        dx_bl (int): Baseline diagnosis group (0 for CN, 1 for MCI).

    Returns:
        numpy.ndarray: Interpolated true positive rate (TPR) for the ROC curve at each false positive rate (FPR) value.
    """
    
    if dx_bl == 0:
        # For CN baseline diagnosis, use the probability of MCI diagnosis.
        y_pred = y_pred[:, 1] 
        y_true = (y_true == 1).astype(int)
        y_pred = 1 - y_pred
        y_true = 1 - y_true
    if dx_bl == 1:
        # For MCI baseline diagnosis, use the probability of AD diagnosis.
        y_pred = y_pred[:, 2]
        y_true = (y_true == 2).astype(int)
        
    # Compute the FPR, TPR, and thresholds for the ROC curve.
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)  

    # Interpolate the TPR values for a fixed set of FPR values (0 to 1 in increments of 0.01).
    mean_fpr = np.linspace(0, 1, 100)     
    interp_tpr = np.interp(mean_fpr, fpr, tpr)

    # Set the first TPR value to 0 to ensure the curve starts at (0, 0).
    interp_tpr[0] = 0.0
    
    # Scale TPR values to percentages.
    
    save_plot(mean_fpr, smooth_interpolate_tpr(fpr, tpr), thresholds, year, rt, dx_bl_name)
    
    #print(interp_tpr * 100)
    
    return interp_tpr * 100
    
def smooth_interpolate_tpr(fpr, tpr):
    unique_fpr, indices = np.unique(fpr, return_index=True)
    unique_tpr = tpr[indices]
    
    # Ensure the last point is exactly (1,1) if not present, to prevent cutoff
    if unique_fpr[-1] != 1:
        unique_fpr = np.append(unique_fpr, 1)
        unique_tpr = np.append(unique_tpr, 1)

    # 2. Initialize the PCHIP Interpolator
    # This creates a function `pchip_func` that represents the smooth curve
    pchip_func = PchipInterpolator(unique_fpr, unique_tpr)
    
    # 3. Generate the smooth points
    mean_fpr = np.linspace(0, 1, 100)     
    interp_tpr = pchip_func(mean_fpr)

    # --- SMOOTHING LOGIC END ---
    
    # Set start to 0 and clip to ensure no floating point errors exceed 0-1 range
    interp_tpr[0] = 0.0
    interp_tpr = np.clip(interp_tpr, 0, 1)
    
    print("Smoothing TPR")
    
    return interp_tpr


def get_roc_curves(df, rt):
    """
    Computes the ROC curves for each follow-up year and baseline diagnosis (CN or MCI).

    Args:
        df (pandas.DataFrame): DataFrame containing the true follow-up diagnosis (FollowupDX), predicted probabilities (Pred),
            and sample weights (SampleWeight) for each subject at each follow-up year and baseline diagnosis.

    Returns:
        dict: A dictionary containing the interpolated true positive rates (TPR) for each FPR value for each ROC curve.
    """

    # Initialize dictionary to store ROC curves.
    roc_curves = {}

    # Loop through each baseline diagnosis (CN or MCI)
    for dx_bl, dx_bl_name in zip([0, 1], ['CN', 'MCI']):
        # Filter dataframe by baseline diagnosis
        df_bl = df.loc[df['BaselineDX']==dx_bl]
        # Loop through each follow-up year
        for year in np.unique(df['FollowupYear']):
            # Filter dataframe by follow-up year
            year_df = df_bl.loc[df_bl['FollowupYear']==year]
            
            # Extract true follow-up diagnoses and predicted probabilities
            y_true = year_df['FollowupDX'].values
            y_pred = np.vstack(year_df['Pred'].values)
            
            # Compute the ROC curve for the given diagnosis group and follow-up year.
            roc_curves['ROC_'+dx_bl_name+'_'+str(int(year))] = [get_roc_curve(y_true, y_pred, dx_bl, year, rt, dx_bl_name)]
            
    return roc_curves
    
def save_aggregate_roc():
    store_mean_rocs()
    
    print("Saved master ROC file")

