from typing import List, Union
import operator

import pandas as pd
import seaborn as sns
import numpy as np


def get_all_features_same_id(data_df:pd.DataFrame, feature_id:int):
    """
        This function takes a feature ID and returns all entries from that ID.
        This is useful for IDs that have several entries like 41270
    """
    filtered_features = data_df.filter(regex=f"{feature_id}-*|eid")
    eids = filtered_features["eid"]
    del filtered_features["eid"]

    filtered_features_values = list(filtered_features.values)
    a = []
    for subject_list in filtered_features_values:
        a.append([x for x in subject_list if str(x) != "nan"])

    return eids, a

def check_coverage(data_df:pd.DataFrame, field_ids:List[str], short:bool=False):
    """
        Take a list of field ids and count the number of subjects that have a non-NA value for every field.
        The short argument prevents the printing of the field ids that were searched for and just prints the count.
        Prints count and returns list of indices of rows where condition was fufilled so that the data frame can be filtered.
    """
    notna_rows = data_df[field_ids].notna().all(axis='columns')
    if short:
        print('Out of {} subjects, {} have all desired fields.'.format(len(data_df),sum(notna_rows)))
    else:
        print('Out of {} subjects, {} have all desired fields ({}).'.format(len(data_df),sum(notna_rows),','.join(field_ids)))
    return notna_rows

def plot_hist(data_df:pd.DataFrame, field_id:str, datadict_df:pd.DataFrame=None):
    """
        Quickly print histogram of a field. Parses and sets the x axis name from the data dictionary if the field ids are still in integer format.
    """
    if field_id[0].isnumeric():
        field_name = get_name(field_id, datadict_df)
    else:
        field_name = field_id.split('-')[0]
    if data_df[field_id].dtype=='bool':
        data_df[field_id] = data_df[field_id].astype(float)
    ax = sns.histplot(data=data_df,x=field_id)
    ax.set(xlabel=field_name)
    coverage = data_df[field_id].notna().sum()/len(data_df)*100
    print(f'Coverage of field is {coverage:.2f}%')

def get_name(field_id:str, datadict_df:pd.DataFrame):
    """
        Used when field id is still in original int format to grab the full string version from the data dictionary.
    """
    name = datadict_df[datadict_df['FieldID']==int(field_id.split('-')[0])]['Field'].iloc[0]
    return name

def grab_sorted_values(data_df:pd.DataFrame, field_id:str, descending:bool=True):
    """
        Grabs all non NA values in df of field id, sorts them (default descending order) and returns the sorted values.
        Used to examine outliers. Useful for determining cutoffs when removing outliers.
    """
    values = list(data_df[~data_df[field_id].isna()][field_id])
    values.sort(reverse=descending)
    return values

def remove_outliers(data_df:pd.DataFrame, field_id:str, limit:float, greater:bool=True):
    """
        Sets values above/below a defined limit to NA of the specified field. 
        The sorted values should have already been calculated so the limit could be defined and can be passed to prevent double calculation. Will be recalculated if not passed.
    """
    if greater:
        data_df.loc[data_df[field_id]>limit,field_id] = pd.NA
    else:
        data_df.loc[data_df[field_id]<limit,field_id] = pd.NA
    
def calc_and_save_mean(data_df:pd.DataFrame, field_id:str):
    """
        Takes a field with multiple array values, calculates the mean of all array values and creates a new column for that mean. 
        NOTE: Field id provided must point to first item in array. (i.e. 1234-2.0)
    """
    all_recording_ids = []
    cols = data_df.columns
    while field_id in cols:
        all_recording_ids.append(field_id)
        split = field_id.split('.')
        field_id = f'{split[0]}.{int(split[1])+1}'
    mean_id = f'{split[0]}.mean'
    data_df[mean_id] = data_df[all_recording_ids].mean(axis=1)

def rename(data_df:pd.DataFrame, datadict_df:pd.DataFrame):
    """
        Grabs all the names for each integer field id using the datadict and renames the columns from integer ids to names. 
        NOTE: Make sure no double names are in datadict (i.e. BMI) before calling this.
    """
    rename = {}
    cols = [c for c in data_df.columns if not c.startswith('eid') and not c.startswith('age')]
    for c in cols:
        new_name=f'{get_name(c, datadict_df)}-{c.split("-")[1]}'
        rename[c] = new_name
    data_df.rename(columns=rename, inplace=True)

def update_through_age(data_df:pd.DataFrame, diag_id:str, age_id:str):
    """
        If there exists a field that specifies an age when something happened (i.e. diagnosis), use the presence of a value in the age field as valid evidence that the thing (diagnosis) happened. 
        This was written because some subjects specified an age for a diagnosis but didn't have that diagnosis recorded. This allows us to update the diagnosis field through the age field.
    """
    diagnosed_through_age = data_df[age_id].notna()
    data_df.loc[:,diag_id] = (diagnosed_through_age | data_df[diag_id])

def check_answer(data_df:pd.DataFrame, answer:int, field_name:str, instance_array_size:int, option_array_size:int, agnostic_field:str=None):
    """
        Goes through all instances and array values for each instance to see if the subject ever indicated the searched for answer. Returns the array of boolean values.
        If there is a separate field that is instance agnostic (that has been calculated separately), that field is checked first.
    """
    superset = pd.Series([False for i in range(len(data_df))])
    for i in range(instance_array_size):
        for j in range(option_array_size):
            field_id = f'{field_name}-{i}.{j}'
            superset = (superset | (data_df[field_id]==answer))
    if agnostic_field:
        superset = (superset | data_df[agnostic_field])
    return superset

def check_answer_single_visit(data_df:pd.DataFrame, answer:int, visit:int, field_name:str, option_array_size:int, agnostic_field:str=None):
    """
        Goes through all instances and array values for each instance to see if the subject ever indicated the searched for answer. Returns the array of boolean values.
        If there is a separate field that is instance agnostic (that has been calculated separately), that field is checked first.
    """
    superset = pd.Series([False for i in range(len(data_df))])
    for j in range(option_array_size):
        field_id = f'{field_name}-{visit}.{j}'
        superset = (superset | (data_df[field_id]==answer))
    if agnostic_field:
        superset = (superset | data_df[agnostic_field])
    return superset

def operations_performed(data_df:pd.DataFrame, field_name:str, operation_codes:List[str]):
    """
        Counts the occurrences of the selected operation codes across all instances. Returns the array of counts.
    """
    operations = [0 for i in range(len(data_df))]
    for i in range(3):
        for j in range(32):
            field_id = f'{field_name}-{i}.{j}'
            operations = operations + data_df[field_id].isin(operation_codes)
    return operations

# The following functions are to be used when converting from pandas dataframe to a vector usable for DL

def one_hot_encode(value:int, num_classes:int, one_based:bool=False):
    """
        Makes a one hot encoding of an integer categorical variable.
        It is assumed the values start at 0. Some start at one and thus the one_based flag should be used.
    """
    if pd.isna(value):
        vec = np.full([num_classes], np.nan)
    else:
        value = int(value)
        if one_based:
            vec = np.eye(num_classes, dtype=int)[value-1]
        else:
            vec = np.eye(num_classes, dtype=int)[value]
    return vec

def clean_categorical(value:int):
    """
        Ensures nans are properly saved and categorical variables are ints.
    """
    if pd.isna(value):
        return np.nan
    else:
        return int(value)

def expand_list(data_df:pd.DataFrame, label_id:str):
    """
        If a fields values are lists, we need a new field per list entry. A new dataframe is created so the original df needs to be overwritten in the main code.
    """
    num_classes = len(data_df[label_id].iloc[0])
    cols = [f'{label_id}-{i}' for i in range(num_classes)]
    data_expanded_df = pd.concat([data_df, pd.DataFrame(data_df[label_id].tolist(), columns=cols)], axis=1)
    data_expanded_df = data_expanded_df.drop(label_id, axis=1)
    return data_expanded_df

def cardiac_features_to_vector_df(df):
    vec = []
    vec.append(df['eid'])
    vec.append(df['eid_old'])
    vec.append(df['Pulse wave Arterial Stiffness index-2.0'])
    vec.append(df['Systolic blood pressure-2.mean'])
    vec.append(df['Diastolic blood pressure-2.mean'])
    vec.append(df['Pulse rate-2.mean'])
    vec.append(df['Body fat percentage-2.0'])
    vec.append(df['Whole body fat mass-2.0'])
    vec.append(df['Whole body fat-free mass-2.0'])
    vec.append(df['Whole body water mass-2.0'])
    vec.append(df['Body mass index (BMI)-2.0'])
    vec.append(df['Cooked vegetable intake-2.0'])
    vec.append(df['Salad / raw vegetable intake-2.0'])
    vec.append(df['Cardiac operations performed'])
    vec.append(df['Total mass-2.0'])
    vec.append(df['Basal metabolic rate-2.0'])
    vec.append(df['Impedance of whole body-2.0'])
    vec.append(df['Waist circumference-2.0'])
    vec.append(df['Hip circumference-2.0'])
    vec.append(df['Standing height-2.0'])
    vec.append(df['Height-2.0'])
    vec.append(df['Sitting height-2.0'])
    vec.append(df['Weight-2.0'])
    vec.append(df['Ventricular rate-2.0'])
    vec.append(df['P duration-2.0'])
    vec.append(df['QRS duration-2.0'])
    vec.append(df['PQ interval-2.0'])
    vec.append(df['RR interval-2.0'])
    vec.append(df['PP interval-2.0'])
    vec.append(df['Cardiac output-2.0'])
    vec.append(df['Cardiac index-2.0'])
    vec.append(df['Average heart rate-2.0'])
    vec.append(df['Body surface area-2.0'])
    vec.append(df['Duration of walks-2.0'])
    vec.append(df['Duration of moderate activity-2.0'])
    vec.append(df['Duration of vigorous activity-2.0'])
    vec.append(df['Time spent watching television (TV)-2.0'])
    vec.append(df['Time spent using computer-2.0'])
    vec.append(df['Time spent driving-2.0'])
    vec.append(df['Time spent driving-2.0'])
    vec.append(df['Heart rate during PWA-2.0'])
    vec.append(df['Systolic brachial blood pressure during PWA-2.0'])
    vec.append(df['Diastolic brachial blood pressure during PWA-2.0'])
    vec.append(df['Peripheral pulse pressure during PWA-2.0'])
    vec.append(df['Central systolic blood pressure during PWA-2.0'])
    vec.append(df['Central pulse pressure during PWA-2.0'])
    vec.append(df['Number of beats in waveform average for PWA-2.0'])
    vec.append(df['Central augmentation pressure during PWA-2.0'])
    vec.append(df['Augmentation index for PWA-2.0'])
    vec.append(df['Cardiac output during PWA-2.0'])
    vec.append(df['End systolic pressure during PWA-2.0'])
    vec.append(df['End systolic pressure index during PWA-2.0'])
    vec.append(df['Total peripheral resistance during PWA-2.0'])
    vec.append(df['Stroke volume during PWA-2.0'])
    vec.append(df['Mean arterial pressure during PWA-2.0'])
    vec.append(df['Cardiac index during PWA-2.0'])
    vec.append(df['Sleep duration-2.0'])
    vec.append(df['Exposure to tobacco smoke at home-2.0'])
    vec.append(df['Exposure to tobacco smoke outside home-2.0'])
    vec.append(df['Pack years of smoking-2.0'])
    vec.append(df['Pack years adult smoking as proportion of life span exposed to smoking-2.0'])
    vec.append(df['LVEDV (mL)'])
    vec.append(df['LVESV (mL)'])
    vec.append(df['LVSV (mL)'])
    vec.append(df['LVEF (%)'])
    vec.append(df['LVCO (L/min)'])
    vec.append(df['LVM (g)'])
    vec.append(df['RVEDV (mL)'])
    vec.append(df['RVESV (mL)'])
    vec.append(df['RVSV (mL)'])
    vec.append(df['RVEF (%)'])
    
    vec.append(df['Worrier / anxious feelings-2.0'].apply(clean_categorical))
    vec.append(df['Shortness of breath walking on level ground-2.0'].apply(clean_categorical))
    vec.append(df['Sex-0.0'].apply(clean_categorical))
    vec.append(df['Diabetes diagnosis'].apply(clean_categorical))
    vec.append(df['Heart attack diagnosed by doctor'].apply(clean_categorical))
    vec.append(df['Angina diagnosed by doctor'].apply(clean_categorical))
    vec.append(df['Stroke diagnosed by doctor'].apply(clean_categorical))
    vec.append(df['High blood pressure diagnosed by doctor'].apply(clean_categorical))
    vec.append(df['Cholesterol lowering medication regularly taken'].apply(clean_categorical))
    vec.append(df['Blood pressure medication regularly taken'].apply(clean_categorical))
    vec.append(df['Insulin medication regularly taken'].apply(clean_categorical))
    vec.append(df['Hormone replacement therapy medication regularly taken'].apply(clean_categorical))
    vec.append(df['Oral contraceptive pill or minipill medication regularly taken'].apply(clean_categorical))
    vec.append(df['Pace-maker-2.0'].apply(clean_categorical))
    vec.append(df['Ever had diabetes (Type I or Type II)-0.0'].apply(clean_categorical))
    vec.append(df['Long-standing illness, disability or infirmity-2.0'].apply(clean_categorical))
    vec.append(df['Tense / \'highly strung\'-2.0'].apply(clean_categorical))
    vec.append(df['Ever smoked-2.0'].apply(clean_categorical))

    vec.append(df['Sleeplessness / insomnia-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3, one_based=True)))
    vec.append(df['Frequency of heavy DIY in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=7)))
    vec.append(df['Alcohol intake frequency.-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6, one_based=True)))
    vec.append(df['Processed meat intake-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Beef intake-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Pork intake-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Lamb/mutton intake-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Overall health rating-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=4, one_based=True)))
    vec.append(df['Alcohol usually taken with meals-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Alcohol drinker status-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Frequency of drinking alcohol-0.0'].apply(lambda col: one_hot_encode(value=col, num_classes=5)))
    vec.append(df['Frequency of consuming six or more units of alcohol-0.0'].apply(lambda col: one_hot_encode(value=col, num_classes=5, one_based=True)))
    vec.append(df['Amount of alcohol drunk on a typical drinking day-0.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6, one_based=True)))
    vec.append(df['Falls in the last year-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3, one_based=True)))
    vec.append(df['Weight change compared with 1 year ago-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Number of days/week walked 10+ minutes-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Number of days/week of moderate physical activity 10+ minutes-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Number of days/week of vigorous physical activity 10+ minutes-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Usual walking pace-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3, one_based=True)))
    vec.append(df['Frequency of stair climbing in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Frequency of walking for pleasure in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=7)))
    vec.append(df['Duration walking for pleasure-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Frequency of strenuous sports in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=7)))
    vec.append(df['Duration of strenuous sports-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Duration of light DIY-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Duration of heavy DIY-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Frequency of other exercises in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=7)))
    vec.append(df['Duration of other exercises-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Current tobacco smoking-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Past tobacco smoking-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=4, one_based=True)))
    vec.append(df['Smoking/smokers in household-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Smoking status-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    return vec