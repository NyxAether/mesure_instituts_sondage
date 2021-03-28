import numpy as np
import pandas as pd
import os

def error_p(p,n=1000,Za=1.96):
    return np.sqrt(p*(1-p)/n)*Za

def margin_p(p,n=1000,Za=1.96) -> np.array:
    er=error_p(p,n,Za)
    return np.array([p-er,p+er])

def error_in_margin(p,n,variation,Za=1.96,double_margin=True):
    # Compute old values
    old_p=p-variation
    # Compute err
    err=error_p(p/100,n,Za)*100
    # Compute err old values
    old_err=error_p(old_p/100,n,Za)*100
    if double_margin:
        return (np.abs(variation)<=err+old_err) & variation!=0
    else:
        return (np.abs(variation)<=err) & variation!=0

def extract_in_margin(fav,defav,evol_fav,evol_defav,pop_size,double_margin=True):
    # Suppression des valeurs sans variation
    fav=fav[evol_fav!=0]
    defav=defav[evol_defav!=0]
    pop_size_fav=pop_size[evol_fav!=0]
    pop_size_defav=pop_size[evol_defav!=0]
    evol_fav=evol_fav[evol_fav!=0]
    evol_defav=evol_defav[evol_defav!=0]

    # Calcul des erreurs
    err_fav=error_p(fav/100,pop_size_fav)*100
    err_defav=error_p(defav/100,pop_size_defav)*100

    # Calcul des anciennes valeurs
    old_fav=fav-evol_fav
    old_defav=defav-evol_defav
    # Calcul des erreurs anciennes valeurs
    err_old_fav=error_p(old_fav/100,pop_size_fav)*100
    err_old_defav=error_p(old_defav/100,pop_size_defav)*100
    if double_margin:
        total_fav_in_margin=np.sum(err_fav+err_old_fav>=np.abs(evol_fav))
        total_defav_in_margin=np.sum(err_defav+err_old_defav>=np.abs(evol_defav))
    else:
        total_fav_in_margin=np.sum(err_fav>=np.abs(evol_fav))
        total_defav_in_margin=np.sum(err_defav>=np.abs(evol_defav))
    total_in_margin=total_fav_in_margin+total_defav_in_margin
    total_lines=len(fav)+len(defav)

    return (total_in_margin,total_lines)

def load_data(filepath):
    filenames=[f.rstrip('.p') for f in os.listdir(filepath)]
    df=pd.read_pickle("{}/{}.p".format(filepath,filenames[0]))
    all_data=pd.DataFrame(columns=df.columns)
    all_data.Base=all_data.Base.astype(int)
    for fn in filenames:
        df=pd.read_pickle("{}/{}.p".format(filepath,fn))
        if df.Base.dtype==object:
            df.Base=df.Base.astype(int)
        if 'Date' not in df.columns:
            df['Date']=fn
        all_data=pd.concat([all_data,df],ignore_index=True)
        in_margin,nb_lines=extract_in_margin(df['ST Favorable'],df['ST Défavorable'],df['Evolution ST Favorable'],df['Evolution ST Défavorable'],df.Base)
    all_data.Date=all_data.Date.astype('datetime64')
    return all_data

def load_data_perso(filepath):
    filenames=[f.rstrip('.p') for f in os.listdir(filepath)]
    df=pd.read_pickle("{}/{}.p".format(filepath,filenames[0]))
    all_data=pd.DataFrame(columns=df.columns)
    all_data.Base=all_data.Base.astype(int)
    for fn in filenames:
        df=pd.read_pickle("{}/{}.p".format(filepath,fn))
        if df.Base.dtype==object:
            df.Base=df.Base.astype(int)
        all_data=pd.concat([all_data,df],ignore_index=True)
    all_data.rename(columns={'Opinion positive':'ST Favorable'},inplace=True)
    all_data['ST Favorable']=all_data['ST Favorable'].astype(float)
    all_data.Date=all_data.Date.astype('datetime64')
    return all_data
