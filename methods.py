import numpy as np
import pandas as pd
import os
from typing import Dict,Tuple,List
from sklearn.linear_model import LinearRegression



##################################
### Calcul d'erreurs et marge ####
##################################

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
        return (np.abs(variation)<=error_p(p/100,n,2.328)*100) & variation!=0
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

###########################
### Régression linéaire ###
###########################

def estimate_base(X,y,base_y,weighted=True) -> list:
    """Estime le nombre d'individu dans une base de sondage en fonction des données 
    La méthode la régression linéraire en sachant que coef[0]*X[0]+...+coef[n]*X[n]=y avec coef les coefficents à trouver

    Args:
        X (numpy.array): Contient les données dont l'on doit retourner les bases. Organisé en colonne pour chaque catégorie
        y (numpy.array): Contient la liste des valeurs total du groupe considéré soit a0*X[i,0]+...+a*nX[i,n]=y[i]
        base_y (int): Taille de la base de l'ensemble total

    Returns:
        list: liste des bases estimé pour chacune des colonnes de X tels que coef[0]*X[0]+...+coef[n]*X[n]=y
    """
    lr=LinearRegression( normalize=True,positive=True)
    if weighted:
        w=np.sum(np.square((X-X.mean(axis=1).reshape(-1,1)))/np.square(y.reshape(-1,1))+1,axis=1)
        w=np.nan_to_num(w,nan=1)
        lr.fit(X,y,w)
    else:
        lr.fit(X,y)
    new_base=np.floor( (lr.coef_/lr.coef_.sum())*base_y)
    if new_base.sum()< base_y:
        imax=np.argmax(new_base)
        new_base[imax]=new_base[imax]+base_y-new_base.sum()
    return new_base

def flatten_xy(df:pd.DataFrame,
                col_names:List[str],
                categorie_x:str,
                groupe_x:List[str],
                categorie_y:str,
                groupe_y:str,
                col_cat='Categorie',
                col_groupe='Groupe',
                col_nom='Nom')-> Tuple[np.array]:
    """Créé un numpy array x et y qui sera utilisé pour la régression à partir d'un Dataframe et des catégories et groupes concernés

    Args:
        df (DataFrame): Dataframe pour créer les donnéees
        col_names (List[str]) : Liste des colonnes pour les comparaisons
        categorie_x (str): Catégorie pour former le X
        groupe_x (List[str]): Liste des groupes pour former le x
        categorie_y (str): Catégorie pour former le y
        groupe_y (str): Groupe pour former le y
        col_cat (str): Nom de la colonne de catégorie
        col_groupe (str): Nom de la colonne de groupe
        col_nom (str): Nom de la colonne de Nom de personnalité
        col_date (str): Nom de la colonne de date

    Returns:
        Tuple[array]: Retourne un tuple contenant deux listes X et y
    """
    X=np.array([[]]*len(groupe_x))
    y=np.array([])
    for nom in df[col_nom].unique():
        df_nom=df[df[col_nom]==nom]
        sub_x=df_nom[(df_nom[col_cat]==categorie_x) & (df_nom[col_groupe].isin(groupe_x))]
        sub_y=df_nom[(df_nom[col_cat]==categorie_y) & (df_nom[col_groupe]==groupe_y)]
        try:
            if len(sub_x)!=0 and len(sub_y)!=0:
                X=np.append(X,sub_x[col_names].to_numpy(),axis=1)
                y=np.append(y,sub_y[col_names].to_numpy().flatten())
        except Exception:
            print(f"Col cat {categorie_x}")
            print(f"Col groupe {groupe_x}")
            print({c:c in groupe_x for c in df_nom[col_groupe].unique()})
            print(sub_x.columns)
            print(col_names)
            print(sub_x.shape)
            print(X.shape)
    return (X.T,y)

def fill_zero_base_values(base:np.array,min_base=0.05):
    """Move base values from non zeros to zeros values with respect to proportion.
        In case of ceil values not rounding to the total, transfert of values is taken from the largest proportion

    Args:
        base (np.array): percentages of base for each categorie
        min_value_base (int): value to use in case of zeros
    """
    min_value_base=int(np.floor(min_base*base.sum()))
    zero_index=np.where(base==0)[0]
    if len(zero_index)!=0:
        adjust_prop=base/base.sum()
        adjust_prop=np.floor(adjust_prop*min_value_base*len(zero_index))
        # In case ceil does not round to the expect value to transfert
        if adjust_prop.sum()<min_value_base*len(zero_index):
            diff=min_value_base*len(zero_index)-adjust_prop.sum()
            adjust_prop[np.argmax(adjust_prop)]=adjust_prop[np.argmax(adjust_prop)]+diff
        # Remove values
        adjust_prop=base-adjust_prop
        adjust_prop[adjust_prop==0.]=min_value_base
        return adjust_prop
    else:
        return base


def recompute_base(df,relations,columns,col_base='Base',col_cat='Categorie',col_groupe='Groupe',col_nom='Nom'):
    """Recalcule la base des groupes fournis dans 'relations' en utilisant les valeurs dans le dataframe

    Args:
        df (Dateframe): Dataframe dont nous devons recalculer les bases de sondages
        relations (Tuple): Liste de tuples donc chaque tuples est defini comme suit (catX,groupeX,catY,groupeY)
        columns (List): Liste des noms de colonnes servant pour le recalcul des bases
        col_base (str, optional): Nom de colonne pour la base. Defaults to 'Base'.
        col_cat (str, optional): Nom de colonne pour la catégorie. Defaults to 'Categorie'.
        col_groupe (str, optional): Nom de colonne pour le groupe. Defaults to 'Groupe'.
        col_nom (str, optional): Nom de colonne pour le nom de la personnalité. Defaults to 'Nom'.

    Returns:
        [type]: [description]
    """
    df_alter=df.copy()
    for r in relations:
        # print(f"Current Relation {r}")
        X,y=flatten_xy(df_alter,columns,r[0],r[1],r[2],r[3],col_cat=col_cat,col_groupe=col_groupe,col_nom=col_nom)
        base_y=df_alter[(df_alter[col_cat]==r[2]) & (df_alter[col_groupe]==r[3])][col_base].unique()[0]
        new_base=estimate_base(X,y,base_y)
        new_base=fill_zero_base_values(new_base)
        # print('{}\n{}'.format(r,new_base))
        for i in range(len(r[1])):
            df_alter.loc[(df_alter[col_cat]==r[0]) & (df_alter[col_groupe]==r[1][i]), 'Base']=new_base[i]
    return df_alter



#############################
### Chargement de données ###
#############################

def load_data(filepath,rename_cols={},evolution=True):
    filenames=[f.rstrip('.p') for f in os.listdir(filepath)]
    df=pd.read_pickle("{}/{}.p".format(filepath,filenames[0]))
    if len(rename_cols)>0:
        df=df.rename(columns=rename_cols)
    all_data=pd.DataFrame(columns=df.columns)
    all_data.Base=all_data.Base.astype(int)
    for fn in filenames:
        df=pd.read_pickle("{}/{}.p".format(filepath,fn))
        if len(rename_cols)>0:
            df=df.rename(columns=rename_cols)
        if df.Base.dtype==object:
            df.Base=df.Base.astype(int)
        if 'Date' not in df.columns:
            df['Date']=fn
        all_data=pd.concat([all_data,df],ignore_index=True)
        if evolution:
            in_margin,nb_lines=extract_in_margin(df['ST Favorable'],df['ST Défavorable'],df['Evolution ST Favorable'],df['Evolution ST Défavorable'],df.Base)

    all_data.Date=all_data.Date.astype('datetime64')
    return all_data

def load_data_perso(filepath,rename_cols={}):
    filenames=[f.rstrip('.p') for f in os.listdir(filepath)]
    df=pd.read_pickle("{}/{}.p".format(filepath,filenames[0]))
    all_data=pd.DataFrame(columns=df.columns)
    all_data.Base=all_data.Base.astype(int)
    for fn in filenames:
        df=pd.read_pickle("{}/{}.p".format(filepath,fn))
        if df.Base.dtype==object:
            df.Base=df.Base.astype(int)
        all_data=pd.concat([all_data,df],ignore_index=True)
    all_data.rename(columns=rename_cols,inplace=True)
    all_data['ST Favorable']=all_data['ST Favorable'].astype(float)
    all_data.Date=all_data.Date.astype('datetime64')
    return all_data

