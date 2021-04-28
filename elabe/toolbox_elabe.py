import numpy as np
import pandas as pd
import camelot as km
import os


def get_categories(df):
    """Récupère toutes les catégories dans l'ordre du tableau (sauf ENSEMBLE)

    Args:
        df (pandas.DataFrame): le dataframe associé au tableau à récupérer
    """
    # On récupère tous les champs dans la première colonne qui contient les groupes et les catégories et on récupère ceux qui n'ont pas de valeurs de base
    return df[df[1]==''][0]

def max_interval(l,i):
    for v in l:
        if i <= v:
            return v

def extend_series_categorie(cats):
    indexes=cats.index
    return [cats.loc[max_interval(indexes,i)] for i in range(max(indexes)+1)]

def get_line_ensemble(df):
    iline=df[0][df[0].str.contains('Ensemble')].index[0]
    return df.loc[iline],iline


def build_one_page_df(nom,df,date,
                    cols=['tres_positif','positif','negatif','tres_negatif','nsp','total_positif','total_negatif'],
                    ignore_ensemble=False)->pd.DataFrame:
    """Génère un dataframe utilisant le template et le df contenant les données.
    Les noms de colonnes pour chaque colonne doit être fournis en argument.

    Args:
        nom (str): Nom de la personnalité pour la page étudiée
        df (pandas.DataFrame): Dataframe contenant les données
        cols (list, optional): Noms de colonnes de valeurs. Defaults to ['Très favorable','Plutôt favorable', 'ST Favorable', 'Plutôt défavorable', 'Très défavorable', 'ST Défavorable', 'Nsp', 'Evolution ST Favorable', 'Evolution ST Défavorable'].
        ignore_ensemble (boolean,optional): ignore la première ligne dans le tableau si True

    Returns:
        pd.DataFrame: Le dataframe de toutes les données d'une page d'un pdf de elabe
    """
    print(nom)
    result_df=pd.DataFrame(columns=['Nom','Categorie','Groupe','Base','Date']+cols)
    # Récupération de la ligne ensemble
    line_ens,index_deb=get_line_ensemble(df)
    # Filtre le dataframe à partir de la ligne ensemble exclue
    data=df.loc[index_deb+1:]
    # Récupération des catégories
    cats=extend_series_categorie(get_categories(data))
    # Nettoyage des categories
    data=data[data[1]!=''].reset_index(drop=True)
    # Ajout de la ligne d'ensemble
    result_df.loc[0]=[nom,'Ensemble','Ensemble',100,date]+line_ens[2:].to_list()
    # Parcours et stockage des données
    for i in range(len(data)):
        result_df.loc[i+1]=[nom,cats[i],data[0][i],100,date]+data.loc[i,2:].to_list()
    # Indication des types
    result_df=result_df.replace('','0')
    # Correction des types
    result_df['Base']=result_df['Base'].astype(float)
    result_df[cols]=result_df[cols].astype(float)
    result_df['Date']=result_df['Date'].astype('datetime64')
    return result_df

def build_df_personnalite(path:str,pages:list,nom:str,date:str)->pd.DataFrame:
    """Construit le dataframe complet pour la personnalité donnée avec les pages correspondantes

    Args:
        path (str): Chemin vers le fichier
        pages (list): listes de pages à filtrer pour extraire les données
        nom (str): nom de la personnalité
        date (str): date du sondage

    Returns:
        pandas.DataFrame: Données extraites sous la forme d'un DataFrame
    """
    data=km.read_pdf('{}/{}.pdf'.format(path,date), flavor='stream',pages='{}'.format(pages[0]))[0].df
    df=build_one_page_df(nom,data,date)
    for p in pages:
        data=km.read_pdf('{}/{}.pdf'.format(path,date), flavor='stream',pages='{}'.format(pages[0]))[0].df
        df_temp=build_one_page_df(nom,data,date,ignore_ensemble=True)
        df=pd.concat([df,df_temp],axis=0,ignore_index=True)
    return df

# tables=km.read_pdf('elabe/rapport/2021-04-07.pdf', flavor='stream',pages='12')
# data=tables[0].df
# data
# extend_series_categorie(get_categories(data))
# # template=build_one_page_template_df(data,'2018-12-08')
# build_df_personnalite('elabe/rapport',[11,12],'MAcron','2021-04-07')

