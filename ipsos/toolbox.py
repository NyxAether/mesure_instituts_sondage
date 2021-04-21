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
    return df[df[1]!=''][0]

def build_one_page_template_df(df,date):
    """Construit le template pour récupérer les données ensuite. Il contient les groupes, les catégories et les bases et la date

    Args:
        df (pandas.DataFrame): le dataframe associé au tableau à récupérer
    """
    # Suppression des lignes inutiles potientiellement au début
    data=df[df[0]!=''].reset_index(drop=True)
    template=pd.DataFrame(columns=['Categorie','Groupe','Base','Date'])
    # Nettoyage des groupes en enlevant les points
    cat_group=data[0].str.replace(" ?[ ,\n]?\.\.+$",'')
    # Nettoyage des espace des bases
    base=data[1].str.replace(' ','')
    # Ajout de la première ligne de l'ensemble
    template.loc[0]=['ENSEMBLE','ENSEMBLE',base[0],date]
    current_categorie='ENSEMBLE'
    for i in range(1,len(data)):
        if base[i]=='':
            # La categorie change si il n'y a pas de valeur de base. Le nom de la catégorie est indiqué dans la première colonne avec les autres groupes.
            current_categorie=cat_group[i]
        else:
            # Sinon il s'agit juste d'un nouveau groupe
            template.loc[i]=[current_categorie,cat_group[i],base[i],date]
    template=template.reset_index(drop=True)
    return template

def build_one_page_df(nom,template,df,
                    cols=['Très favorable','Plutôt favorable', 'ST Favorable',
                    'Plutôt défavorable', 'Très défavorable', 'ST Défavorable',
                    'Nsp', 'Evolution ST Favorable', 'Evolution ST Défavorable'])->pd.DataFrame:
    """Génère un dataframe utilisant le template et le df contenant les données.
    Les noms de colonnes pour chaque colonne doit être fournis en argument.

    Args:
        nom (str): Nom de la personnalité pour la page étudiée
        template (pandas.DataFrame): dataframe servant de template
        df (pandas.DataFrame): Dataframe contenant les données
        cols (list, optional): Noms de colonnes de valeurs. Defaults to ['Très favorable','Plutôt favorable', 'ST Favorable', 'Plutôt défavorable', 'Très défavorable', 'ST Défavorable', 'Nsp', 'Evolution ST Favorable', 'Evolution ST Défavorable'].

    Returns:
        pd.DataFrame: Le dataframe de toutes les données d'une page d'un pdf de ipsos
    """
    print(nom)
    one_page_df=template.copy()
    one_page_df.replace(',','.',regex=True)
    # Suppression des lignes sans données
    data=df[df[1]!='']
    data=data[data[0]!=''].reset_index(drop=True)
    # Enlève les données déjà récupérées (groupe catégorie et base)
    data=data[range(2,len(data.columns))] 
    # Remplacement des virgules en point
    for label,serie in data.items():
        data[label]=serie.str.replace(',','.')
    # Indication des types
    data=data.replace('','0').astype(float)
    # Fusion des données
    one_page_df=pd.concat([one_page_df,data],axis=1) 
    # Changement des noms de colonnes
    map_col=dict(zip(range(2,len(one_page_df.columns)),cols))
    one_page_df=one_page_df.rename(map_col,axis=1)
    # Ajout du nom
    one_page_df['Nom']=nom
    # Réageancement des colonnes
    one_page_df=one_page_df[['Nom']+list((one_page_df.columns[:-1]))]
    # Correction des types
    one_page_df['Base']=one_page_df['Base'].astype(float)
    one_page_df['Date']=one_page_df['Date'].astype('datetime64')
    return one_page_df

# tables=km.read_pdf('rapport/2018-12-08.pdf', flavor='stream',pages='18')
# data=tables[0].df
# template=build_one_page_template_df(data,'2018-12-08')
# build_one_page_df('Martine',template,data)

