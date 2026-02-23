from sklearn.datasets import load_wine
import pandas as pd 
import numpy as np
import os

def import_raw_data():
    wine = load_wine()
    X = wine.data
    y = wine.target
    df = pd.DataFrame(X, columns=wine.feature_names)
    df['target'] = y

    return df

def process_data(df):
    if df.isnull().sum().any():
        df = df.dropna()  

    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()  

    return df

def salvar_dados(df, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    try:
        df.to_csv(path, index=False)
        print(f"Dados salvos com sucesso em {path}")
    except Exception as e:
        print(f"Erro ao salvar dados em {path}: {e}")

def main():
    df = import_raw_data()
    df_processado = process_data(df)
    #salva dados crus
    salvar_dados(df, r'data\raw\wine_raw.csv')
    salvar_dados(df_processado, r'data\processed\wine_processado.csv')

if __name__ == "__main__":
    main()