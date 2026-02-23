import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import pandas as pd
import os 
import yaml
import subprocess

# parametros do modelo
def importar_parametros(caminho):
    with open(caminho, 'r') as file:
        parametros = yaml.safe_load(file)
    return parametros

def treinar_modelo(X_train, y_train, X_test, y_test, caminho_db):
    # encontando o commit atual do Git nesta run
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    # encontrando o hash do dataset atual utilizado nesta run
    with open("dvc.lock", 'rb') as f:
        deps = yaml.safe_load(f)['stages']['train']['deps']  
    for dep in deps:
        if dep['path'] == caminho_db:  
            dvc_md5 = dep['md5']  

    # Experimento atual
    mlflow.set_experiment(f"Wine_Classification_{commit}")  

    with mlflow.start_run():
        # Parametros de rastreamento do modelo
        mlflow.log_param("commit", commit)
        mlflow.log_param("dvc_md5", dvc_md5)

        parametros = importar_parametros(r'wine_params.yaml')
        modelo = RandomForestClassifier(random_state=42, max_depth=parametros["max_depth"], n_estimators=parametros["n_estimators"])
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("max_depth", parametros["max_depth"])
        mlflow.log_param("n_estimators", parametros["n_estimators"])
        mlflow.log_metric("accuracy", acc)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log dos parametros e metricas do modelo para acompanhamento e comparação futura
        mlflow.log_param("max_depth", parametros["max_depth"])
        mlflow.log_param("n_estimators", parametros["n_estimators"])
        mlflow.log_metric("accuracy", acc)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        plt.savefig("confusion_matrix.png")
        # Log da matriz de confusão como artefato para visualização posterior
        mlflow.log_artifact("confusion_matrix.png") 
        os.remove("confusion_matrix.png")  # Remove o arquivo local após o log para manter o ambiente limpo

        # Log de tags para melhor organização e busca dos experimentos
        mlflow.set_tags({"model_type": "RandomForest", "dataset": "WineDataSet", "author": "Guilherme Monteiro", "project": "MLflow Teste"})
        mlflow.sklearn.log_model(modelo, "random_forest_model") # log do modelo treinado
        mlflow.log_artifact(__file__)  # Log do código fonte do test

        print(f"Accuracy: {acc}")

def main():
    dagshub.init(repo_owner="guifzy", repo_name="mlflow_teste", mlflow=True)

    mlflow.set_tracking_uri("https://dagshub.com/guifzy/mlflow_teste.mlflow")  # Configura o MLflow para usar o DagsHub como backend

    caminho_db = r'data\raw\wine.csv'
    df = pd.read_csv(caminho_db)
    X = df.drop('target', axis=1)
    y = df['target']

    # divisoes do dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    treinar_modelo(X_train, y_train, X_test, y_test, caminho_db)

if __name__ == "__main__":
    main()