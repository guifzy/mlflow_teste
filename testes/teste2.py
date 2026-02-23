import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from dotenv import load_dotenv
import dagshub

dagshub.init(repo_owner="guifzy", repo_name="mlflow_teste", mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/guifzy/mlflow_teste.mlflow")  # Configura o MLflow para usar o DagsHub como backend

wine = load_wine()
X = wine.data
y = wine.target

# divisoes do dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# parametros do modelo
max_depth = 12
n_estimators = 6

mlflow.set_experiment("Wine_Classification_teste2")

with mlflow.start_run():
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log dos parametros e metricas do modelo para acompanhamento e comparação futura
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.savefig("confusion_matrix.png")
    # Log da matriz de confusão como artefato para visualização posterior
    mlflow.log_artifact("confusion_matrix.png") 

    # Log de tags para melhor organização e busca dos experimentos
    mlflow.set_tags({"model_type": "RandomForest", "dataset": "WineDataSet", "author": "Guilherme Monteiro", "project": "MLflow Teste"})
    mlflow.sklearn.log_model(model, "random_forest_model") # log do modelo treinado
    mlflow.log_artifact(__file__)  # Log do código fonte do test

    print(f"Accuracy: {acc}")
