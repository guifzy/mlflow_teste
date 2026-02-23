import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import dagshub
from sklearn.model_selection import GridSearchCV

dagshub.init(repo_owner="guifzy", repo_name="mlflow_teste", mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/guifzy/mlflow_teste.mlflow")  # Configura o MLflow para usar o DagsHub como backend

wine = load_wine()
X = wine.data
y = wine.target

# divisoes do dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# parametros do modelo
parametros = {
    'max_depth': [5, 10, 15, None],
    'n_estimators': [10, 50, 100, 200]
}

mlflow.set_experiment("Wine_Classification_teste3")

with mlflow.start_run() as main_run:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=parametros, cv=kf)
    grid.fit(X_train, y_train)

    for i in range(len(grid.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child_run:
            mlflow.log_params(grid.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid.cv_results_['mean_test_score'][i])

    best_params = grid.best_params_
    best_score = grid.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)

    y_pred = grid.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log dos parametros e metricas do modelo para acompanhamento e comparação futura
    mlflow.log_param("max_depth", best_params['max_depth'])
    mlflow.log_param("n_estimators", best_params['n_estimators'])
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
    mlflow.sklearn.log_model(grid, "random_forest_model") # log do modelo treinado
    mlflow.log_artifact(__file__)  # Log do código fonte do test

    print(f"Accuracy: {acc}")
