# Estudo sobre MLFlow e DVC

Projeto de classificação do dataset **Wine** com **DVC** (pipeline e versionamento de dados) + **MLflow** (rastreamento de experimentos) integrado ao **DagsHub**.

## Objetivo

Construir um pipeline reprodutível de ML com duas etapas:

1. **Processamento de dados**  
2. **Treinamento e rastreabilidade de modelo**

## Estrutura do repositório

- Pipeline DVC: [dvc.yaml](dvc.yaml)
- Parâmetros de treino: [wine_params.yaml](wine_params.yaml)
- Dependências Python: [requirements.txt](requirements.txt)
- Processamento: [src/processa.py](src/processa.py)
- Treino + MLflow: [src/train.py](src/train.py)
- Scripts de testes/experimentos:
  - [testes/teste1.py](testes/teste1.py)
  - [testes/teste2.py](testes/teste2.py)
  - [testes/teste3.py](testes/teste3.py)
- Configuração DVC remota: [.dvc/config](.dvc/config)

## Como o pipeline funciona

O pipeline está definido em [dvc.yaml](dvc.yaml) com os stages:

### 1) `processa`
- Comando: `python src/processa.py`
- Script principal: [`src.processa.main`](src/processa.py)
- Funções relevantes:
  - [`src.processa.import_raw_data`](src/processa.py): carrega dataset `load_wine()`
  - [`src.processa.process_data`](src/processa.py): remove nulos e duplicatas
  - [`src.processa.salvar_dados`](src/processa.py): salva CSVs
- Saídas:
  - `data/raw/wine_raw.csv`
  - `data/processed/wine_processado.csv`

### 2) `treina`
- Comando: `python src/train.py`
- Script principal: [`src.train.main`](src/train.py)
- Funções relevantes:
  - [`src.train.importar_parametros`](src/train.py): lê [wine_params.yaml](wine_params.yaml)
  - [`src.train.treinar_modelo`](src/train.py): treina RandomForest, loga métricas/artefatos no MLflow
- Dependências:
  - `data/processed/wine_processado.csv`
  - [src/train.py](src/train.py)
  - [wine_params.yaml](wine_params.yaml)

## Rastreamento no MLflow / DagsHub

No treino ([src/train.py](src/train.py)):

- Inicializa integração com DagsHub (`dagshub.init`)
- Usa tracking URI:
  - `https://dagshub.com/guifzy/mlflow_teste.mlflow`
- Loga:
  - parâmetros (`max_depth`, `n_estimators`)
  - métrica (`accuracy`)
  - artefatos (matriz de confusão + código-fonte)
  - modelo (`mlflow.sklearn.log_model`)
  - metadados de reprodutibilidade:
    - commit Git
    - hash MD5 do dado via `dvc.lock` (stage `processa`)

## Como reproduzir localmente

## Pré-requisitos
- Python 3.10+ (idealmente 3.12)
- Git
- DVC
- Conta/autenticação no DagsHub (para push remoto e tracking)

## Passos

1. **Clonar repositório**
```bash
git clone https://dagshub.com/guifzy/mlflow_teste.git
cd mlflow_teste
```

2. **Criar e ativar ambiente virtual**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Instalar dependências**
```bash
pip install -r requirements.txt
```

4. **(Opcional) puxar cache de dados do DVC**
```bash
dvc pull
```

5. **Executar pipeline completo**
```bash
dvc repro
```

6. **Visualizar DAG do pipeline**
```bash
dvc dag
```

## Alterar hiperparâmetros

Edite [wine_params.yaml](wine_params.yaml):

```yaml
train:
  max_depth: 5
  n_estimators: 50
```

Depois rode novamente:

```bash
dvc repro
```

## Como abrir no DagsHub

- Repositório:  
  https://dagshub.com/guifzy/mlflow_teste

- UI de experimentos MLflow:  
  https://dagshub.com/guifzy/mlflow_teste.mlflow

No DagsHub, abra:
1. **Repository Files** para código e versionamento.
2. **Experiments / MLflow** para runs, métricas, parâmetros e artefatos.
3. **Data / DVC** para histórico e versionamento dos dados.

## Scripts de experimento (pasta `testes/`)

- [testes/teste1.py](testes/teste1.py): treino simples com tracking URI via `.env`
- [testes/teste2.py](testes/teste2.py): treino simples com `dagshub.init`
- [testes/teste3.py](testes/teste3.py): `GridSearchCV` + runs aninhadas no MLflow

Esses scripts são úteis para testes rápidos, enquanto o fluxo principal reprodutível está no pipeline DVC ([dvc.yaml](dvc.yaml)).

## Observações

- Arquivos de dados em `data/` estão ignorados no Git em [.gitignore](.gitignore).
- Configuração do remoto DVC está em [.dvc/config](.dvc/config).