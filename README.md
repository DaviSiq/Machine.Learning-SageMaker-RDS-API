# Machine Learning com SageMaker e API Docker

Este repositório é um estudo e implementação de um projeto de Machine Learning, desde o treinamento local com scikit-learn até a implantação de modelos usando AWS SageMaker, com dados do AWS S3 e uma API para inferência em Docker.

## Descrição

- **Treinamento Local**: Scripts para treinar modelos usando `scikit-learn` (`Random Forest`, `XGBoost`).
- **SageMaker**: Treinamento de modelos com AWS SageMaker usando dados do AWS S3, salvando o modelo no S3.
- **API de Inferência**: API em Python (Flask/FastAPI) em Docker para carregar o modelo do S3 e realizar predições via `/api/v1/predict`.
### Aplicação rodando localmente
![Print da Aplicação](print_local.jpg)

## Estrutura do Projeto

- `python/models/`: Scripts para treinamento local e modelo salvo para inferência posterior.
- `api/docker/`: Arquivos de configuração para a API em Docker.
- `sagemaker/`: Notebooks e scripts para treinamento e implantação no SageMaker.
- `api/docker/main.py`: Código da API.

## Uso
Incompleto por enquanto, a inferência está configurada manualmente no arquivo da API, basta executar `main.py`
### Treinamento Local

Para detalhes sobre o treinamento local, consulte o README na pasta `models`.

### Treinamento com SageMaker

1. Configure o ambiente AWS.
2. Use os notebooks na pasta `sagemaker/` para treinar e implantar modelos.
3. O modelo treinado será salvo no S3.

### API de Inferência
Em construção...
