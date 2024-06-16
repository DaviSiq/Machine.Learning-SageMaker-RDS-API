# Machine Learning com SageMaker, RDS e API Docker

Este repositório é um estudo e implementação de um projeto completo de Machine Learning, desde o treinamento local com scikit-learn até a implantação (e deploy) de modelos usando AWS SageMaker , com dados do AWS RDS e uma API para inferência em Docker.

## Descrição

- **Treinamento Local**: Scripts para treinar modelos usando scikit-learn (`Random Forest`, `XGboost`).
- **SageMaker**: Treinamento de modelos com AWS SageMaker usando dados do AWS RDS, salvando o modelo no S3.
- **API de Inferência**: API em Python (Flask/FastAPI) em Docker para carregar o modelo do S3 e realizar predições via `/api/v1/predict`.
## Uso

### Treinamento Local

```sh
Para ver dados referentes ao treinamento local, acesse o Readme de 'treino_local'.
# Execute os scripts de treinamento
# Machine.Learning-SageMaker-RDS-API