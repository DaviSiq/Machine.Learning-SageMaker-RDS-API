import argparse
import os
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == '__main__':
    # Parsear argumentos enviados pelo estimator do SageMaker
    parser = argparse.ArgumentParser()

    # Hiperparâmetros enviados pelo SageMaker
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    # Argumentos de ambiente do SageMaker
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Carregar os dados de treinamento do S3
    train_data = pd.read_csv(os.path.join(args.train, 'modified_hotel_reservations.csv'))
    train_data = train_data.drop(columns=['Booking_ID', 'booking_status'])
    # Separar as features e o label
    X = train_data.drop('label_avg_price_per_room', axis=1)
    y = train_data['label_avg_price_per_room']

    # Identificar colunas categóricas e numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Criar os transformers para os pipelines de pré-processamento
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Criar o pré-processador com ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Criar o pipeline incluindo o pré-processador e o modelo
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=500,
            random_state=42))
        ])

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Prever nos dados de teste e calcular a acurácia
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Salvar o modelo treinado no diretório especificado pelo SageMaker
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))

def model_fn(model_dir):
    """Carregar o modelo treinado."""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model
