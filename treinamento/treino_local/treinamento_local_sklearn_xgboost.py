import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Carregar o CSV
df = pd.read_csv('treinamento/data_config/modified_hotel_reservations.csv')

# Testando sem as colunas id e status
df = df.drop(columns=['Booking_ID'])

# Adicionar a nova coluna
df['no_total_people'] = df['no_of_adults'] + df['no_of_children']
df['no_total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

# Separar as features e o label
X = df.drop('label_avg_price_per_room', axis=1)
y = df['label_avg_price_per_room']

# Ajustando os rótulos da variável alvo para começarem de 0
y = y - 1

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

# Criar o pipeline incluindo o pré-processador e o modelo com XGBClassifier configurado com os melhores hiperparâmetros
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42))
])

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Treinando o modelo com os melhores hiperparâmetros encontrados pelo GridSearchCV
model.fit(X_train, y_train)

# Prever nos dados de teste e calcular a acurácia com o modelo treinado
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
