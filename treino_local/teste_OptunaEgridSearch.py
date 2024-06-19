import pandas as pd
import optuna
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Carregar o CSV 
df = pd.read_csv('csv_config/modified_hotel_reservations.csv')

# Testando sem as colunas id e status
df = df.drop(columns=['Booking_ID', 'booking_status'])
df['no_total_people'] = df['no_of_adults'] + df['no_of_children']
df['no_total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

# Separar as features e o label
X = df.drop('label_avg_price_per_room', axis=1)
y = df['label_avg_price_per_room']

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

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Definir a função objetivo para o estudo do Optuna
from sklearn.ensemble import RandomForestClassifier

# Criar o pipeline incluindo o pré-processador e o modelo com RandomForestClassifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parâmetros para GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 500],
    'classifier__max_depth': [10, 20, None],
    'classifier__class_weight': ['balanced', {1: 1.0, 2: 1.0, 3: 1.0}]
}

# Configurar GridSearchCV
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5,
                           scoring='accuracy', verbose=10)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Treinando o modelo com GridSearchCV
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados pelo GridSearchCV
print("Melhores parâmetros:", grid_search.best_params_)

# Prever nos dados de teste e calcular a acurácia com o melhor modelo encontrado pelo GridSearchCV
y_pred = grid_search.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
