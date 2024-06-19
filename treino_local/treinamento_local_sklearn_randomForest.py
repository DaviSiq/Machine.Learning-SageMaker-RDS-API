import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
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
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 200, 750)
    max_depth = trial.suggest_int('max_depth', 20, 40)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    criterion = trial.suggest_categorical('criterion', ['gini'])

    # Criar o modelo com os hiperparâmetros sugeridos pelo Optuna
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    
    # Criar o pipeline incluindo o pré-processador e o modelo sugerido pelo Optuna
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    
    # Treinar o modelo com os dados de treino
    model.fit(X_train, y_train)
    
    # Fazer previsões nos dados de teste e retornar a acurácia
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Criar um estudo do Optuna e otimizar a função objetivo

study_rf = optuna.create_study(direction='maximize')

study_rf.optimize(objective_rf, n_trials=100)

# Após a otimização, imprimir os melhores hiperparâmetros encontrados
print("Best trial:")
trial = study_rf.best_trial

print("Value: {}".format(trial.value))
print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Você pode usar os melhores hiperparâmetros encontrados para treinar seu modelo final
best_rf_params = trial.params

best_rf_params['random_state'] = 42  # Adicionar random_state aos parâmetros

best_rf = RandomForestClassifier(**best_rf_params)
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_rf)
])

# Treinar o modelo final com os melhores hiperparâmetros
final_model.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo final
y_pred_final = final_model.predict(X_test)

print("Final Accuracy:", accuracy_score(y_test, y_pred_final))
print("Final Classification Report:\n", classification_report(y_test, y_pred_final))
