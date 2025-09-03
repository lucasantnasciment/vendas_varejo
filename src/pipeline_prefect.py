from prefect import flow, task
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

@task
def load_data(path="data/energy_data.csv"):
    return pd.read_csv(path)

@task
def train_model(df):
    X = df[["temperatura", "hora_dia", "dia_semana"]]
    y = df["consumo"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    return model, score

@task
def save_best_model(model, score, threshold=0.7):
    if score >= threshold:
        joblib.dump(model, "models/model.pkl")
        return f"Novo modelo promovido (R²={score:.3f})"
    else:
        return f"Modelo não atingiu threshold (R²={score:.3f})"

@flow
def retraining_pipeline():
    df = load_data()
    model, score = train_model(df)
    result = save_best_model(model, score)
    print(result)

if __name__ == "__main__":
    retraining_pipeline()
