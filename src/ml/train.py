from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from pandas import DataFrame
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging
from sklearn.ensemble import GradientBoostingRegressor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


random_state = 42
housing = fetch_california_housing(as_frame=True)
# Prepare the data
X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)


def train_model(model: BaseEstimator, X_train: DataFrame, y_train: DataFrame):
    logger.info("Starting model training...")
    # Define the search
    model.fit(X_train, y_train)
    logger.info("Model training completed.")
    return model

def evalute_model(model: BaseEstimator, X_test, y_test):
    logger.info("Starting model testing...")
    y_pred = model.predict(X_test)
    scores = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    logger.info("Model testing completed.")
    return scores

def log_model(run_name, model, params, metrics, experiment_name="Imo_production", model_name="Production-model"):
    logger.info(f"Logging model to MLflow with run name: {run_name}...")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=model,
                                 artifact_path="sklearn-model",
                                 signature=signature,
                                 registered_model_name=model_name)
    logger.info("Model logged to MLflow.")



def main():
    # Define the grid
    params = {
            'n_estimators': 150,
            'max_depth': 5,
            "random_state": random_state
            }

    model = GradientBoostingRegressor(**params)
    run_name = "Production-model"
    model_name = "Production-model"
    logger.info(f"Starting the entire training and testing pipeline with run name: {run_name}...")
    model = train_model(model, X_train, y_train)
    scores = evalute_model(model, X_test, y_test)
    #metrics = dict(**scores,**time)
    log_model(run_name, model, params, scores, model_name=model_name)
    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
