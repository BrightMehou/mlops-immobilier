from pathlib import Path

from src.api.db import get_all_predictions, save_prediction_request


def test_save_and_get_predictions(tmp_path: Path) -> None:
    db_path = tmp_path / "monitoring.db"

    features = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 880.0,
        "AveBedrms": 129.0,
        "Population": 322.0,
        "AveOccup": 126.0,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    save_prediction_request(features, 200_000.0, db_path)
    rows = get_all_predictions(db_path)

    assert len(rows) == 1
    assert rows[0]["MedInc"] == features["MedInc"]
    assert rows[0]["prediction"] == 200_000.0
