from pathlib import Path

from src.api.db import get_all_predictions, init_db_from_csv, save_prediction_request


def test_save_and_get_predictions(tmp_path: Path) -> None:
    db_path = tmp_path / "monitoring.db"
    csv_path = tmp_path / "prod_data.csv"
    csv_path.write_text(
        "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,prediction\n",
        encoding="utf-8",
    )
    init_db_from_csv(csv_path, db_path)

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


def test_init_db_from_csv(tmp_path: Path) -> None:
    """Teste l'initialisation de la base de données à partir d'un fichier CSV."""
    csv_path = tmp_path / "prod_data.csv"
    db_path = tmp_path / "monitoring.db"
    csv_path.write_text(
        "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,prediction\n"
        "8.3252,41.0,6.984,1.023,322.0,2.555,37.88,-122.23,4.1\n"
        "3.8462,52.0,6.237,0.972,2401.0,2.109,37.86,-122.22,3.2\n",
        encoding="utf-8",
    )

    init_db_from_csv(csv_path, db_path)
    rows = get_all_predictions(db_path)

    assert len(rows) == 2
    assert rows[0]["MedInc"] == 8.3252
    assert rows[0]["prediction"] == 4.1
