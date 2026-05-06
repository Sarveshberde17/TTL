import os

import joblib
import pandas as pd
import pytest

from app import create_app


class DummyModel:
    def predict(self, input_data):
        return [1234567.89]


@pytest.fixture()
def test_app(tmp_path):
    data_path = tmp_path / "Cleaned_data.csv"
    model_path = tmp_path / "RandomForestModel.pkl"

    df = pd.DataFrame(
        [
            {
                "locality_name": "Tarwala Nagar",
                "region_name": "Nashik",
                "price": 8499000.0,
                "value_per_sqft": 3761,
                "area": 2260,
                "construction_status": "Under Construction",
                "house_type": "Apartment",
                "total_rooms": 6,
                "total_beds": 4,
                "new_resale": "New",
                "age": 1.0,
            }
        ]
    )
    df.to_csv(data_path, index=False)
    joblib.dump(DummyModel(), model_path)

    app = create_app(
        {
            "TESTING": True,
            "APP_ENV": "test",
            "SECRET_KEY": "test-secret",
            "DATA_PATH": str(data_path),
            "MODEL_PATH": str(model_path),
            "RATE_LIMIT_PER_MINUTE": 1000,
            "REQUEST_TIMEOUT_SECONDS": 5,
        }
    )
    yield app

    for env_key in ("DATA_PATH", "MODEL_PATH"):
        if env_key in os.environ:
            del os.environ[env_key]


@pytest.fixture()
def client(test_app):
    return test_app.test_client()
