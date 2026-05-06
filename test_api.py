def test_health_endpoint_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["checks"]["model_loaded"] is True


def test_predict_endpoint_returns_prediction(client):
    response = client.post(
        "/predict",
        data={
            "location": "Tarwala Nagar",
            "region": "Nashik",
            "house-type": "Apartment",
            "area": "2260",
            "total_rooms": "6",
            "total_beds": "4",
            "age": "1",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["predicted_price"] == 1234567.89


def test_predict_endpoint_invalid_input(client):
    response = client.post(
        "/predict",
        data={
            "location": "Tarwala Nagar",
            "region": "Nashik",
        },
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload == {"status": "error", "message": "Invalid input. All fields are required."}
