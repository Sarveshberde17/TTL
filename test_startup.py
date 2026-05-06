import pytest

from start import StartupValidationError, validate_startup


def test_startup_validation_fails_when_model_missing(tmp_path, monkeypatch):
    data_path = tmp_path / "Cleaned_data.csv"
    data_path.write_text("col\nvalue\n", encoding="utf-8")
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(StartupValidationError, match="Required model file not found"):
        validate_startup(str(tmp_path), "production")
