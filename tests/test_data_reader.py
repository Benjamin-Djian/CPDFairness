import pandas as pd
import pytest

from src.data.data_reader import (
    AdultDataReader,
    DataReader,
    GermanDataReader,
    LawDataReader,
)


class TestDataReader:
    """Tests for DataReader base class."""

    @pytest.fixture
    def valid_df(self):
        return pd.DataFrame({
            "feature1": [1.0, 0.0, 1.0, 0.0],
            "feature2": [0.5, 0.5, 0.5, 0.5],
            "target": [0, 1, 0, 1],
            "sens_attr": [0, 0, 1, 1]
        })

    @pytest.fixture
    def temp_csv(self, tmp_path, valid_df):
        file_path = tmp_path / "test_data.csv"
        valid_df.to_csv(file_path, index=True, index_label="inputId")
        return file_path

    def test_init(self, temp_csv):
        """Test DataReader initialization."""
        reader = DataReader(
            data_path=temp_csv,
            index_column="inputId",
            target_name="target",
            sens_attr_name="sens_attr"
        )

        assert reader.data_path == temp_csv
        assert reader.index_column == "inputId"
        assert reader.target_name == "target"
        assert reader.sens_attr_name == "sens_attr"

    def test_read_data_returns_dataframe(self, temp_csv):
        """Test read_data returns a DataFrame."""
        reader = DataReader(
            data_path=temp_csv,
            index_column="inputId",
            target_name="target",
            sens_attr_name="sens_attr"
        )

        df = reader.read_data()

        assert isinstance(df, pd.DataFrame)
        assert "feature1" in df.columns
        assert "feature2" in df.columns
        assert "target" in df.columns
        assert "sens_attr" in df.columns

    def test_read_data_uses_index_column(self, temp_csv):
        """Test read_data uses the specified index column."""
        reader = DataReader(
            data_path=temp_csv,
            index_column="inputId",
            target_name="target",
            sens_attr_name="sens_attr"
        )

        df = reader.read_data()

        assert "inputId" not in df.columns
        assert df.index.name == "inputId"

    def test_read_data_raises_on_missing_target(self, tmp_path):
        """Test read_data raises ValueError if target column is missing."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0],
            "feature2": [0.5, 0.5],
            "sens_attr": [0, 1]
        })
        file_path = tmp_path / "test_data.csv"
        df.to_csv(file_path, index=True, index_label="inputId")

        reader = DataReader(
            data_path=file_path,
            index_column="inputId",
            target_name="missing_target",
            sens_attr_name="sens_attr"
        )

        with pytest.raises(ValueError, match="target.*not in dataframe"):
            reader.read_data()

    def test_read_data_raises_on_missing_sens_attr(self, tmp_path):
        """Test read_data raises ValueError if sens_attr column is missing."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0],
            "feature2": [0.5, 0.5],
            "target": [0, 1]
        })
        file_path = tmp_path / "test_data.csv"
        df.to_csv(file_path, index=True, index_label="inputId")

        reader = DataReader(
            data_path=file_path,
            index_column="inputId",
            target_name="target",
            sens_attr_name="missing_sens_attr"
        )

        with pytest.raises(ValueError, match="Sensitive attribute.*not in dataframe"):
            reader.read_data()


class TestAdultDataReader:
    """Tests for AdultDataReader class."""

    def test_init(self):
        """Test AdultDataReader initialization."""
        reader = AdultDataReader(sens_attr_name="sex")

        assert reader.sens_attr_name == "sex"
        assert reader.target_name == "income"
        assert reader.index_column == "inputId"


class TestGermanDataReader:
    """Tests for GermanDataReader class."""

    def test_init(self):
        """Test GermanDataReader initialization."""
        reader = GermanDataReader(sens_attr_name="sex")

        assert reader.sens_attr_name == "sex"
        assert reader.target_name == "good_3_bad_2_customer"
        assert reader.index_column == "inputId"


class TestLawDataReader:
    """Tests for LawDataReader class."""

    def test_init(self):
        """Test LawDataReader initialization."""
        reader = LawDataReader(sens_attr_name="sex")

        assert reader.sens_attr_name == "sex"
        assert reader.target_name == "pass_bar"
        assert reader.index_column == "inputId"
