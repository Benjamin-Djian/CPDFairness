import os
from collections import OrderedDict
from unittest.mock import patch

import pytest

from src.utils.env import (
    get_csv_column_types,
    getenv_int,
    getenv_float,
    getenv_str,
    getenv_list,
)


class TestGetEnvInt:
    """Tests for getenv_int function."""

    def test_getenv_int_valid(self):
        """Test retrieving a valid integer env var."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = getenv_int("TEST_INT")
            assert result == 42

    def test_getenv_int_negative(self):
        """Test retrieving a negative integer env var."""
        with patch.dict(os.environ, {"TEST_NEG": "-10"}):
            result = getenv_int("TEST_NEG")
            assert result == -10

    def test_getenv_int_missing_raises_keyerror(self):
        """Test missing env var raises KeyError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(KeyError, match="Required environment variable"):
                getenv_int("NONEXISTENT_VAR")

    def test_getenv_int_invalid_raises_valueerror(self):
        """Test invalid value raises ValueError."""
        with patch.dict(os.environ, {"TEST_INVALID": "not_a_number"}):
            with pytest.raises(ValueError):
                getenv_int("TEST_INVALID")


class TestGetEnvFloat:
    """Tests for getenv_float function."""

    def test_getenv_float_valid(self):
        """Test retrieving a valid float env var."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = getenv_float("TEST_FLOAT")
            assert result == 3.14

    def test_getenv_float_missing_raises_keyerror(self):
        """Test missing env var raises KeyError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(KeyError, match="Required environment variable"):
                getenv_float("NONEXISTENT_VAR")

    def test_getenv_int_invalid_raises_valueerror(self):
        """Test invalid value raises ValueError."""
        with patch.dict(os.environ, {"TEST_INVALID": "not_a_number"}):
            with pytest.raises(ValueError):
                getenv_float("TEST_INVALID")

class TestGetEnvStr:
    """Tests for getenv_str function."""

    def test_getenv_str_valid(self):
        """Test retrieving a valid string env var."""
        with patch.dict(os.environ, {"TEST_STR": "hello"}):
            result = getenv_str("TEST_STR")
            assert result == "hello"

    def test_getenv_str_missing_raises_keyerror(self):
        """Test missing env var raises KeyError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(KeyError, match="Required environment variable"):
                getenv_str("NONEXISTENT_VAR")

class TestGetEnvList:
    """Tests for getenv_list function."""

    def test_getenv_list_default_separator(self):
        """Test retrieving list with default comma separator."""
        with patch.dict(os.environ, {"TEST_LIST": "a, b, c"}):
            result = getenv_list("TEST_LIST")
            assert result == ["a", "b", "c"]

    def test_getenv_list_custom_separator(self):
        """Test retrieving list with custom separator."""
        with patch.dict(os.environ, {"TEST_LIST": "a|b|c"}):
            result = getenv_list("TEST_LIST", sep="|")
            assert result == ["a", "b", "c"]

    def test_getenv_list_empty_items_ignored(self):
        """Test empty items are ignored."""
        with patch.dict(os.environ, {"TEST_LIST": "a,,b,c"}):
            result = getenv_list("TEST_LIST")
            assert result == ["a", "b", "c"]

    def test_getenv_list_missing_raises_keyerror(self):
        """Test missing env var raises KeyError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(KeyError, match="Required environment variable"):
                getenv_list("NONEXISTENT_VAR")


class TestGetCsvColumnTypes:
    """Tests for get_csv_column_types function."""

    def test_get_csv_column_types_valid(self):
        """Test parsing valid column type string."""
        with patch.dict(os.environ, {"TEST_COLS": "col1:int,col2:str,col3:float"}):
            result = get_csv_column_types("TEST_COLS")
            assert isinstance(result, OrderedDict)
            assert result["col1"] is int
            assert result["col2"] is str
            assert result["col3"] is float

    def test_get_csv_column_types_bool(self):
        """Test parsing bool column type."""
        with patch.dict(os.environ, {"TEST_COLS": "flag:bool"}):
            result = get_csv_column_types("TEST_COLS")
            assert result["flag"] is bool
