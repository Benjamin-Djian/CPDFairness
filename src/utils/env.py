import math
import os
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_csv_column_types(key: str) -> OrderedDict[str, type]:
    type_map = {
        "int": int,
        "str": str,
        "float": float,
        "bool": bool
    }
    header = os.environ.get(key)
    if header is None:
        raise ValueError(f"Header {key} not found")
    column_dict = OrderedDict()
    for key, value in (item.split(":") for item in header.split(",")):
        column_dict[key] = type_map[value]

    return column_dict


def getenv_int(key: str) -> int:
    """Retrieve an environment variable as int. Raises KeyError if missing."""
    value = os.environ.get(key)
    if value is None:
        raise KeyError(f"Required environment variable '{key}' not found.")
    return int(value)


def getenv_float(key: str) -> float:
    """Retrieve an environment variable as float. Raises KeyError if missing."""
    value = os.environ.get(key)
    if value is None:
        raise KeyError(f"Required environment variable '{key}' not found.")
    return float(value)


def getenv_list(key: str, sep: str = ",") -> list:
    """Retrieve an environment variable as a list of strings. Raises KeyError if missing."""
    value = os.environ.get(key)
    if value is None:
        raise KeyError(f"Required environment variable '{key}' not found.")
    return [x.strip() for x in value.split(sep) if x]


def getenv_str(key: str) -> str:
    """Retrieve an environment variable as string. Raises KeyError if missing."""
    value = os.environ.get(key)
    if value is None:
        raise KeyError(f"Required environment variable '{key}' not found.")
    return value


# ----- CALCUL -----
POS_UNDEF_INT = getenv_int("POS_UNDEF_INT")
EPSILON = getenv_float("EPSILON")
EPSILON_PREC = int(abs(math.log10(EPSILON)))
LOW_SMOOTHED_PROB = getenv_float("LOW_SMOOTHED_PROB")
LSP_PREC = int(abs(math.log10(LOW_SMOOTHED_PROB)))
DISPLAY_PREC = getenv_int("DISPLAY_PREC")

# ----- PATHS -----
ADULT_DATA_PATH = Path(getenv_str("ADULT_DATA_PATH"))
GERMAN_DATA_PATH = Path(getenv_str("GERMAN_DATA_PATH"))
LAW_DATA_PATH = Path(getenv_str("LAW_DATA_PATH"))

# ----- SAVE PATHS -----
MODEL_SAVE_PATH = Path(getenv_str("MODEL_SAVE_PATH"))
HIST_SAVE_PATH_0 = Path(getenv_str("HIST_SAVE_PATH_0"))
HIST_SAVE_PATH_1 = Path(getenv_str("HIST_SAVE_PATH_1"))
LH_G0_H0 = Path(getenv_str("LH_G0_H0"))
LH_G0_H1 = Path(getenv_str("LH_G0_H1"))
LH_G1_H0 = Path(getenv_str("LH_G1_H0"))
LH_G1_H1 = Path(getenv_str("LH_G1_H1"))

# ----- DATA -----
ADULT_TARGET = getenv_str("ADULT_TARGET")
GERMAN_TARGET = getenv_str("GERMAN_TARGET")
LAW_TARGET = getenv_str("LAW_TARGET")

# ----- LOGGING -----
LOG_LEVEL = getenv_str("LOG_LEVEL")
LOG_FORMAT = getenv_str("LOG_FORMAT")

# ----- FILE READER -----
# Contribs
CONTRIBS_HEADER = get_csv_column_types("CONTRIBS_HEADER")
CONTRIB_INPUT_POS = getenv_int("CONTRIB_INPUT_POS")
CONTRIB_LAYER_POS = getenv_int("CONTRIB_LAYER_POS")
CONTRIB_NODE_POS = getenv_int("CONTRIB_NODE_POS")
CONTRIB_VALUE_POS = getenv_int("CONTRIB_VALUE_POS")

# Histograms
HIST_HEADER = get_csv_column_types("HIST_HEADER")
HIST_NODE_POS = getenv_int("HIST_NODE_POS")
HIST_BIN_POS = getenv_int("HIST_BIN_POS")
HIST_LB_POS = getenv_int("HIST_LB_POS")
HIST_UB_POS = getenv_int("HIST_UB_POS")
HIST_FREQ_POS = getenv_int("HIST_FREQ_POS")

# Indexes
INDEXES_HEADER = get_csv_column_types("INDEXES_HEADER")
INDEX_INPUT_ID_POS = getenv_int("INDEX_INPUT_ID_POS")
TRUE_CLASS_POS = getenv_int("TRUE_CLASS_POS")
PRED_CLASS_POS = getenv_int("PRED_CLASS_POS")
SENS_ATTR_POS = getenv_int("SENS_ATTR_POS")
SET_NAME_POS = getenv_int("SET_NAME_POS")

# Likelihood
LH_HEADER = get_csv_column_types("LH_HEADER")
LH_INPUT_ID_POS = getenv_int("LH_INPUT_ID_POS")
LH_SCORE_POS = getenv_int("SCORE_POS")

# ----- Visualisation -----
DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_DPI = 100
