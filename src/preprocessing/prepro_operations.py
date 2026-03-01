from abc import ABC, abstractmethod

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class PreprocessingOperation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, df: pd.DataFrame):
        pass


class MakeCategorical(PreprocessingOperation):
    def __init__(self, lb: int, ub: int):
        super().__init__()
        self.lb = lb
        self.ub = ub

    def run(self, df: pd.DataFrame):
        # We select all columns that are not number or bool dtype
        categorical_columns = df.select_dtypes(exclude=['number', 'bool']).columns.tolist()

        # Categorical columns are also numerical columns with 3 to numeric_as_categorical_max_thr different values
        for c in df.select_dtypes(include=['number']).columns:
            if self.lb <= df[c].value_counts().shape[0] <= self.ub:
                categorical_columns.append(c)

        logger.info(f"Categorical columns are : {categorical_columns}")
        df = pd.get_dummies(df, columns=categorical_columns, prefix_sep='=')


class Scale(PreprocessingOperation):
    """Scale all numerical column between 0 and 1"""

    def run(self, df: pd.DataFrame):
        scaler = MinMaxScaler()
        numeric_columns = [c for c in df.columns if (is_numeric_dtype(df[c]))]
        if numeric_columns:
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        else:
            logger.warning(f'SCALE_DATA : scale_data - Scaler not applied')


class ToFloat(PreprocessingOperation):
    """Convert all values of dataset to flaot type"""

    def run(self, df: pd.DataFrame):
        df = df.astype(float)


class CorrelationRemoverPrepro(PreprocessingOperation):
    def run(self, df: pd.DataFrame):
        from fairlearn.preprocessing import CorrelationRemover

        target_col = df[self.target]
        index_col = df.index

        cr = CorrelationRemover(sensitive_feature_ids=[protec_attr])
        X_cr = cr.fit_transform(X=df, y=target_col)
        cr_col = list(df.columns)
        cr_col.remove(protec_attr)
        X_cr = pd.DataFrame(X_cr, columns=cr_col)
        X_cr.index = index_col
        X_cr[protec_attr] = df[protec_attr]
        X_cr[self.target] = target_col

        # column order
        X_cr = X_cr[list(df.columns)]
        df = X_cr


class DisparateImpactRemoverPrepro(PreprocessingOperation):
    def run(self, df: pd.DataFrame):
        from aif360.datasets import StandardDataset
        from aif360.algorithms.preprocessing import DisparateImpactRemover

        standard_data = StandardDataset(df=df,
                                        label_name=self.target,
                                        protected_attribute_names=[protec_attr],
                                        favorable_classes=favorable_classes,
                                        # Label that is considered as positive
                                        privileged_classes=privileged_classes)  # protected attr that are considered privileged

        dir_ = DisparateImpactRemover(sensitive_attribute=protec_attr)
        data_dir = dir_.fit_transform(standard_data)
        data_dir, _ = data_dir.convert_to_dataframe()
        data_dir = data_dir[list(df.columns)]
        df = data_dir
        df.index = df.index.astype(int)
