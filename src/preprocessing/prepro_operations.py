import warnings
from abc import ABC, abstractmethod

import pandas as pd

with warnings.catch_warnings(action="ignore"):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import StandardDataset
from fairlearn.preprocessing import CorrelationRemover
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.data.dataset import IndexDataset
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class PreprocessingOperation(BaseEstimator, TransformerMixin, ABC):
    """Base class for sklearn-compatible preprocessing operations on IndexDataset."""

    @abstractmethod
    def fit(self, X: IndexDataset, y=None):
        pass

    @abstractmethod
    def transform(self, X: IndexDataset) -> IndexDataset:
        pass


class MakeCategorical(PreprocessingOperation):
    """One-hot encodes categorical columns with cardinality between lower and upper bounds."""

    def __init__(self, lb: int, ub: int):
        self.lb = lb
        self.ub = ub
        self.categorical_columns_ = None
        self.encoder_ = None
        self.encoded_column_names_ = None

    def fit(self, X: IndexDataset, y=None):
        self.categorical_columns_ = X.df.select_dtypes(exclude=['number', 'bool']).columns.tolist()
        self.categorical_columns_ = [c for c in self.categorical_columns_ if c not in [X.target_name, X.sens_attr_name]]
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder_.fit(X.df[self.categorical_columns_])

        self.encoded_column_names_ = self.encoder_.get_feature_names_out(self.categorical_columns_)

        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        res = X.copy()
        if not self.categorical_columns_:
            return res

        encoded = self.encoder_.transform(res.df[self.categorical_columns_])
        encoded = pd.DataFrame(encoded, columns=self.encoded_column_names_, index=res.index)
        res.df = res.df.drop(columns=self.categorical_columns_)
        res.df = pd.concat([res.df, encoded], axis=1)
        return res


class Scale(PreprocessingOperation):
    """Scale all numerical column between 0 and 1"""

    def __init__(self):
        self.scaler_ = None
        self.numeric_columns_ = None

    def fit(self, X: IndexDataset, y=None):
        self.numeric_columns_ = [c for c in X.df.columns if
                                 (is_numeric_dtype(X.df[c])) and c not in [X.target_name, X.sens_attr_name]]
        self.scaler_ = MinMaxScaler()
        if self.numeric_columns_:
            self.scaler_.fit(X.df[self.numeric_columns_])
        else:
            logger.warning(f'SCALE_DATA : scale_data - Scaler not fitted')

        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        if self.numeric_columns_:
            new_df = X.copy()
            new_df.df[self.numeric_columns_] = self.scaler_.transform(X.df[self.numeric_columns_])
            return new_df
        else:
            logger.warning(f'SCALE_DATA : scale_data - Scaler not applied')
            return X


class ToFloat(PreprocessingOperation):
    """Convert all values of dataset to float type"""

    def __init__(self):
        self.fitted_ = None

    def fit(self, X: IndexDataset, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        new_X = X.copy()
        new_X.df = new_X.df.astype(float)
        return new_X


class TargetAtTheEnd(PreprocessingOperation):
    """Put target column at the end of the dataframe"""

    def __init__(self):
        self.fitted_ = None

    def fit(self, X: IndexDataset, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        cols = [c for c in X.df.columns if c != X.target_name] + [X.target_name]
        new_df = X.copy()
        new_df.df = new_df.df[cols]
        return new_df


class DownSampling(PreprocessingOperation):
    """Undersamples majority class to balance dataset."""

    def __init__(self, seed):
        self.rus_ = RandomUnderSampler(sampling_strategy='majority', random_state=seed)

    def fit(self, X: IndexDataset, y=None):
        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        res_X = X.df.drop(columns=[X.target_name])
        res_y = X.df[X.target_name]
        res_X_resampled, res_y_resampled = self.rus_.fit_resample(res_X, res_y)
        res_df = pd.concat([res_X_resampled, res_y_resampled], axis=1)
        return X.create_from_df(res_df)


class UpSampling(PreprocessingOperation):
    """Oversamples minority class to balance dataset."""

    def __init__(self, seed):
        self.ros_ = RandomOverSampler(sampling_strategy='minority', random_state=seed)

    def fit(self, X: IndexDataset, y=None):
        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        res_X = X.df.drop(columns=[X.target_name])
        res_y = X.df[X.target_name]
        res_X_resampled, res_y_resampled = self.ros_.fit_resample(res_X, res_y)
        res_df = pd.concat([res_X_resampled, res_y_resampled], axis=1)
        return X.create_from_df(res_df)


class CorrelationRemoverPrepro(PreprocessingOperation):
    """Apply Correlation Remover algorithm from FairLearn"""

    def __init__(self, sens_attr_name: str):
        self.cr_ = None
        self.sens_attr_name_ = sens_attr_name

    def fit(self, X: IndexDataset, y=None):
        self.cr_ = CorrelationRemover(sensitive_feature_ids=[self.sens_attr_name_])

    def transform(self, X: IndexDataset) -> IndexDataset:
        cr_col = list(X.df.columns)
        cr_col.remove(self.sens_attr_name_)

        df_cr = self.cr_.transform(X=X.df)
        df_cr = pd.DataFrame(df_cr, columns=cr_col)
        df_cr.index = X.df.index
        df_cr[self.sens_attr_name_] = X.df[self.sens_attr_name_]
        df_cr[X.target_name] = X.df[X.target_name]

        df_cr = df_cr[list(X.df.columns)]
        X_cr = X.create_from_df(df_cr)
        return X_cr


class DisparateImpactRemoverPrepro(PreprocessingOperation):
    """Applies Disparate Impact Remover algorithm from aif360."""

    def __init__(self, sens_attr_name: str, favorable_class: int, privileged_group: int):
        self.dir_ = None
        self.standard_data_ = None
        self.sens_attr_name_ = sens_attr_name
        self.favorable_class_ = favorable_class
        self.privileged_group_ = privileged_group

    def fit(self, X: IndexDataset, y=None):
        self.dir_ = DisparateImpactRemover(sensitive_attribute=self.sens_attr_name_)
        self.standard_data_ = StandardDataset(df=X.df,
                                              label_name=X.target_name,
                                              protected_attribute_names=[self.sens_attr_name_],
                                              favorable_classes=[self.favorable_class_],
                                              privileged_classes=[[self.privileged_group_]])
        self.dir_.fit(self.standard_data_)

    def transform(self, X: IndexDataset) -> IndexDataset:
        self.standard_data_ = StandardDataset(df=X.df,
                                              label_name=X.target_name,
                                              protected_attribute_names=[self.sens_attr_name_],
                                              favorable_classes=[self.favorable_class_],
                                              privileged_classes=[[self.privileged_group_]])
        data_dir = self.dir_.transform(self.standard_data_)
        data_dir, _ = data_dir.convert_to_dataframe()
        data_dir = data_dir[list(X.df.columns)]
        data_dir.index = data_dir.index.astype(int)
        X_dir = X.create_from_df(data_dir)
        return X_dir
