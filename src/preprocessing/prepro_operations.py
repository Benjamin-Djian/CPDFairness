from abc import ABC, abstractmethod

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.data.dataset import IndexDataset
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class PreprocessingOperation(BaseEstimator, TransformerMixin, ABC):
    @abstractmethod
    def fit(self, X: IndexDataset, y=None):
        pass

    @abstractmethod
    def transform(self, X: IndexDataset) -> IndexDataset:
        pass


class MakeCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, lb: int, ub: int):
        self.lb = lb
        self.ub = ub
        self.categorical_columns_ = None
        self.encoder_ = None
        self.encoded_column_names_ = None

    def fit(self, X: IndexDataset, y=None):
        logger.debug("Fitting One Hot Encoding")
        self.categorical_columns_ = X.df.select_dtypes(exclude=['number', 'bool']).columns.tolist()
        self.categorical_columns_ = [c for c in self.categorical_columns_ if c not in [X.target_name, X.sens_attr_name]]
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder_.fit(X.df[self.categorical_columns_])

        self.encoded_column_names_ = self.encoder_.get_feature_names_out(self.categorical_columns_)

        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        logger.debug("Starting One Hot Encoding")
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
        logger.debug("Fitting Scaling")
        self.numeric_columns_ = [c for c in X.df.columns if
                                 (is_numeric_dtype(X.df[c])) and c not in [X.target_name, X.sens_attr_name]]
        self.scaler_ = MinMaxScaler()
        if self.numeric_columns_:
            self.scaler_.fit(X.df[self.numeric_columns_])
        else:
            logger.warning(f'SCALE_DATA : scale_data - Scaler not fitted')

        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        logger.debug("Starting Scaling")
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
        logger.debug("Fitting float conversion")
        self.fitted_ = True
        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        logger.debug("Starting float conversion")
        new_X = X.copy()
        new_X.df = new_X.df.astype(float)
        return new_X


class TargetAtTheEnd(PreprocessingOperation):
    """Put target column at the end of the dataframe"""

    def __init__(self):
        self.fitted_ = None

    def fit(self, X: IndexDataset, y=None):
        logger.debug("Fitting column reorder")
        self.fitted_ = True
        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        logger.debug("Starting column reorder")
        cols = [c for c in X.df.columns if c != X.target_name] + [X.target_name]
        new_df = X.copy()
        new_df.df = new_df.df[cols]
        return new_df


class DownSampling(PreprocessingOperation):
    def __init__(self, seed):
        self.rus_ = RandomUnderSampler(sampling_strategy='majority', random_state=seed)

    def fit(self, X: IndexDataset, y=None):
        logger.debug("Fitting downsampling")
        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        logger.debug("Starting downsampling")
        res_X = X.df.drop(columns=[X.target_name])
        res_y = X.df[X.target_name]
        res_X_resampled, res_y_resampled = self.rus_.fit_resample(res_X, res_y)
        res_df = pd.concat([res_X_resampled, res_y_resampled], axis=1)
        return X.create_from_df(res_df)


class UpSampling(PreprocessingOperation):
    def __init__(self, seed):
        self.ros_ = RandomOverSampler(sampling_strategy='minority', random_state=seed)

    def fit(self, X: IndexDataset, y=None):
        logger.debug("Fitting upsampling")
        return self

    def transform(self, X: IndexDataset) -> IndexDataset:
        logger.debug("Starting upsampling")
        res_X = X.df.drop(columns=[X.target_name])
        res_y = X.df[X.target_name]
        res_X_resampled, res_y_resampled = self.ros_.fit_resample(res_X, res_y)
        res_df = pd.concat([res_X_resampled, res_y_resampled], axis=1)
        return X.create_from_df(res_df)

# class CorrelationRemoverPrepro(PreprocessingOperation):
#
#     def __init__(self, sens_attr_name: str):
#         from fairlearn.preprocessing import CorrelationRemover
#         self.cr = CorrelationRemover(sensitive_feature_ids=[sens_attr_name])
#
#     def fit(self, X: IndexDataset, y: pd.Series):
#         self.cr.fit(X=df, y=df[target_name])
#
#     def transform(self, X: IndexDataset) -> IndexDataset:
#         target_col = df[target_name]
#         index_col = df.index
#
#         X_cr = self.cr.transform(X=df)
#         cr_col = list(df.columns)
#         cr_col.remove(sens_attr_name)
#         X_cr = pd.DataFrame(X_cr, columns=cr_col)
#         X_cr.index = index_col
#         X_cr[sens_attr_name] = df[sens_attr_name]
#         X_cr[target_name] = target_col
#
#         X_cr = X_cr[list(df.columns)]
#         return X_cr
#
#
# class DisparateImpactRemoverPrepro(PreprocessingOperation):
#     def __init__(self, sens_attr_name: str):
#         from aif360.algorithms.preprocessing import DisparateImpactRemover
#         self.dir = DisparateImpactRemover(sensitive_attribute=sens_attr_name)
#         self.standard_data = None
#
#     def fit(self, X: IndexDataset, y: pd.Series):
#         self.standard_data = StandardDataset(df=df,
#                                              label_name=target_name,
#                                              protected_attribute_names=[sens_attr_name],
#                                              favorable_classes=favorable_classes,
#                                              # Label that is considered as positive
#                                              privileged_classes=privileged_classes)  # protected attr that are considered privileged
#         self.dir.fit(self.standard_data)
#
#     def transform(self, X: IndexDataset) -> IndexDataset:
#         self.standard_data = StandardDataset(df=df,
#                                              label_name=target_name,
#                                              protected_attribute_names=[sens_attr_name],
#                                              favorable_classes=favorable_classes,
#                                              # Label that is considered as positive
#                                              privileged_classes=privileged_classes)  # protected attr that are considered privileged
#         data_dir = self.dir.transform(self.standard_data)
#         data_dir, _ = data_dir.convert_to_dataframe()
#         data_dir = data_dir[list(df.columns)]
#         data_dir.index = data_dir.index.astype(int)
#         return data_dir
