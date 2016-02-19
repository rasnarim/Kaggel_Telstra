__author__ = 'rasoul'

import pandas as pd
import os
import copy
import numpy as np

from constants import TRINING_DATA_FINLE_NAME, TEST_DATA_FINLE_NAME, \
    OTHER_FEATURES, CONSIDER_LOCATIN, DATA_PATH, LOG_FEATRE_FINLE_NAME


def aggregate_rows(df):
    """
    :param df: dummy_variables
    :type df: pandas DataFrame
    :return: aggregated based on index
    :rtype: pandas DataFrame
    """
    groups = df.groupby(df.index).sum()
    return groups


def make_dummy_using_mask(original_col, values_of_interest):
    """
    :param original_col: column to be converted to dummy variables
    :type original_col: Pandas series
    :param values_of_interest: values of interest i.e. training ids
    :type values_of_interest: Pandas series
    :return: binary dummy variables
    :rtype: Pandas DataFrame
    """

    temp_col_name = 'not_interesting'
    location_mask = original_col.isin(values_of_interest)
    original_col[~location_mask] = temp_col_name
    dummy_values = pd.get_dummies(original_col)
    # removing temp column for location that are not in training
    dummy_values.drop(temp_col_name, axis=1, inplace=True)
    return dummy_values


class FeatureVecBuilder:
    def __init__(self, consider_location=CONSIDER_LOCATIN, other_features=OTHER_FEATURES):
        self.consider_location = consider_location
        self.train_file = TRINING_DATA_FINLE_NAME
        self.test_file = TEST_DATA_FINLE_NAME
        self.other_features = other_features
        self.id_column_name = 'id'
        self.location_col_name = 'location'
        self.fault_severity_col_name = 'fault_severity'
        self.vecrzd_data = pd.DataFrame()
        self.train_lables = pd.DataFrame()
        self.all_ids = pd.Series()
        self.train_ids = pd.Series()
        self.test_ids = pd.Series()

    def vectorize_data(self):
        # import train and test data files
        train_data = pd.read_csv(os.path.join(DATA_PATH, TRINING_DATA_FINLE_NAME), index_col=self.id_column_name)
        test_data = pd.read_csv(os.path.join(DATA_PATH, TEST_DATA_FINLE_NAME), index_col=self.id_column_name)
        print train_data[self.fault_severity_col_name].value_counts()
        # make train_id, train_lable and test id
        self.train_ids = train_data.index
        self.test_ids = test_data.index
        all_data = pd.concat([train_data, test_data])
        self.train_lables = train_data.loc[:, self.fault_severity_col_name]
        # make dummy_variable
        # for each file in other_features --> new features are added to dummy_variable
        #
        if self.consider_location:
            # only considering locations in training
            original_col = all_data[self.location_col_name]
            self.vecrzd_data = make_dummy_using_mask(original_col, train_data[self.location_col_name])

        else:
            self.vecrzd_data = all_data
            self.vecrzd_data.drop(self.location_col_name, axis=1, inplace=True)
            # self.vecrzd_data.set_index(self.id_column_name, inplace=True)
        self.vecrzd_data.sort_index(inplace=True)
        for file_name in self.other_features:
            df = pd.read_csv(os.path.join(DATA_PATH, file_name), index_col=self.id_column_name)
            # make feature based on rows belonging to training data
            dummy_data = make_dummy_using_mask(copy.deepcopy(df.iloc[:, 0]),
                                               copy.deepcopy(df.loc[self.train_ids, :].iloc[:, 0]))

            if file_name == LOG_FEATRE_FINLE_NAME:
                # log_feature has an additional column hence extra process is considered for it
                func = lambda x: np.asarray(x) * df['volume'].values
                # alternative function:
                # func = lambda x: np.asarray(x) * np.log10(df['volume'].values + 1)
                dummy_data = dummy_data.apply(func)

            dummy_data = aggregate_rows(dummy_data)
            self.vecrzd_data = pd.concat([self.vecrzd_data, dummy_data], axis=1)


if __name__ == "__main__":
    feature_builder = FeatureVecBuilder()
    feature_builder.vectorize_data()
    vectorized_data = feature_builder.vecrzd_data
    train_vecs = vectorized_data.loc[sorted(feature_builder.train_ids)]
    test_vecs = vectorized_data.loc[sorted(feature_builder.test_ids)]
    training_lables = feature_builder.train_lables