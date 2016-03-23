import unittest
import pandas as pd
import os

from utils import FeatureVecBuilder
from constants import TEST_DATA_PATH, OTHER_FEATURES, TRINING_DATA_FINLE_NAME, TEST_DATA_FINLE_NAME

class TestFeatureVecBuilder(unittest.TestCase):
    """
    tests FeatureVecBuilder based on test data provided on Github
    """
    def init_builder(self):
        # initializing feature vector builder
        return FeatureVecBuilder(False, OTHER_FEATURES, TEST_DATA_PATH)

    def test_id_list(self):
        # making feature vectors
        vectorized_data = self.init_builder()
        vectorized_data.vectorize_data()
        # checking if training labels are correct
        df_train = pd.read_csv(os.path.join(TEST_DATA_PATH, TRINING_DATA_FINLE_NAME))
        self.assertListEqual(sorted(df_train.id), sorted(vectorized_data.train_ids))
        # checking if test labels are correct
        df_test = pd.read_csv(os.path.join(TEST_DATA_PATH, TEST_DATA_FINLE_NAME))
        self.assertListEqual(sorted(df_test.id), sorted(vectorized_data.test_ids))

    def test_label_vs_types(self):
        # making feature vectors
        vectorized_data = self.init_builder()
        vectorized_data.vectorize_data()
        all_lables = vectorized_data.all_lables

        # checking type 0 labels:
        self.assertListEqual(sorted(all_lables.index[all_lables == 0].values), [5, 8, 13, 23])
        # checking type 1 labels:
        self.assertListEqual(sorted(all_lables.index[all_lables == 1].values), [1, 6, 19, 24, 28])
        # checking type 2 labels:
        self.assertListEqual(sorted(all_lables.index[all_lables == 2].values), [20, 26, 27, 29])

if __name__ == "__main__":
    unittest.main()