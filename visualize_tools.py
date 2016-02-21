__author__ = 'rasoul'

import matplotlib.pyplot as plt

from constants import OTHER_FEATURES
from utils import FeatureVecBuilder

def visualize_feature_vecs(list_of_file_names=OTHER_FEATURES):
    # plot vectors of different features according to training lables
    for ind, file_name in enumerate(list_of_file_names):
        #  vectorizing data
        feature_builder = FeatureVecBuilder(False, [file_name])
        feature_builder.vectorize_data()
        vectorized_data = feature_builder.vecrzd_data
        # selecting training vectors
        train_vecs = vectorized_data.loc[sorted(feature_builder.train_ids)]
        # plot vectors belonging to the same class separately
        # there are only three classes: type 0, 1, 2
        for label in range(3):
            type_i_vecs = train_vecs[feature_builder.all_lables == label]
            fig = plt.figure(ind * 10 + label)
            plt.title(file_name + 'class_name: ' + str(label))
            plt.imshow(type_i_vecs, interpolation='nearest', aspect='auto')
    plt.show()

if __name__ == "__main__":
    visualize_feature_vecs()