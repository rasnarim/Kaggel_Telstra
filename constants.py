__author__ = 'rasoul'


DATA_PATH = '~/DATA/Kaggle/Telstra'
# DATA_PATH = '.'
TRINING_DATA_FINLE_NAME = 'train.csv'
TEST_DATA_FINLE_NAME = 'test.csv'
EVENT_TYPES_FINLE_NAME = 'event_type.csv'
LOG_FEATRE_FINLE_NAME = 'log_feature.csv'
RESOURCE_TYPE_FINLE_NAME = 'resource_type.csv'
SEVERITY_TYPE_FINLE_NAME = 'severity_type.csv'
SUBMISION_FILE = 'sample_submission.csv'
CONSIDER_LOCATIN = True
OTHER_FEATURES = [
    EVENT_TYPES_FINLE_NAME,  # 57 features
    RESOURCE_TYPE_FINLE_NAME,  # there are 11 features
    SEVERITY_TYPE_FINLE_NAME,  # there are 6 features
    LOG_FEATRE_FINLE_NAME  # 445 features
]