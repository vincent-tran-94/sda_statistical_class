import logging
from collections import OrderedDict

import numpy as np
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.colinear_features_cleaner import (
    ColinearFeatureCleanerPreprocessor,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.column_dropper_preprocessor import (
    ColumnDropperPreprocessor,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.nan_dropper_preprocessor import (
    NanDropperPreprocessor,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessing_pipeline import (
    PreprocessingPipeline,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.qualitative_column_preprocessor import (
    QualitativeColumnsOneHotEncodingPreprocessor,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.target_splitter_preprocessor import (
    TargetSplitterPreprocessor,
)

log_format = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger("preprocessing_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def test_nan_dropper():
    data = DataFrame(
        {
            "feature1": [1, 2, 3, np.nan],
            "feature2": [4, 5, 6, 7],
            "feature3": [8, 9, np.nan, 11],
        }
    )

    nan_dropper = NanDropperPreprocessor()
    nan_dropper.fit(data)
    transformed_data = nan_dropper.transform(data)

    expected_data = DataFrame(
        data={"feature1": [1, 2], "feature2": [4, 5], "feature3": [8, 9]}
    )
    assert_frame_equal(left=transformed_data, right=expected_data, check_dtype=False)

    assert_frame_equal(
        left=transformed_data,
        right=NanDropperPreprocessor().fit_transform(X=data),
        check_dtype=False,
    )


def test_column_dropper():
    data = DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [8, 9, 11],
        }
    )
    column_dropper = ColumnDropperPreprocessor()
    column_dropper.fit(X=data, column_name="feature2")

    dropped_column_data = column_dropper.transform(data)

    expected_data = DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature3": [8, 9, 11],
        }
    )

    assert_frame_equal(left=dropped_column_data, right=expected_data, check_dtype=False)
    assert_frame_equal(
        left=dropped_column_data,
        right=column_dropper.fit_transform(X=data, column_name="feature2"),
        check_dtype=False,
    )


def test_qualitative_columns_one_hot_encoding():
    data = DataFrame(
        {
            "feature1": ["ba", "bar", "foo"],
            "feature2": ["a", "b", "c"],
        }
    )
    column_dropper = QualitativeColumnsOneHotEncodingPreprocessor()
    column_dropper.fit(X=data, columns=["feature1", "feature2"])

    encoded_data = column_dropper.transform(data)

    expected_data = DataFrame(
        {
            "feature1_ba": [1, 0, 0],
            "feature1_bar": [0, 1, 0],
            "feature1_foo": [0, 0, 1],
            "feature2_a": [1, 0, 0],
            "feature2_b": [0, 1, 0],
            "feature2_c": [0, 0, 1],
        }
    )

    assert_frame_equal(left=encoded_data, right=expected_data, check_dtype=False)
    assert_frame_equal(
        left=encoded_data,
        right=column_dropper.fit_transform(X=data),
        check_dtype=False,
    )


def test_multicolinear_features_remover():
    data = DataFrame(
        {
            "target_column": [1, 2, 3],
            "colinear_feature": [9, 18, 27],
            "non_colinear_feature": [1003, 4, 22],
        }
    )
    column_dropper = ColinearFeatureCleanerPreprocessor()
    column_dropper.fit(X=data, target_column="target_column", threshold=0.9)

    cleaned_data = column_dropper.transform(data)

    expected_data = DataFrame(
        {
            "target_column": [1, 2, 3],
            "non_colinear_feature": [1003, 4, 22],
        }
    )

    # Assert that the cleaned data matches the expected output
    assert_frame_equal(left=cleaned_data, right=expected_data, check_dtype=False)
    assert_frame_equal(
        left=cleaned_data,
        right=column_dropper.fit_transform(
            X=data, target_column="target_column", threshold=0.9
        ),
        check_dtype=False,
    )


def test_target_splitter():
    data = DataFrame(
        {
            "target_column": [1, 2, 3],
            "colinear_feature": [9, 18, 27],
            "non_colinear_feature": [1003, 4, 22],
        }
    )
    splitter = TargetSplitterPreprocessor()
    splitter.fit(X=data, target_column="target_column")

    X, y = splitter.transform(data)

    expected_data = DataFrame(
        {
            "colinear_feature": [9, 18, 27],
            "non_colinear_feature": [1003, 4, 22],
        }
    )

    # Assert that the cleaned data matches the expected output
    assert_frame_equal(left=X, right=expected_data, check_dtype=False)
    assert np.array_equal(a1=y.values, a2=data.loc[:, "target_column"].values)

    X, y = splitter.fit_transform(X=data, target_column="target_column")
    assert_frame_equal(
        left=X,
        right=expected_data,
        check_dtype=False,
    )


def test_end_to_end_pipeline():
    input_data = DataFrame(
        {
            "target_column": [1, 2, 3],
            "nan_col": [1998, -16, np.nan],
            "colinear_feature": [9, 18, 27],
            "non_colinear_feature": [1003, 4, 22],
            "column_to_drop": [1, 2, 3],
            "qualitative_feature": ["ba", "bar", "foo"],
        }
    )

    pipeline_steps = OrderedDict(
        [
            ("nan_dropper", NanDropperPreprocessor()),
            ("column_dropper", ColumnDropperPreprocessor()),
            ("qualitative_encoder", QualitativeColumnsOneHotEncodingPreprocessor()),
            ("colinear_feature_cleaner", ColinearFeatureCleanerPreprocessor()),
            ("target_splitter", TargetSplitterPreprocessor()),
        ]
    )

    arguments = {
        "nan_dropper": {},
        "column_dropper": {"column_name": "column_to_drop"},
        "colinear_feature_cleaner": {
            "target_column": "target_column",
            "threshold": 0.99,
        },
        "qualitative_encoder": {"columns": ["qualitative_feature"]},
        "target_splitter": {"target_column": "target_column"},
    }

    pipeline = PreprocessingPipeline(
        preprocessors=pipeline_steps, preprocessor_kwargs=arguments
    )

    X, y = pipeline.fit_transform(X=input_data)

    expected_result = DataFrame(
        {
            "qualitative_feature_ba": [1, 0],
            "qualitative_feature_bar": [0, 1],
            "qualitative_feature_foo": [0, 0],
        }
    )

    assert all(y.values == [1, 2])
    assert "target_column" not in X.columns
    assert_frame_equal(left=X, right=expected_result, check_dtype=False)
