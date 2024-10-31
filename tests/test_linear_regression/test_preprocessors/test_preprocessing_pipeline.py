import pytest
from collections import OrderedDict
from unittest.mock import Mock
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessing_pipeline import (
    PreprocessingPipeline,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessor_interface import (
    PreprocessorInterface,
)


# Fixtures and Mocks
@pytest.fixture
def sample_data():
    """Fixture to create sample DataFrame for testing"""
    return DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})


@pytest.fixture
def mock_preprocessors():
    """Fixture to create mock preprocessors implementing PreprocessorInterface"""
    preprocessor1 = Mock(spec=PreprocessorInterface)
    preprocessor2 = Mock(spec=PreprocessorInterface)
    preprocessor1.fit.return_value = preprocessor1
    preprocessor2.fit.return_value = preprocessor2
    preprocessor1.transform.return_value = DataFrame(
        {"feature1": [10, 20, 30], "feature2": [40, 50, 60]}
    )
    preprocessor2.transform.return_value = DataFrame(
        {"feature1": [100, 200, 300], "feature2": [400, 500, 600]}
    )
    return OrderedDict(
        [("preprocessor1", preprocessor1), ("preprocessor2", preprocessor2)]
    )


@pytest.fixture
def mock_kwargs():
    """Fixture to create kwargs for each preprocessor"""
    return {
        "preprocessor1": {"param1": "value1"},
        "preprocessor2": {"param2": "value2"},
    }


@pytest.fixture
def pipeline(mock_preprocessors, mock_kwargs):
    """Fixture to create a PreprocessingPipeline with mock preprocessors and kwargs"""
    return PreprocessingPipeline(
        preprocessors=mock_preprocessors, preprocessor_kwargs=mock_kwargs
    )


def test_fit(pipeline, sample_data, mock_preprocessors, mock_kwargs):
    pipeline.fit(sample_data)

    # Check that fit is called on each preprocessor with the DataFrame and corresponding kwargs
    for name, preprocessor in mock_preprocessors.items():
        preprocessor.fit.assert_called_once_with(sample_data, **mock_kwargs[name])


def test_transform(pipeline, sample_data, mock_preprocessors, mock_kwargs):
    transformed_data = pipeline.transform(sample_data)

    # Ensure each transform is called once
    for name, preprocessor in mock_preprocessors.items():
        preprocessor.transform.assert_called_once()

    # Check that the output of transform matches the expected final transform result
    expected_output = mock_preprocessors["preprocessor2"].transform.return_value
    assert_frame_equal(transformed_data, expected_output)


def test_fit_transform(pipeline, sample_data, mock_preprocessors, mock_kwargs):
    transformed_data = pipeline.fit_transform(sample_data)

    # Check that fit is called on each preprocessor in sequence with kwargs
    for name, preprocessor in mock_preprocessors.items():
        preprocessor.fit.assert_called_once_with(sample_data, **mock_kwargs[name])

        # Ensure transform is called once without directly comparing arguments
        preprocessor.transform.assert_called_once()

    # Check that the output of fit_transform matches the final transform result
    expected_output = mock_preprocessors["preprocessor2"].transform.return_value
    assert_frame_equal(transformed_data, expected_output)


def test_empty_preprocessors():
    pipeline = PreprocessingPipeline(OrderedDict(), {})
    sample_data = DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

    # fit and transform should work without errors with empty preprocessors
    pipeline.fit(sample_data)
    transformed_data = pipeline.transform(sample_data)

    # With no preprocessors, the output should be the same as the input
    assert transformed_data.equals(sample_data)


def test_fit_returns_self(pipeline, sample_data):
    result = pipeline.fit(sample_data)
    assert result is pipeline  # fit should return self for method chaining
