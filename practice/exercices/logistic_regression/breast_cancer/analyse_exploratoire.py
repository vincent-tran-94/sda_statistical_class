from demos.utlis import setup_plot
from practice.solutions.logistic_regression.utils.utils import load_breast_cancer_data

if __name__ == "__main__":
    colors = setup_plot()

    # Load the dataset
    X, y = load_breast_cancer_data()

    # todo : Describe the features/target relationship & get statistics about the dataset (nb samples, missing values, ...)
