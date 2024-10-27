import numpy as np
from scipy.stats import shapiro

from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems
from practice.solutions.utils.plotter.plotter import plot_two_samples

if __name__ == "__main__":
    problem = mock_problems.get("normality_test")
    sample = problem.get_data()
    print(problem.problem_statement)
    print(problem.get_hints())

    sample_mean = np.mean(sample)
    sample_std = np.std(sample)

    plot_two_samples(
        sample,
        np.random.normal(sample_mean, sample_std, len(sample)),
        figure_title="Normality test",
        sample_2_label="Normal distribution",
    )

    # your code here
    result = shapiro(sample)
    result_text = (
        "La distribution est normalement distribuée"
        if result.pvalue >= 0.05
        else "La distribution n'est pas normalement distribuée"
    )

    print(result)
    print(result_text)
