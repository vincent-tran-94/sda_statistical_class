from scipy.stats import levene

from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems
from practice.solutions.hypothesis_testing.utils.plotter import plot_two_samples

if __name__ == "__main__":
    problem = mock_problems.get("variance_comparison")
    sample_1, sample_2 = problem.get_data()
    print(problem.problem_statement)
    print(problem.get_hints())

    # your code here
    plot_two_samples(sample_1, sample_2)

    result = levene(sample_1, sample_2)
    result_text = (
        "Les variances sont égales"
        if result.pvalue >= 0.05
        else "Les variances sont différentes"
    )
    print(result)
