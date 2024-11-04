from scipy.stats import kstest, norm
from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems
from practice.solutions.hypothesis_testing.utils.plotter import plot_two_samples

if __name__ == "__main__":
    problem = mock_problems.get("normality_large_sample")
    sample = problem.get_data()
    plot_two_samples(
        sample,
        norm.rvs(size=len(sample)),
        figure_title="Normality test",
        sample_1_label="Sample",
        sample_2_label="Normal distribution",
        alternative_plot="ecdf",
    )
    # Calculate the mean and standard deviation of the sample
    sample_mean = sample.mean()
    sample_std = sample.std(ddof=1)  # Sample standard deviation

    # Perform the Kolmogorov-Smirnov test
    result = kstest(sample, "norm", args=(sample_mean, sample_std))

    result_text = (
        "La distribution est normalement distribuée"
        if result.pvalue >= 0.05
        else "La distribution n'est pas normalement distribuée"
    )

    print(result)
    print(result_text)
