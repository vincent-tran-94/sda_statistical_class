from scipy.stats import kstest, norm
from statsmodels.stats._lilliefors import lilliefors

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

    kstat, pvalue = lilliefors(x=sample, dist="norm")
    # h0 : La distribution est normalement distribuée
    result_text = (
        "La distribution est normalement distribuée"
        if pvalue >= 0.05
        else "La distribution n'est pas normalement distribuée"
    )

    print(f"{result_text} - pvalue: {pvalue:.4f}")
