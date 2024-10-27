from scipy.stats import pearsonr
from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems
from practice.solutions.utils.plotter.plotter import plot_two_samples

if __name__ == "__main__":
    problem = mock_problems.get("sales_vs_ads")
    sales, ads_budget = problem.get_data()
    print(problem.problem_statement)
    print(problem.get_hints())

    correlation_coefficient, p_value = pearsonr(sales, ads_budget)

    plot_two_samples(
        sample_1=sales,
        sample_2=ads_budget,
        sample_1_label="Sales",
        figure_title="Sales vs Ads Budget",
        sample_2_label="Ads Budget",
        alternative_plot="scatter",
    )

    result_text = (
        "Il y a une corrélation significative positive"
        if p_value < 0.05 and correlation_coefficient > 0
        else (
            "Il y a une corrélation significative négative"
            if p_value < 0.05 and correlation_coefficient < 0
            else "Il n'y a pas de corrélation significative"
        )
    )

    print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")
    print(result_text)
