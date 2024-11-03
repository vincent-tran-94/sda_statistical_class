from scipy.stats import kruskal
from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems

if __name__ == "__main__":
    problem = mock_problems.get("service_ratings_comparison")
    sample_1, sample_2, sample_3 = problem.get_data()

    # Perform the Kruskal-Wallis H test
    result = kruskal(sample_1, sample_2, sample_3)

    result_text = (
        "Les notes de service diffèrent significativement"
        if result.pvalue < 0.05
        else "Les notes de service ne diffèrent pas significativement"
    )

    print(f"Kruskal-Wallis H Statistic: {result.statistic}")
    print(f"P-value: {result.pvalue}")
    print(result_text)
