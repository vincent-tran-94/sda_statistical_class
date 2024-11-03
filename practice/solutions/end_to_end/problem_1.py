from scipy.stats import chi2_contingency
from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems

if __name__ == "__main__":
    problem = mock_problems.get("customer_purchase_category")
    sample_1, sample_2 = problem.get_data()

    # Create a contingency table
    contingency_table = [sample_1, sample_2]

    # Perform the Chi-squared test
    chi2_stat, p_value, _, expected = chi2_contingency(contingency_table)

    result_text = (
        "Les distributions sont significativement différentes"
        if p_value < 0.05
        else "Les distributions ne sont pas significativement différentes"
    )

    print(f"Chi-squared Statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    print(result_text)
