from scipy.stats import f_oneway
from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems

if __name__ == "__main__":
    problem = mock_problems.get("multiple_means_comparisons")
    sample_1, sample_2, sample_3 = problem.get_data()
    print(problem.problem_statement)

    # Performing ANOVA to compare the means of the three samples
    result = f_oneway(sample_1, sample_2, sample_3)
    result_text = (
        "Les moyennes sont égales entre les groupes"
        if result.pvalue >= 0.05
        else "Au moins une moyenne est différente"
    )

    print(result)
    print(result_text)
