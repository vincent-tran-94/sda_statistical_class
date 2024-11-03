from utils.stat_test_practice_dataclass import ProblemContext


def test_practice_dataclass():
    test_context = ProblemContext(
        data=([1], [2]), problem_statement="foo", hints=["bar", "baz"], solution=12
    )

    assert test_context.data == ([1], [2])
    assert test_context.problem_statement == "foo"

    assert test_context.get_hints() == "Hints available:\n- bar\n- baz"
    assert test_context.solution == 12
