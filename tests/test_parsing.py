from rsrl.envs.parsing import extract_tagged_answer


def test_last_answer_wins():
    assert extract_tagged_answer("<answer>1</answer> then <answer> 2 </answer>") == "2"


def test_unclosed_mention_does_not_swallow_answer():
    response = "I'll put the final expression in the <answer> tags.\n...\n<answer>(3 + 5) * 2</answer>"
    assert extract_tagged_answer(response) == "(3 + 5) * 2"


def test_no_answer():
    assert extract_tagged_answer("no tags here, just <answer> unclosed") is None
