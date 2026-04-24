import pytest

from kvtrace.generators.base import Generator, GenerationResult


def test_generation_result_required_fields():
    r = GenerationResult(
        idx=0,
        raw="<text>",
        think="reasoning",
        final_response="answer",
        boxed_answer="42",
        think_complete=True,
        finish_reason="stop",
        token_ids=[1, 2, 3],
        prompt_tokens=5,
        generated_tokens=3,
    )
    d = r.to_dict()
    assert d["idx"] == 0
    assert d["boxed_answer"] == "42"
    assert d["num_generated_tokens"] == 3


def test_generator_is_abstract():
    with pytest.raises(TypeError):
        Generator()  # type: ignore[abstract]


def test_generator_context_manager_calls_load_and_unload():
    calls = []

    class DummyGen(Generator):
        def load(self, model_id: str, quant_config) -> None:
            calls.append(("load", model_id))
        def generate(self, problems):
            calls.append(("generate", len(problems)))
            return []
        def unload(self) -> None:
            calls.append(("unload",))

    g = DummyGen()
    with g:
        g.load("m", None)
        g.generate([1, 2, 3])
    assert ("load", "m") in calls
    assert ("generate", 3) in calls
    assert ("unload",) in calls
