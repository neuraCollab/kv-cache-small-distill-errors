from kvtrace.fdp.tokenizer_align import decode_window, first_token_mismatch


def test_first_token_mismatch_same_lists():
    assert first_token_mismatch([1, 2, 3], [1, 2, 3]) is None


def test_first_token_mismatch_diff_at_0():
    assert first_token_mismatch([1, 2, 3], [4, 2, 3]) == 0


def test_first_token_mismatch_diff_in_middle():
    assert first_token_mismatch([1, 2, 3, 4], [1, 2, 5, 4]) == 2


def test_first_token_mismatch_different_lengths_shorter_wins():
    assert first_token_mismatch([1, 2, 3], [1, 2]) == 2
    assert first_token_mismatch([1, 2], [1, 2, 3]) == 2


def test_decode_window_clips_left():
    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "|".join(str(i) for i in ids)
    result = decode_window(FakeTokenizer(), [1, 2, 3, 4, 5], center=0, context=2)
    # Should clip left to 0 and go up to context tokens to the right.
    assert result == "1|2|3"


def test_decode_window_clips_right():
    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "|".join(str(i) for i in ids)
    # center near right edge with large context → window clipped to end of list,
    # left side runs as far as it can (down to 0).
    result = decode_window(FakeTokenizer(), [1, 2, 3, 4, 5], center=4, context=10)
    assert result == "1|2|3|4|5"
