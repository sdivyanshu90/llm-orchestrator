from __future__ import annotations

from reasoner.stages import _extract_steps_from_raw_text


def test_extract_steps_from_malformed_decomposition_json() -> None:
    """Malformed near-JSON decomposition output should still yield recoverable steps."""
    raw = (
        '{"steps":[{"step_id":1,"description":"Identify the partitioning strategy used by the standard '
        'quicksort implementation (e.g., Lomuto or Hoare) and how the pivot is chosen.","depends_on":[]},'
        '{"step_id":2,"description":"Explain how the choice of pivot relative to the current subarray elements '
        'determines the sizes of the resulting left and right subarrays.","depends_on":[1]},'
        '"step_id":3,"description":"Determine the input configuration that causes the pivot to be the smallest '
        'or largest element in every recursive call, leading to one subarray of size n-1 and one of size 0.",'
        '"depends_on":[2]},"step_id":4,"description":"Show that this configuration produces a recursion tree of depth n, '
        'yielding a total number of comparisons proportional to 1+2+...+n = O(n²).","depends_on":[3]}]}'
    )

    steps = _extract_steps_from_raw_text(raw)

    assert len(steps) == 4
    assert steps[0].step_id == 1
    assert steps[1].depends_on == [1]
    assert steps[2].step_id == 3
    assert "recursion tree" in steps[3].description