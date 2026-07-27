"""
Network-free (and GPU-free) sanity checks for gpu_utils.py: VRAM-tiered
budget selection and the OOM-backoff batching helper. `safe_batched_call`'s
retry logic is exercised by *simulating* torch.cuda.OutOfMemoryError from a
fake function -- constructing/raising that exception class doesn't require
an actual CUDA device, so this suite runs anywhere torch does.
"""
import torch

from neural_foxp2.gpu_utils import GPUBudget, recommended_budget, safe_batched_call, memory_snapshot


def test_recommended_budget_falls_back_without_cuda():
    # In a CPU-only environment (this test sandbox), detect_vram_gb returns
    # None and recommended_budget must fall back to the plain GPUBudget()
    # defaults rather than raising.
    budget = recommended_budget(device="cpu")
    assert isinstance(budget, GPUBudget)
    assert budget.prompt_batch_size > 0
    assert budget.sae_dtype == torch.bfloat16


def test_memory_snapshot_reports_cuda_unavailable_gracefully():
    snap = memory_snapshot(device="cpu")
    # On a CPU-only box torch.cuda.is_available() is False regardless of the
    # device string passed in; the snapshot must degrade gracefully.
    assert "cuda_available" in snap


def test_safe_batched_call_backoff_and_correctness():
    """fn raises a simulated OOM for any chunk larger than 2 items; the
    batching helper should back off to a smaller batch size and still
    process every item, in order, without dropping or duplicating any."""
    call_sizes = []

    def fn(chunk):
        call_sizes.append(len(chunk))
        if len(chunk) > 2:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return torch.tensor(chunk)

    items = [1, 2, 3, 4, 5]
    out = safe_batched_call(items, fn, batch_size=4, min_batch_size=1, combine="cat")

    assert out.tolist() == items
    # The first attempt at batch_size=4 must have failed and backed off.
    assert max(call_sizes) <= 4
    assert any(s > 2 for s in call_sizes)  # the failed attempt was recorded before raising


def test_safe_batched_call_reraises_below_min_batch_size():
    def always_oom(chunk):
        raise torch.cuda.OutOfMemoryError("simulated OOM")

    try:
        safe_batched_call([1, 2, 3], always_oom, batch_size=2, min_batch_size=1, combine="cat")
        assert False, "expected OutOfMemoryError to propagate once min_batch_size is reached"
    except torch.cuda.OutOfMemoryError:
        pass


def test_safe_batched_call_list_combine():
    def fn(chunk):
        return [x * 10 for x in chunk]

    out = safe_batched_call([1, 2, 3, 4], fn, batch_size=2, combine="list")
    assert out == [10, 20, 30, 40]


if __name__ == "__main__":
    test_recommended_budget_falls_back_without_cuda()
    test_memory_snapshot_reports_cuda_unavailable_gracefully()
    test_safe_batched_call_backoff_and_correctness()
    test_safe_batched_call_reraises_below_min_batch_size()
    test_safe_batched_call_list_combine()
    print("All GPU-utils sanity checks passed.")
