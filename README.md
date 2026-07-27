# Neural FOXP2

Language-specific SAE steering ("Neural FOXP2: Language-Specific Steering for
Targeted Language Improvement in LLMs") using **publicly released, pretrained
SAEs only** (Gemma Scope, Llama Scope, Qwen SAE-Res) — no SAE training.

Implements all three stages of the paper:

- **Stage I** — localize language-selective SAE features (`Sel_j`, causal
  `LiftSlope_j`, composite `Score_j` → `N_lt`).
- **Stage II** — layerwise SVD of the localized language-shift matrix,
  effective-rank + eigengap steering rank, contiguous window selection by
  spectral mass × bootstrap stability.
- **Stage III** — signed sparse activation steering: constant "push" toward
  the target-language direction + state-dependent "suppress" of the
  English-default attractor, applied only inside the selected window.

Every stage's intermediate output is saved as **JSON** (plus raw tensors as
`.pt`) so you can analyze feature rankings, spectra, and generation metrics
without re-running the model.

## Repo layout

```
neural-foxp2/
├── pyproject.toml            # pip install -e .  ->  `foxp2-run` CLI
├── requirements.txt
├── src/neural_foxp2/
│   ├── config.py              # MODELS / LANGUAGES registry (edit this to add models)
│   ├── sae_utils.py           # pretrained-SAE loaders (Gemma Scope / Llama Scope / Qwen SAE)
│   ├── activations.py         # generic residual-stream capture/inject hooks
│   ├── metrics.py             # Delta_M defaultness metric + token-set construction
│   ├── data_utils.py          # FLORES+ matched-pair + weak-prompt construction
│   ├── stage1_localize.py     # Stage I
│   ├── stage2_geometry.py     # Stage II
│   ├── stage3_steer.py        # Stage III
│   ├── serialization.py       # JSON logging for every stage
│   ├── pipeline.py            # NeuralFOXP2Pipeline: orchestrates I -> II -> III
│   └── cli.py                 # `foxp2-run` entry point
├── scripts/
│   ├── run_pipeline.py         # thin wrapper, works without installing the package
│   ├── run_batch.py            # run multiple (model, language) jobs sequentially, one GPU
│   ├── batch_jobs_example.json
│   └── prompts_example.txt
├── tests/
│   ├── test_synthetic.py      # network-free sanity checks (SVD/eigengap/projector math)
│   └── test_gpu_utils.py      # network-free sanity checks (VRAM-tiered batching, OOM backoff)
└── outputs/                   # created at runtime, gitignored
```

`src/neural_foxp2/gpu_utils.py` is new: it holds `GPUBudget` (batch sizes +
SAE dtype), VRAM auto-detection, a `torch.cuda.OutOfMemoryError`-safe batching
helper, and memory-snapshot reporting -- see "GPU memory management" below.

## Install

```bash
git clone <this-repo> neural-foxp2
cd neural-foxp2
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # installs torch/transformers/datasets/sae_lens/etc. + registers `foxp2-run`
```

(Or `pip install -r requirements.txt` if you don't want an editable install.)

Run the offline math tests any time (no GPU / network needed):

```bash
pytest tests/ -v
```

---

## Running on a GPU terminal

### 1. One-time setup on the GPU box

```bash
ssh you@your-gpu-box
git clone <this-repo> neural-foxp2 && cd neural-foxp2

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

nvidia-smi                       # sanity-check the GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Authenticate with Hugging Face

Several checkpoints are gated (Llama, Gemma Scope):

```bash
huggingface-cli login            # paste a token with read access to the gated repos you need
```

### 3. Run the full pipeline (Stage I → II → III) + generate

```bash
foxp2-run \
  --model_key llama3_1_8b_instruct \
  --lang_code hi \
  --device cuda \
  --output_dir outputs/llama3_1_8b_instruct-hi \
  --n_disc 150 --n_calib 40 --n_weak 60 \
  --top_k_per_layer 8 --lam 4.0 --beta 4.0 \
  --prompts_file scripts/prompts_example.txt \
  --gammas 0,0.5,1,2 \
  --max_new_tokens 128
```

(No install? Use `python scripts/run_pipeline.py ...` with the same flags.)

This will:
1. Load the model + per-layer pretrained SAEs onto the GPU.
2. Run Stage I/II/III, printing progress (`[foxp2] Stage I: ...`, etc.).
3. Write JSON artifacts under `outputs/llama3_1_8b_instruct-hi/` (see below).
4. Generate each prompt in `scripts/prompts_example.txt` at every `gamma` in
   `--gammas`, printing the output and appending a record to
   `generation_log.json`.

For a long run, background it and keep it alive after you disconnect:

```bash
tmux new -s foxp2
# ... run the command above inside tmux ...
# Ctrl-b d to detach; `tmux attach -t foxp2` to reattach later.
```

or

```bash
nohup foxp2-run --model_key llama3_1_8b_instruct --lang_code hi \
  --device cuda --output_dir outputs/llama3_1_8b_instruct-hi \
  --prompts_file scripts/prompts_example.txt --gammas 0,1,2 \
  > outputs/llama3_1_8b_instruct-hi/run.log 2>&1 &
tail -f outputs/llama3_1_8b_instruct-hi/run.log
```

### 4. Multi-GPU / bigger models

For models that don't fit on one GPU, load with `device_map="auto"` instead
of a fixed device — the cleanest way is to set `CUDA_VISIBLE_DEVICES` to all
GPUs you want to use and pass `--device cuda` (the underlying
`AutoModelForCausalLM.from_pretrained` call in `pipeline.py` can be swapped
to add `device_map="auto"` if you need automatic sharding across GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 foxp2-run --model_key qwen3_8b --lang_code zh --device cuda ...
```

### 5. Running many (model, language) jobs back-to-back

```bash
python scripts/run_batch.py \
  --jobs_file scripts/batch_jobs_example.json \
  --output_root outputs/batch_run \
  --device cuda
```

Each job in the JSON file gets its own model instance, its own output
subdirectory, and a full `pipe.close()` (frees the model + all SAEs + CUDA
memory) before the next job loads — so job N+1 never inherits job N's
memory footprint. A failing job is recorded in `batch_summary.json` with its
traceback rather than aborting the rest of the batch.

---

## GPU memory management (tuned for RTX PRO 6000, 96 GB)

Three things actually cause CUDA OOMs in this kind of pipeline, and all three
are now handled:

1. **Forgetting `torch.no_grad()`.** Every activation-capture / scoring /
   causal-lift forward pass runs under `@torch.no_grad()`
   (`activations.capture_hidden_states_batched`, `metrics.next_token_distributions`,
   Stage I/II probes) and every model parameter is `requires_grad_(False)`
   at load time. Building a full backward graph over an 8-9B model's forward
   pass, even just once by accident, costs far more memory than every other
   fix below combined — this was the single largest risk in the original code.

2. **Loading too many SAEs in fp32 at once.** SAEs are loaded per-layer, and
   Stage I/II need every *candidate* layer's SAE simultaneously. At fp32 this
   adds up fast:

   | Model | d_model × d_sae | SAE size/layer (fp32 → bf16) | All candidate layers (fp32 → bf16) |
   |---|---|---|---|
   | `gemma2_9b_it` | 3584 × 16384 | 0.47 GB → 0.24 GB | 17.9 GB → **8.9 GB** |
   | `llama3_1_8b_instruct` | 4096 × 32768 | 1.07 GB → 0.54 GB | 30.1 GB → **15.0 GB** |
   | `qwen3_8b` | 4096 × 65536 | 2.15 GB → 1.07 GB | 68.7 GB → **34.4 GB** |

   `GPUBudget.sae_dtype` defaults to **bf16**, roughly halving this. Combined
   with the base model (~16-20 GB bf16), even the worst case here
   (Qwen: 34.4 GB SAEs + ~18 GB model ≈ 52 GB) leaves comfortable headroom on
   a 96 GB card. Once Stage II selects the final intervention window
   (typically 2-5 layers), every *non-window* layer's SAE is freed
   (`GPUBudget.offload_non_window_saes`, on by default) — standing memory
   during generation drops to just the window layers.

3. **Unbounded batch sizes.** `n_disc`/`n_calib`/`n_weak` prompt lists and
   the Stage I causal-lift decoding loop are always chunked
   (`GPUBudget.prompt_batch_size`, `lift_probe_batch_size`,
   `generate_batch_size`), never forwarded as one giant batch. Every chunked
   forward pass is wrapped so that a `torch.cuda.OutOfMemoryError` **halves
   the batch size and retries the same chunk** (down to `min_batch_size=1`)
   instead of crashing the run — see `gpu_utils.safe_batched_call` and its
   inline use in `activations.py` / `metrics.py` / `pipeline.py`.

`GPUBudget` auto-detects VRAM and picks defaults accordingly
(`gpu_utils.recommended_budget`):

| Detected VRAM | prompt_batch_size | lift_probe_batch_size | generate_batch_size |
|---|---|---|---|
| ≥ 80 GB (RTX PRO 6000 / A100-80GB / H100) | 32 | 64 | 16 |
| ≥ 40 GB (A100-40GB / RTX 6000 Ada) | 16 | 32 | 8 |
| ≥ 20 GB (RTX 4090 / 3090) | 8 | 16 | 4 |
| < 20 GB | 4 | 8 | 2 |

Override any of these explicitly:

```bash
foxp2-run --model_key llama3_1_8b_instruct --lang_code hi --device cuda \
  --output_dir outputs/llama3_1_8b_instruct-hi \
  --batch_size 32 --lift_probe_batch_size 64 --generate_batch_size 16 \
  --sae_dtype bfloat16 \
  --prompts_file scripts/prompts_example.txt --gammas 0,1,2
```

Or from Python:

```python
from neural_foxp2 import NeuralFOXP2Pipeline, GPUBudget
import torch

budget = GPUBudget(prompt_batch_size=32, lift_probe_batch_size=64,
                    generate_batch_size=16, sae_dtype=torch.bfloat16)
pipe = NeuralFOXP2Pipeline(model_key="llama3_1_8b_instruct", lang_code="hi",
                            device="cuda", gpu_budget=budget)
```

Every run also writes `memory_report.json` (GPU memory snapshots taken
before the run and after each stage) so you can confirm actual usage after
the fact:

```python
import json
print(json.load(open("outputs/llama3_1_8b_instruct-hi/memory_report.json")))
# {"before_run": {...}, "after_stage1": {...}, "after_stage2": {...}, "after_stage3": {...}}
```

If you're VRAM-constrained on a smaller card, shrink `--candidate_layers` to
a narrower mid-network band (e.g. `--candidate_layers 10,11,...,24`) rather
than scanning every layer — fewer candidate layers means fewer SAEs loaded
at once during Stage I/II, on top of everything above.

---

## Inspecting the JSON outputs

After a run, `outputs/<run_name>/` contains:

| File | Contents |
|---|---|
| `run_config.json` | Every hyperparameter used (model, language, n_disc/n_calib/n_weak, lambda/beta/gamma, seed, candidate layers). |
| `stage1_features.json` | Per layer: each selected feature's index, selectivity `Sel_j`, causal `LiftSlope_j`, composite `Score_j`. |
| `stage2_geometry.json` | Per layer: support size, steering rank `r_l`, spectral mass, bootstrap stability, gain probe, singular-value preview, and the selected window `W`. |
| `stage2_directions.pt` | Raw right-singular-vector matrices `V[:, :r_l]` per layer (torch, for re-loading/plotting spectra in full). |
| `stage3_vectors.json` | Per window layer: `lambda_l`, `beta_l`, support size, `‖mu_target‖`, `‖mu_en‖`, vector previews. |
| `stage3_vectors.pt` | Raw `mu_target_final` / `mu_en_masked` / `support_mask` tensors per layer. |
| `memory_report.json` | GPU memory snapshots (`gpu_utils.memory_snapshot`) taken before the run and after each stage — confirms actual VRAM usage per stage. |
| `generation_log.json` | One record per prompt per `.generate(...)`/`.generate_batch(...)` call: prompt, `gamma`, `delta_m_baseline`, `delta_m_steered`, `delta_m_gain`, and the generated text — a growing JSON list, safe to `tail`/reload mid-run. |

Quick analysis example:

```python
import json
import pandas as pd

log = json.load(open("outputs/llama3_1_8b_instruct-hi/generation_log.json"))
df = pd.DataFrame(log)
print(df.groupby("gamma")["delta_m_gain"].mean())   # does defaultness gain scale with gamma?
```

```python
geo = json.load(open("outputs/llama3_1_8b_instruct-hi/stage2_geometry.json"))
for layer, g in geo["layers"].items():
    print(layer, "rank=", g["rank"], "mass=", round(g["mass"], 3), "stab=", round(g["stability"], 3))
```

---

## What's faithful vs. simplified relative to the paper

See the module docstrings in `stage1_localize.py` / `stage2_geometry.py` /
`stage3_steer.py` for the exact equations implemented. In short:

**Faithful to the main-text equations (Sec. 2):** Stage I selectivity +
causal-lift-slope scoring and composite ranking; Stage II effective rank,
eigengap rank selection, spectral mass, bootstrap stability, contiguous
window selection; Stage III's exact positive/negative edit equations and
composed `Π_N P_S Π_N` projector; the single `gamma` control-intensity knob.

**Simplified relative to the appendix:** no anti-confound format-sensitivity
pre-filter; `(lambda_l, beta_l)` are passed in directly rather than
grid-searched against a KL-trust-region target; `V_target`/`V_english` token
sets use a simple Unicode-script/diacritic heuristic rather than the
corpus-derived, transliteration-aware construction of Appendix D.1; no
KL-trust-region / semantic-invariance / safety guardrail checks are wired in
— add these before treating any `gamma` as "deployment-safe" in the paper's
sense.

`config.py`'s `qwen3_5_9b` entry (`Qwen/Qwen3.5-9B`) does not correspond to a
model/SAE release verified here; treat it as a placeholder until confirmed.
