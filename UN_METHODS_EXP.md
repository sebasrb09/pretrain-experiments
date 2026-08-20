# UN_METHODS_EXP.md

Operational reference for the **unlearning-method comparison experiments** on
branch `unlearning-experiments`. Where the other docs sit:

| File | Covers |
|---|---|
| `CLAUDE.md` | The repository: architecture, model zoo, eval data locations, driver inventory |
| `HYPER-PARAMS.md` | Published hyperparameters per method, and the 179M → 546M → 1B transfer plan |
| `UNLEARNING_EVALUATIONS.md` | Per-method protocol history, earlier sweeps, output layouts |
| **`UN_METHODS_EXP.md`** | **Which experiments run, and how to launch them** |

## The experiment

Eight post-hoc unlearning methods, each swept over **four values of one
designated hyperparameter**, produce 32 trained checkpoints. Every checkpoint
plus three reference checkpoints is evaluated, giving one point per
`(method, knob value)` on a **forgetting (x) versus utility (y)** plot — a
trade-off curve per method rather than a single winner per method.

That is the key difference from `HYPER-PARAMS.md`, which describes a staged
coordinate-descent search converging on one winning configuration. Here the
cells that over-forget and wreck utility are wanted: they define the shape of
the frontier.

- **Model**: `sbordt/OLMo-2-1B-Exp-Unlearning` @ `stage1-step100000-tokens210B` (16 layers)
- **Forget set**: library default — full `sbordt/OLMo-2-1B-Exp-Dataset` minus the
  11 `iid-replacements-*` controls
- **Retain set**: OLMo-2 stage1 sequences ahead of step 100k, via
  `unlearning_utils.build_olmo_retain_dataset`

## Methods and drivers

| Method | Driver | Retain? | Curve knob | Swept values | Held fixed | Epochs |
|---|---|---|---|---|---|---|
| GradAscent | `reweighted_ga.py` | no | learning rate | `1e-7 3e-7 1e-6 3e-6` | β₁=0, β₂=0, λ=0 | 10 |
| CE-U | `ce_u.py` | no | learning rate | `1e-7 3e-7 1e-6 3e-6` | *no method hyperparameters exist* | 8 |
| WGA | `reweighted_ga.py` | no | β₁ (=α) | `0.5 1.0 2.0 5.0` | β₂=0, λ=0, LR 1e-6 | 5 |
| SatImp | `reweighted_ga.py` | yes | β₁ | `1.0 2.0 5.0 10.0` | β₂=1, λ=1, LR 1e-6 | 5 |
| GradDiff | `grad_diff.py` | yes | λ retain weight | `0.5 1.0 2.0 5.0` | LR 1e-6 | 10 |
| NPO | `npo.py` | yes | β | `1e-4 1e-3 1e-2 1e-1` | λ=1, LR 1e-5, frozen ref fp32 | 10 |
| SimNPO | `simnpo.py` | yes | β | `0.1 0.5 1.0 2.5` | γ=0, λ=1, LR 1e-5 | 10 |
| RMU | `rmu.py` | yes | steering coef `c` | `2.0 4.0 6.5 10.0` | ℓ=7, α=1200, 3 layers, LR 5e-5 | **200 steps** |

Epoch counts are each method's own published protocol, not a shared value:
NPO and SimNPO 10 (arXiv:2404.05868, arXiv:2410.07163), CE-U 8 — its best
reported epoch (arXiv:2503.01224), WGA and SatImp 5 (arXiv:2502.19301,
arXiv:2505.11953), GradAscent 10 from Jang's 8–14-epochs-to-threshold at ~1B.
RMU is the exception: its paper budgets ~100–200 optimizer **steps**, because it
updates only `down_proj` on three layers.

**GradDiff's 10 is a judgment call, not a citation** — it is a baseline rather
than a paper with its own protocol, and runs at 10 epochs in the NPO/SimNPO
papers and 5 in the WGA/SatImp ones.

### Why three methods share one driver

`reweighted_ga.py` implements `w = p^β₁ · (1-p)^β₂`, so:

- **WGA** is that at β₂=0
- **GradAscent** is that at β₁=β₂=0, where `w = 1` and the loss reduces exactly
  to −mean CE
- **SatImp** is that at β₂=1 with the retain term on

Each cell records its own `--method-label`, so the three stay separate in the
output tree and the results table.

GradAscent deliberately does **not** run through `gradient_ascent.py`: that
driver hard-requires a single `--forget-experiment` and has no `--max-steps`, so
it cannot express the library-default forget set under this budget. It remains
the reference implementation and serves the in-flight 179M sweep.

`lunar.py` is implemented but out of scope; add `lunar` to `DEFAULT_METHODS` and
a dispatch branch to enable it.

## Shared budget

```
steps = ( |D_f| / TOTAL_BATCH ) × epochs,   capped at HARD_STEP_CAP
```

| Setting | Value | Note |
|---|---|---|
| `TOTAL_BATCH` | 512 | Forget sequences per optimizer step — **identical for every method**, so the step count is a function of the epoch count alone. This is OLMo's own `global_train_batch_size`. |
| `MICRO_BATCH` | 4 | OLMo's own `device_train_microbatch_size`. Accumulation derived: 512/4 = 128 |
| `HARD_STEP_CAP` | 10000 | 10% of the 100k-step pretraining run. A ceiling, not a target — methods finishing below it is expected. |
| `MAX_SEQ_LEN` | 1024 | Pretraining used 4096; forget items are truncated |
| `DTYPE` | `bfloat16` | Weights and Adam states stay fp32 under autocast |
| `FROZEN_DTYPE` | `float32` | NPO's log-ratio is a difference of large summed NLLs — see the precision note in `npo.py` |
| `INFERENCE_MAX_NUM_SEQS` | 16 | **Eval only.** The established 1B setting on A100-40GB (`UNLEARNING_EVALUATIONS.md`), which is also the MeluXina GPU. |

### Where the batch sizes come from

Both are read off `configs/official-0425/OLMo2-1B-stage1.yaml` in
[sbordt/OLMo](https://github.com/sbordt/OLMo) — the same config the retain
stream is built from, so the sweep and the retain loader agree by construction:

```yaml
global_train_batch_size: 512
device_train_microbatch_size: 4
max_sequence_length: 4096
seed: 6198          # no data.seed -- build_olmo_retain_dataset falls back to this
d_model: 2048
n_layers: 16        # the source of RMU's l=7 depth mapping
```

Two consequences worth knowing:

- **`--retain-seed-override` is not needed.** The config has no `data.seed`, so
  `build_olmo_retain_dataset` falls back to `cfg.seed = 6198` — the same
  permutation the continued-pretraining baselines saw.
- **512 matches OLMo in sequences, not tokens.** We truncate to
  `MAX_SEQ_LEN=1024` where OLMo trains at 4096, so one of our optimizer steps is
  ~524k tokens against OLMo's ~2.1M. Matching token throughput would mean
  `MAX_SEQ_LEN=4096`, at 4× the activation memory. Same reasoning applies to
  `MICRO_BATCH=4`: it is a quarter of the tokens per forward that OLMo assumes,
  so `MICRO_BATCH=16` is the token-equivalent and runs 4× fewer accumulation
  steps if throughput matters more than matching the config literally.

Methods with a retain term draw a further `TOTAL_BATCH` retain sequences per
step, so they cost ~2× the FLOPs at the same step count. That is the
TOFU / OpenUnlearning convention: equal steps and equal forget exposure, not
equal compute. `n_total_tokens` is recorded per run so the asymmetry is a column
rather than a footnote.

## Quick start

```bash
# 0. Measure |D_f| first -- it sets the epoch sanity check and the eval's
#    --max-tokens. CPU only, no GPU, a few minutes.
python internal/uwiki/measure_forget_set.py

# 1. Always dry-run: this submits 32 jobs.
DRY_RUN=1 bash internal/uwiki/launch_pareto_sweep_1B.sh

# 2. Launch (galvani / ferranti)
bash internal/uwiki/launch_pareto_sweep_1B.sh

# 2'. Launch (MeluXina) -- same grid, same budget, different site wrapper
CELL_SCRIPT=internal/meluxina/unlearn_cell_1B.sh \
  bash internal/uwiki/launch_pareto_sweep_1B.sh

# A single cell
sbatch --export=ALL,METHOD=simnpo,VALUE=0.5 internal/uwiki/unlearn_cell_1B.sh

# A subset
METHODS="ce-u wga" bash internal/uwiki/launch_pareto_sweep_1B.sh
```

Start with one cheap cell before the full grid. `METHODS=ce-u EPOCHS=1` needs
neither the OLMo fork nor a frozen reference, and its log reports the real
forget-set token count.

## Script layout

```
internal/uwiki/unlearn_cell_body.sh      shared: dispatch, budget, argument construction
internal/uwiki/credentials.sh            shared: HF_TOKEN / WANDB_API_KEY resolution
internal/uwiki/unlearn_cell_1B.sh        site wrapper: galvani / ferranti
internal/meluxina/env.sh                 site environment: modules, venv, storage roots
internal/meluxina/unlearn_cell_1B.sh     site wrapper: MeluXina
internal/uwiki/launch_pareto_sweep_1B.sh the 32-cell grid, both sites
internal/uwiki/measure_forget_set.py     |D_f| and insertion-likelihood cost
```

Everything about *what* runs is in `unlearn_cell_body.sh`; everything about
*where* is in the site wrappers. Adding a site means writing one wrapper.

## Environment variables

Experiment-level, read by `unlearn_cell_body.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `METHOD` | *required* | `gradient-ascent \| grad-diff \| npo \| simnpo \| rmu \| ce-u \| wga \| satimp` |
| `VALUE` | *required* | This method's curve-knob value |
| `TOTAL_BATCH` | 512 | Forget sequences per optimizer step |
| `MICRO_BATCH` | 8 | Per-forward micro batch; accumulation derived |
| `EPOCHS` | per method | Overrides the published epoch count for **all** methods |
| `MAX_STEPS` | per method | Overrides the step ceiling for **all** methods |
| `HARD_STEP_CAP` | 10000 | Ceiling for methods with no step-based protocol |
| `MODEL` / `REVISION` | 1B-Exp-Unlearning @ step100k | Starting checkpoint |
| `FORGET_EXPS` | unset | Whitelist; unset means the library default |
| `LR` | per method | Learning rate, ignored where the LR *is* the curve knob |
| `DTYPE` / `FROZEN_DTYPE` | bfloat16 / float32 | |
| `GRAD_CKPT` | 0 | Set 1 if a cell OOMs |
| `RMU_LAYER` / `RMU_ALPHA` / `RMU_STEPS` | 7 / 1200 / 200 | RMU's fixed knobs |
| `RUN_TAG` | `1B-pareto` | Subdirectory under the output root |
| `OUTPUT_ROOT` | site default | Sweep output root |

Site-level, read by the wrappers:

| Variable | Default | Meaning |
|---|---|---|
| `PE_PROJECT` | `p200xxx` | **MeluXina project code — replace this** |
| `PE_REPO` / `PE_VENV` | under `/project/home/$PE_PROJECT/$USER` | Checkout and virtualenv |
| `PE_MUSE` | `env/release/2025.1` | MeluXina software stack |
| `HF_HOME` | `$PE_PROJECT_DIR/hf` | Model/dataset cache — keep off `$HOME` |
| `OLMO_CONFIG` | `$PE_PROJECT_DIR/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml` | Retain stream config |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset / `$HOME/.hf_token` | See below |

## Credentials

`internal/uwiki/credentials.sh` resolves tokens and is sourced by both site
wrappers. **Nothing is stored in the repository**; `.hf_token` and
`.wandb_token` are gitignored, and only the character count is ever echoed.

`HF_TOKEN` is **required** for `meta-llama/Meta-Llama-3-8B-Instruct`, the gated
judge that scores the denial-of-service eval (`denial_of_service.py:22`).
Elsewhere it is optional but useful — 35 eval tasks pulling the same weights hit
Hub rate limits without it.

Supply it in one of three ways, most preferred first:

```bash
# 1. Once, interactively. Writes ~/.cache/huggingface/token, picked up automatically.
huggingface-cli login

# 2. A private, untracked file.
printf '%s' 'hf_xxxxxxxx' > ~/.hf_token && chmod 600 ~/.hf_token
#    Override the path with HF_TOKEN_FILE=/path/to/file

# 3. Exported before submission; --export=ALL carries it into the job.
export HF_TOKEN=hf_xxxxxxxx
```

Same pattern for `WANDB_API_KEY` via `~/.wandb_token`. The unlearning drivers
write `metrics.jsonl` and do not use W&B; the YAML eval suite does.

## Output layout

```
unlearning-pareto/<RUN_TAG>/<method>/<knob>-<value>/
    <method>_config.json    run config snapshot, incl. forget/retain set info and tokens seen
    metrics.jsonl           per-micro-batch losses and method-specific diagnostics
    epoch-<N>/              HF-format checkpoints, one per epoch
```

The plot uses the **final** checkpoint of each cell, but every epoch is kept, so
a degenerate curve can be re-aggregated at an earlier epoch without retraining.

Diagnostics worth reading early in a run:

- **NPO** — `sigmoid_arg_mean`. Large and positive from step one means the loss
  is already saturated and that cell is a no-op.
- **SimNPO** — `sigmoid_arg_mean`, same reading.
- **WGA / SatImp** — `weight_mean` and `p_true_mean`. The weight peaks at
  `p* = β₁/(β₁+β₂)`; if the token probabilities sit far from `p*`, the
  reweighting is doing nothing.
- **CE-U** — `p_true_mean`. The gradient carries a factor `1/(1-p)`, so heavily
  memorized tokens amplify hard; raise `--min-forget-ce` if the first steps
  diverge.
- **GradDiff** — `ce_forget` climbing without bound *while* `loss_retain` also
  climbs means the cell has diverged, whatever the forget-side eval says.

## Evaluation

35 checkpoints — 32 cells plus 3 anchors — evaluated at the final checkpoint.
All nine categories run. The driver has **two tracks**, because the evals do not
share a calling convention:

| Track | Scripts | Convention |
|---|---|---|
| YAML suite (`EvaluationRunner`) | `insertion_likelihood`, `fictional_knowledge`, `verbatim_memorization`, `prompt_extraction` ×2, `benchmark` ×9, `mathematical_reasoning` ×3, `denial_of_service` ×2, `perplexity` | `--model`, `--results-yaml` |
| Direct invocation | `gaussian_watermark.py`, `newtoken_mia.py` | `--model_dir`, `--target_experiment`, own output shapes |

Prepare once, before the array fans out:

- **Watermark** — convert `sbordt/OLMo-2-1B-Exp-NoiseVectors` to the per-chunk
  pkl layout with `mia-data/build_noise_dir.py`. Warm it in a single job, or 35
  tasks race on the same conversion. `--noise_std 0.075`.
- **MIA** — `--reference_model auto` resolves to `sbordt/OLMo-2-1B` by parameter
  count. Share one `--reference_cache_dir` and warm it first, as
  `run_toaa_mia_newdataset_1B.sh` does. One call per `--target_experiment`.
- **Poison / DoS** — needs the gated judge and `HF_TOKEN`. Two variants
  (± trigger) × 35 checkpoints = 70 runs of an **8B** judge at `max_num_seqs=4`,
  on top of 1000 generations each. Give it its own array job rather than running
  it inline.

The Gaussian watermark is injected in embedding space, so none of the eight
text-based methods target it. Expect a flat line — it reads as a negative
control, not a trade-off curve.

### Anchors

| Anchor | Checkpoint | Reads as |
|---|---|---|
| Baseline | `OLMo-2-1B-Exp-Unlearning` @ `stage1-step100000-tokens210B` | No unlearning; where every cell starts |
| Unlearning baseline | `OLMo-2-1B-Exp-Unlearning` @ `stage1-step110000-tokens231B` | The same 10k-step budget spent on ordinary pretraining |
| Deep ignorance | `OLMo-2-1B-Unlearning` @ `stage1-step110000-tokens231B` | Never saw the insertions; the ideal corner |

## Installation

```bash
pip install -e .          # the framework
pip install -e ".[eval]"  # thefuzz, rouge-score -- needed by the eval scripts
```

The OLMo fork is needed **only** for the retain stream, which `olmo` provides via
`build_olmo_retain_dataset`:

```bash
git clone https://github.com/sbordt/OLMo && cd OLMo
git checkout pretrain-experiments && pip install -e .[all] && pip install h5py
```

So `gradient-ascent`, `ce-u` and `wga` run without it; `grad-diff`, `npo`,
`simnpo`, `rmu` and `satimp` need it.

On MeluXina, do all of this **inside a job** — login nodes have no user software
environment, so `module load Python` and `pip install` both fail there. The full
setup block is at the top of `internal/meluxina/env.sh`.

## Open items

1. **`|D_f|` is unmeasured.** It sets whether the 10k step cap truncates the
   10-epoch protocols, and what `insertion_likelihood --max-tokens` should be —
   that eval defaults to 100M tokens per experiment across ~57 experiments.
   `measure_forget_set.py` answers both.
2. **The eval sweep and aggregator are not written.** Two tracks, an array job
   with skip-if-exists, and a join into one tidy row per `(method, knob, value)`.
3. **RMU's ℓ=7 and c=6.5 / α=1200 were transferred, not measured.** No 179M
   anchor sweep was run, and the paper values are calibrated on instruction-tuned
   Llama-2 activations. The `c` grid brackets 6.5 from 2 to 10 for that reason.
4. **NPO's β range is a prediction.** NPO multiplies β into a *summed* sequence
   NLL where SimNPO uses a length-normalised one, so the paper's 0.1 should
   saturate immediately on 1024-token text. The grid runs three decades below it.
5. **WGA / SatImp weight detachment is unverified** against the authors' code.
   The driver detaches by default; `--differentiable-weight` tests the
   alternative.
6. **MeluXina unknowns**: whether compute nodes reach the network (offline
   toggles are pre-written in `env.sh`), and whether a 1B cell fits comfortably
   in 40GB. At `MICRO_BATCH=4` and 1024 tokens it should; if not, `GRAD_CKPT=1`
   or `MICRO_BATCH=2`, and accumulation rescales to hold the effective batch.
