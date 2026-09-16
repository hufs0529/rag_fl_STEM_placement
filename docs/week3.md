# Week 3 — The federated pipeline, and rehearsing it

**Plan subtask:** 1.2
**Outcome:** a rehearsed pipeline — every path from cached retrieval to a logged
learning curve exercised at reduced scale before a single main run is spent.

---

## What one round actually does

```
server                                   client i (×8, full participation)
──────                                   ─────────────────────────────────
broadcast global (A, B)   ──────────────▶ load adapter
                                          for K steps:
                                            take a question from its own partition
                                            read its CACHED top-50 pool
                                            select 3 passages at depth d     ← the treatment
                                            build prompt, mask prompt tokens
                                            loss on answer tokens only
                                            AdamW step (state persists across rounds)
aggregate (FedAvg)        ◀────────────── return adapter
record bytes / drift / diagnostics
evaluate globally if this round is on the schedule
```

Only LoRA `A` and `B` cross the wire: **1.84 MB per client per direction**.
Reranking changes the *training inputs*, never the model, and its effect
propagates through gradient quality.

---

## Run it

```bash
# reduced-scale rehearsal with synthetic artefacts (no 21M-passage download)
python scripts/run_rehearsal.py --dev --synthetic

# one real run, once Week 2's artefacts and gate 3's schedule exist
python scripts/run_single_run.py --depth 10 --seed 1
python scripts/run_single_run.py --depth 0 --seed 1 --backend inprocess
```

---

## Design decisions that carry the measurement

### Fixed budget, no early stopping

`eval.early_stopping: false`, and `run_experiment` has no stopping branch at all.
Every condition consumes the same `R`. Conditions are compared on **what that
budget buys**, and rounds-to-target are recovered in Week 5 by interpolating the
curves — so the result never depends on a stopping rule. This is the direct
answer to "LoRA converges in few rounds, so round count is too coarse a ruler".

### K, S and R are injected, not configured

`configs/experiment_config.yaml` leaves `local_steps_per_round`,
`total_local_steps` and `rounds` as **`null`**. `resolve_schedule()` reads
`results/logs/gates/calibrated_schedule.json` written by gate 3, and **raises**
if neither that file nor explicit values exist:

```
RuntimeError: K/S/R are undetermined: run scripts/gate3_local_step_calibration.py
first ... (the plan fixes these by pilot, not by convention).
```

The claim "a documented pilot fixed these" is enforced by the code.

### The cache must match the partition seed

`ExperimentData` loads the partition by seed and, until this was fixed, the
retrieval cache **without** one:

```python
self.partition = load_json(data_dir / f"partition_{partition_seed}.json")
self.caches    = {cid: RetrievalCache.load(cache_path(cache_dir, cid))}   # no seed
```

The two partitions hand each client a different passage set, so whichever seed's
`precompute_retrieval` ran last supplied candidates for *every* run. A client
would be trained on candidate lists drawn from passages it does not hold, and
nothing in the logs would say so — `rank_change_rate` and `promotion_rate` would
look normal, because the selection logic was working correctly on the wrong
input.

`cache_path(cache_dir, client_id, partition_seed)` now takes the seed and has no
default for it, so a caller that forgets it fails immediately instead of reading
a mismatched cache.

### The rehearsal writes to its own directory

`make_dev_artefacts.py` fabricates a corpus, partitions and caches. It wrote them
to the **production paths**, because the dev overlay shrank the scale but left
`paths` alone. One rehearsal therefore destroyed Week 2's output:

| Overwritten | Was | Became |
|---|---|---|
| `data/corpus.jsonl` | 200,000 passages | 528 |
| `data/questions_corpus.jsonl` | 80,720 questions | 128 |
| `data/partition_{1001,1002}.json` | 8 clients | 2 |
| `results/cache/retrieval/seed_*/` | the measured caches | synthetic |

A rehearsal exists to exercise the wiring **without** spending the real
artefacts; this one spent them. `configs/dev_config.yaml` now redirects every
path:

```yaml
paths:
  data_dir: data/dev
  cache_dir: results/dev/cache
  log_dir: results/dev/logs
```

Verified by running `make_dev_artefacts.py --dev` and checking that production
`data/` is byte-identical afterwards.

What made the recovery cheap is that `build_corpus.py` is **deterministic** given
the same three scan inputs and seed, so the regenerated corpus has the same
passages in the same order — `corpus_embeddings.npy` and the Qdrant collections
stay valid, and only the caches that were actually overwritten need rebuilding.
The embedding fingerprint decides that automatically rather than by assumption:
if the regenerated corpus differs in any passage, the digest changes and the
embeddings are rebuilt.

### train / val / test are three files, and disjoint

Until this was fixed, none of the three existed on disk — the code reproduced
them deterministically at run time, so there was nothing to point at — and two of
them overlapped.

| Set | Role | Where |
|---|---|---|
| **train** | what the clients learn from | each client's partition minus `test` |
| **val** | **the questions that fixed the settings** — the Week 1 gate probes | `data/split_val_qids.json` (2,000) |
| **test** | the held-out set the final comparison is reported on | `data/split_test_{seed}.jsonl` (1,000) |

`val` and `test` were each drawn independently from the same 80,720 questions, so
they overlapped by **24** (expected 24.8). No model that produces a result was
ever trained on `test` — `split_train_eval` removes it, gate (ii) only measured
retrieval, and gate (iii)'s model was discarded — but "are the three disjoint?"
could not be answered with a yes. `build_eval_set` now takes `exclude` and
`eval_questions()` passes the `val` ids, so the answer is yes.

`scripts/make_splits.py` writes both files and asserts the invariant. The `val`
ids are reconstructed from the gates' own RNG call,
`random.Random(seed).sample(questions, n)`, unioned over the sample sizes the
gates used — the union matters because nesting is **not** guaranteed: CPython's
`random.sample` switches algorithm with the ratio of `k` to population size, so
at 80,720 questions the three samples nest (union 2,000) while at 5,000 they do
not (union 2,009). Excluding a few thousand extra questions costs nothing when
`test` draws 1,000 from 78,752.

### Evaluation is central, greedy, and held at `d = 0`

Three sources of variance are removed by construction:

| Source | Removed by |
|---|---|
| Sampling | greedy decoding |
| Evaluation set | one fixed 1,000-question set, selected independently of the run seed |
| Inference-time context | **`eval.eval_depth: 0` for every condition** |

The third is the one that makes the claim about *learning* rather than *serving*.
If the treated conditions were also evaluated with deeply reranked context, the
measured gain would partly be "better context at inference", which has nothing to
do with round count. The rehearsal asserts this: `eval_gold_recall_at_3` must be
**identical across all four depths**, because the evaluation path never sees `d`.

### Optimiser state persists across rounds

Flower may recreate the client object every round, so the optimiser lives in
`ClientState`, held in a dict outside the client. With small `K`, recreating
AdamW each round would leave it permanently in warm-up and confound the effect of
`K` with the effect of restarting the optimiser. `LocalTrainer` is the **same
class** gate 3 calibrated `K` on — a `K` chosen on a different loop would be
meaningless.

### One model instance, eight clients

Clients differ only in their adapter and their data, so the 135M base model is
instantiated once and parameters are swapped per client. Eight copies would make
memory, not science, the limit on the experiment's scale.

### Why the runs are `inprocess` and not Flower simulation

The plan specifies Flower simulation mode. **It cannot run this design**, and the
reason is worth stating rather than hiding: Ray-based simulation puts each client
in its own process and serialises what it sends there, while two deliberate
choices here keep state in the parent process.

| Choice | Why | Conflicts with Ray because |
|---|---|---|
| One 135M model instance, adapters swapped per client | eight copies would make memory, not science, the limit on scale | each actor would need its own copy |
| Optimiser state in `ClientState`, held in a dict here | small `K` must not leave AdamW permanently in warm-up, which would confound `K` with optimiser restarts | an actor cannot reach this process's dict |

Attempting it fails at serialisation, with three separate blockers:

```
TypeError: cannot pickle 'itertools.cycle' object
```

* `LocalTrainer`'s cycling iterator over its dataloader;
* `build_dataloader`'s `lambda` collate_fn;
* the `ClientState` dict itself.

**The measurement is unaffected.** Both backends call the same
`aggregate_round()` and the same `RoundRecorder`, and with full participation
(`participation: 1.0`) visiting the eight clients sequentially produces the same
FedAvg aggregate, the same byte count and the same drift record as visiting them
in parallel. Only wall-clock differs. `backend` is logged in every run's
metadata, so no run's provenance is ambiguous.

Making Flower work would mean shipping model and optimiser state to actors every
round — more memory and more wall-clock for no change in what is measured. The
trade was declined; `federated.backend` defaults to `inprocess` and selecting
`flower` is refused **before** the model loads, with the explanation above,
rather than crashing after round 0's evaluation has already been spent.

This was not caught earlier because `run_rehearsal.py` defaulted to `inprocess`:
the Flower path had never been executed. The rehearsal now exercises whichever
backend the config selects.

---

## What gets logged, every round

`results/logs/runs/d{depth}_s{seed}.jsonl`, one line per round:

| Field | Why the plan needs it |
|---|---|
| `cumulative_bytes` | `R x N x |AB| x 2` — the quantity being traded |
| `f1`, `em` | the curve (on scheduled rounds only) |
| `drift_divergence`, `drift_mean_pairwise_cosine` | the **mechanism check** |
| `rank_change_rate`, `promotion_rate` | confirms the treatment took effect |
| `eval_gold_recall_at_3` | confirms the evaluation path stayed at `d=0` |
| `train_loss`, `wall_clock_seconds` | sanity and cost |

Analysis in Week 5 reads **only these files**, so the τ sweep can be redone
without retraining anything.

---

## The rehearsal's checks

`scripts/run_rehearsal.py` runs one short run per depth and asserts:

1. `R` rounds logged, curve points present on the scheduled rounds;
2. `cumulative_bytes` equals `R × N × |AB| × 2` computed independently;
3. drift recorded every round;
4. **treatment took effect** — `promotion_rate > 0` for `d > 3`;
5. **control promotes nothing** — `promotion_rate == 0` for `d ∈ {0, 3}`;
6. **evaluation context fixed** — `eval_gold_recall_at_3` identical across depths.

Checks 4–6 are the ones worth having. They fail loudly if the treatment silently
stopped being applied, or if the treatment leaked into evaluation — two bugs that
would otherwise produce a perfectly plausible-looking result.

`scripts/make_dev_artefacts.py` fabricates structurally identical toy artefacts
(gold deliberately dense-ranked 20th–45th, so recall genuinely rises with depth)
so the wiring can be rehearsed offline. Its output is labelled `synthetic: true`
in the cache metadata and is never a result.

---

## Modules introduced this week

| Module | Role |
|---|---|
| `src/schedule.py` | the fixed dense-early/sparse-late evaluation schedule |
| `src/evaluate.py` | central greedy evaluation at `d=0`, plus eval-time recall diagnostic |
| `src/experiment.py` | `RunSpec`, seed→partition assignment, gate-3 schedule injection |
| `src/fl_client.py` | Flower client; persistent `ClientState` |
| `src/fl_server.py` | FedAvg, `RoundRecorder` |
| `src/fl_runner.py` | one full run, both backends |

```bash
pytest -q                 # unit tests, no GPU
pytest -q -m slow         # tiny-GPT2 + LoRA wiring test (needs torch/peft)
```
