# Week 2 — Corpus, partition, indexes and the frozen retrieval cache

**Plan subtasks:** 1.1 (corpus construction and client partitioning), 1.2 (indexing, caching)
**Outcome:** a **frozen data artefact** that every later week only reads.

---

## Run the whole week

```bash
python scripts/run_week2_pipeline.py          # corpus → partition → indexes → cache
python scripts/run_week2_pipeline.py --dev    # reduced scale smoke test
```

Or step by step:

```bash
python scripts/build_corpus.py
python scripts/partition_clients.py --seed 1001
python scripts/build_indexes.py --seed 1001
python scripts/precompute_retrieval.py --seed 1001
```

---

## Step 1 — The 200k corpus

Three components sharing one budget:

| Component | Source | Cap |
|---|---|---|
| **Gold passages** | Week 1 scan | `data.gold_per_question_in_corpus` (default 2) |
| **BM25 hard negatives** | lexically close but **answer-free** | `data.hard_negative_questions` × `hard_negatives_per_question` |
| **Random remainder** | `data/random_passages.jsonl` — the scan's **answer-free** reservoir | fills whatever is left |

Three things that are easy to get wrong here, and are therefore checked:

**The budget is a budget — all three components, not just gold.** 60k questions × up to 20 gold passages each would be
1.2M passages — six times the whole corpus. Gold is capped per question, and if
it still exceeds `corpus_size` the build **raises** rather than quietly returning
a corpus six times the intended size:

> `ValueError: gold passages (1,183,402) already exceed the corpus budget (200,000); lower data.gold_per_question_in_corpus ...`

The same check runs on **gold + hard negatives together**, because that sum is
what actually squeezes the remainder out. With the measured question count the
arithmetic is tight:

```
91,535 questions x 2 gold + 40,000 negatives = 223,070   > 200,000   -> no remainder
91,535 questions x 1 gold + 40,000 negatives = 131,535   -> 68,465 remainder   OK
```

so `data.gold_per_question_in_corpus` defaults to **1**. Overshooting used to
pass silently, producing a 223k corpus with zero random remainder — exactly the
failure the reservoir exists to prevent.

**The remainder must be a real remainder.** It is drawn from
`data/random_passages.jsonl` — the answer-free reservoir the Week 1 scan filled
in the same pass — never from the gold pile. Drawing it from gold would make
*every* passage in the corpus one that contains somebody's answer, and retrieval
difficulty would not be what the design intended. The script **refuses to run**
if that file is missing, and reports a `shortfall` (non-zero exit) if the pool
could not fill the corpus.

**BM25 does not finish by itself.** `rank_bm25` is pure python and scores the
entire corpus per query, so cost multiplies as *questions × candidates*:
60k questions against millions of passages does not terminate in any useful
time. Both sides are capped (`data.hard_negative_questions`,
`data.bm25_pool_size`), progress is logged, and the trade-off is explicit —
larger caps give harder negatives at proportionally more wall-clock. If gate (ii)
later shows a flat recall curve, these caps are the first thing to raise.

The critical constraint is in component 2. Because gold is identified by
answer-string containment, a "hard negative" that happens to contain the answer
string would be scored as a **successful retrieval**. Excluding answer-bearing
distractors is therefore what prevents false-positive recall, and
`verify_no_answer_bearing_negatives()` asserts it rather than assuming it — the
script exits non-zero if a single violation survives.

`bm25_hard_negatives()` over-fetches 4× before filtering, because for some
questions the BM25 top-20 is almost entirely answer-bearing and the target of
5 per question would otherwise be silently missed.

Passages are already fixed at 100 words, so **no chunking step exists**.

Questions whose gold passages all fell outside the final corpus are dropped here
rather than later; leaving them in would inflate the recall denominator with
questions nothing could have retrieved.

---

## Step 2 — Joint topic partition into 8 clients

Questions and corpus are partitioned **jointly, not independently**:

1. Cluster the 200k passages by topic (`MiniBatchKMeans`, 256 clusters on the
   bge embeddings).
2. Allocate whole clusters across the 8 clients, largest-first into the
   currently lightest client.
3. Assign each question to the client holding **most of its gold passages**, and
   restrict its gold set to that client's passages.
4. Trim to exactly equal question counts.

Two details carry the design:

* **Clusters are never split.** Topic disjointness is structural, not
  statistical — it cannot degrade with the seed.
* **Cluster weight is question count, not passage count.** What has to be equal
  across clients is the *training data*, not the index size.
* Step 3's restriction matters: a gold passage sitting on another client is a
  gold the client **cannot retrieve**. Leaving it in the recall denominator
  would measure partition luck instead of ranking difficulty. `partition_summary()`
  reports `gold_reachable_fraction` and the script warns loudly if it is below 1.

### Why not Dirichlet

A Dirichlet(α) partition varies two things at once:

| | Dirichlet | This design |
|---|---|---|
| Tightens topic range | ✅ intended | ✅ intended |
| Makes client sizes unequal | ❌ unintended averaging instability | held equal |
| Redraws per seed | ❌ injects partition variance into the seed estimate | fixed |

Client heterogeneity here is a **fixed background condition, not an experimental
factor** — the design stays one-dimensional. Two partitions (`seeds 1001, 1002`)
are built and assigned across the five run seeds so the result is not tied to
one allocation.

---

## Step 3 — Per-client Qdrant collections

Each client indexes **only its own passages** (`client_00` … `client_07`), in
Qdrant's local on-disk mode. A central index would both break topic disjointness
and contradict the deployment being modelled, where each institution runs RAG
over its own corpus.

`QdrantRetriever.search()` has the same signature as Week 1's
`BruteForceRetriever.search()`, so the gate code and the cache builder are
implementation-agnostic.

Corpus embeddings are computed once in `partition_clients.py` and written to
`data/corpus_embeddings.npy`, then reused for indexing — 200k passages are never
embedded twice. That single pass *is* the one-off indexing term of the Week 5
cost model.

---

## Step 4 — The frozen retrieval cache

```
results/cache/retrieval/client_00.npz … client_07.npz
```

For every question: the **common top-50 candidate pool** in dense order, the
dense scores, and the cross-encoder score of every candidate.

* **One pool serves all four depths.** `d=10`'s scores are a subset of `d=50`'s,
  so a single file answers every condition. Building per-condition caches would
  risk the conditions drifting apart in their candidate pool — the one thing the
  design must hold fixed.
* Storage is `int32` indices into a per-client pid table plus two `float32`
  score matrices, so 60k questions × 50 candidates stays small.
* Short candidate lists (possible only at reduced scale) are padded with `-inf`
  scores so padding can never be selected.

### The cache path carries the partition seed

```
results/cache/retrieval/seed_1001/client_00.npz … client_07.npz
results/cache/retrieval/seed_1002/client_00.npz … client_07.npz
```

The seed is **not** cosmetic and `cache_path()` has no default for it. The two
partitions give each client a different passage set (measured: 22,026–26,870 per
client at seed 1001, 22,710–28,164 at seed 1002), so the candidate pools differ
and the caches cannot be shared. Without the seed in the path, two things broke
at once:

* the second seed's `precompute_retrieval` **overwrote** the first seed's caches,
  discarding 66 minutes of reranking;
* Week 3 loaded the partition by seed but the cache without one, so whichever
  seed ran last supplied candidates for *every* run — clients received candidate
  lists for passages they do not hold, with nothing in the output to say so.

The second is why the parameter is required rather than defaulted: a default lets
a caller forget the seed and still run.

The committed `results/logs/week2/retrieval_cache_*.json` were produced by the
run that built this artefact, **before** the seed was added to the cache path, so
their `cache` fields record the old flat location
(`results/cache/retrieval/client_00.npz`). The measurements are unaffected — only
the recorded path string predates the fix, and the caches were moved into
`seed_1001/` and `seed_1002/` afterwards.

`precompute_retrieval.py` immediately re-measures **gold recall@3 at every
depth** from the cache it just wrote, plus the number that makes those
interpretable: **`gold_recall_at_pool`**, the fraction of questions whose gold
passage is anywhere inside the top-50.

That is the **ceiling for every condition**. Reranking reorders the pool; it
cannot reach outside it, so no depth can beat it — `d = 50` included. Without it,
"`d=50` reached recall 0.53" is unreadable: against a ceiling of 0.55 that is
near-perfect, against a ceiling of 0.95 it is poor. `headroom_captured` reports
the ratio directly.

It also says which knob to turn. A low ceiling means first-stage retrieval is
missing the evidence, and **more depth cannot help** — raise
`retrieval.candidate_pool` instead. This is the same monotonicity property
gate (ii) checked in Week 1, now verified on the real artefact rather than a
scratch index.

### Caching is not a discount

All runs sharing one retrieval pass is an **implementation optimisation**. The
cost model in Week 5 still charges the full deployment-time per-query reranking
cost. To keep that honest, the cache records the *measured* reranker forward
passes and wall-clock seconds in its metadata and in
`results/logs/week2/retrieval_cache_<seed>.json`.

---

## Outputs

| Path | Contents |
|---|---|
| `data/corpus.jsonl` | the 200k passages |
| `results/logs/week2/corpus.json` | composition, shortfall, violation count |
| `data/questions_corpus.jsonl` | questions with at least one gold in the corpus |
| `data/corpus_embeddings.npy` | frozen bge embeddings (indexing + clustering) |
| `data/partition_{1001,1002}.json` | client → questions, client → passages |
| `data/qdrant_storage/seed_*/` | per-client collections |
| `results/cache/retrieval/seed_*/client_*.npz` | **the frozen retrieval artefact**, one set per partition |
| `results/logs/week2/*.json` | composition, partition summary, recall by depth |

```bash
pytest -q tests/test_corpus_build.py tests/test_partitioning.py tests/test_retrieval_cache.py
```
