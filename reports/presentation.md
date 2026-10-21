# Presentation outline

*Generated on 2026-10-21.*

## Slide 1 — The overlooked factor

Federated communication = **bytes per round × rounds required**.
Everyone shrinks the first. This project attacks the second — from retrieval.

## Slide 2 — The coupling nobody measured

Institutions that cannot centralise data deploy RAG, so training should retrieve
too. When retrieval succeeds every client learns the same task; when it fails
each client adapts to its own pattern of irrelevant context — and non-IID updates
diverge and slow convergence.

## Slide 3 — One axis: reranking depth

`d ∈ [0, 3, 7, 10, 25, 50]`, one common top-50 pool, exactly three passages in
every prompt. `d = 3` reorders without promoting — an ordering control.
Depth is a **recurring** per-query cost, so it is commensurable with the
per-round communication it trades against.

## Slide 4 — Measuring without a stopping rule

Fixed budget, no early stopping, curves interpolated. LoRA converges in a few
rounds, so integer round count is too coarse a ruler. `R = S / K`, and `K` was
fixed by a pilot that found the smallest `K` with measurable client drift.

## Slide 5 — Result

At the representative target of 25.02 F1, reranking at d = 50 was 0.15 rounds SLOWER than dense retrieval alone: the treatment cost communication rather than saving it. The largest rounds-to-target saving is inside its own uncertainty, so this measurement resolves no saving and there is no mechanism to explain.

At τ = 25.02: **-4.3 MB** saved at d = 50.

## Slide 6 — The exchange rate, not a verdict

Bytes saved per extra FLOP → break-even bandwidth. Reranking pays on slow links
and loses as query volume grows. The deliverable is an **operating regime**.

## Slide 7 — Honest limits

Five seeds, one model scale, simulated clients, answer-string gold labels, and
a cost model whose assumptions are stated rather than hidden.
