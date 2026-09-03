# System Architecture

## Decision Summary

This is a modular monolith: FastAPI, SQLite, a static same-origin web client and
Python offline jobs in one repository. It minimizes integration state while
keeping data, serving, recommendation, operations and UI ownership explicit.
Redis, a task queue and a separate frontend build are intentionally excluded.

```text
official MicroLens metadata (ignored)
        |
        v
process -> chronological splits + summaries -> stable SVD/content/CF artifact
                                      |              |
                                      |              +-> PyTorch DSSM/DeepFM experiment
                                      +-> covers -> MobileNet/PCA text-image experiment
                                                            |
                                                            v
browser -> FastAPI auth/feed/event/admin -> recommendation engine -> validated pointers
                    |                         |
                    +------ SQLite facts -----+
                              |
                              +-> live profiles / dashboard / operations audit
```

## Offline Data Flow

The processor validates exact columns and types, joins item metadata, sorts by
server timestamp, then uses global timestamp quantiles for train/validation/test.
All rows at or before the train cutoff are train; later rows through the
validation cutoff are validation; remaining rows are test. Cutoff assertions
prove `max(train) <= min(validation) <= min(test)`. Item popularity, vocabularies,
negative sampling pools and learned parameters use train only. Validation metrics
support policy selection; only validation compares the bounded candidate-policy set,
and test runs the single locked policy once for the final report. This is not a
general hyperparameter search.

The cumulative likes/views file is versioned as a source snapshot with
`available_at`, SHA-256, source name, ingest time and quality counts. Missing
`available_at` makes it unavailable to historical training/evaluation. Popularity
features instead use interactions at or before an explicit cutoff and materialize
cumulative, 1/7/30-day, time-decay and recent-growth values.

The learning stack is implicit-feedback TruncatedSVD plus train-vocabulary
character n-gram TF-IDF content profiles and train-cutoff item-item cosine CF. The SVD operates over the train-only sparse
user-item matrix. It is simple enough for the deadline, genuinely learned,
explainable as user/item latent affinity, deterministic with a fixed seed, and
directly usable online. Popular and seeded random baselines use the same
evaluation cohort and negative sets.

Stable, multimodal and deep evaluation share protocol
`deterministic_sampled_negatives_v1`: validation/test retain all eligible positives,
including cold catalog items, and attach 100 unique seeded negatives per user. Query
hashes must match the stable artifact before a deep model can pass its release gate.
Complete-catalog replay remains an optional `--validation-mode full` diagnostic and
is deliberately not release-compatible with sampled metrics.

Artifacts never use pickle. `manifest.json` references NumPy arrays and a
popularity file with hashes, shapes, data version, training configuration,
metric definitions, results and anonymized query-level SVD bad cases. Publication validates every file before an
atomic `artifacts/current.json` replacement. A failed training or load does not
replace the last usable pointer; serving falls back to popular/explore.
The manifest also records `mix_policy`: schema/version, validation-only search
evidence, quality gates and the locked policy consumed by online Feed. Unknown
policy versions reject the new artifact and preserve the previous usable model.

Deep models are isolated from the stable pointer. `recsys.deep` trains a PyTorch
DSSM with deterministic mixed negative sampling, precomputes catalog embeddings,
then trains DeepFM over the deduplicated SVD/DSSM/content/visual/item-CF/popular/
explore union. Its 17 continuous fields are seven normalized source scores, seven
source-presence flags, history density, cold-item and visual-availability; categorical
fields cover user, item/UNK, popularity bucket, history bucket and primary source.
Deterministic train-only identity/collaborative dropout teaches cold-start fallback to
content and vision. Each epoch writes safetensors plus optimizer/seed/cutoff/config
metadata; patience-based validation early stopping exports only the best checkpoint.
`app.deep_artifacts` pins the multimodal version, verifies hashes and warms the model.
The unified 7-source + DeepFM path is the sole personalized production path; a missing,
corrupt or incompatible deep/visual artifact falls back to popular/explore rather than
a separate stable SVD mix. On the sampled-all-items protocol the pure-DeepFM ranking
improved NDCG but regressed Recall and HitRate (missing the Recall release gate); this
trade-off is accepted and recorded in `docs/VERIFICATION.md`.

`recsys.vision` audits and safely extracts the ignored official cover archive,
uses pretrained MobileNetV3-Small offline, fits PCA128 on train-visible covers only,
and builds cutoff-safe visual user profiles. Text/visual rank fusion is selected on
validation and the locked weight is evaluated once on test. Serving loads cached
vectors only; missing images or an invalid visual artifact fall back to text/deep
ranking. Raw covers, weights, checkpoints and embedding caches remain outside Git.

## Online Request Flow

1. Resolve an opaque session cookie to exactly one active server-side user.
2. Load one fixed model version for the request.
3. The unified 7-source path is the sole personalized path. It builds SVD, DSSM,
   content, visual, item-CF, popular and explore candidates with explicit
   eligibility/confidence/support, removes seen, negative, offline and
   zero-support CF rows before source-local normalization, unions and deduplicates
   them, then ranks the open union with a single DeepFM. SVD is a recall source
   here, not an independent serving path. When the deep/multimodal artifacts are
   missing, incompatible, or the model errors, personalized falls back to popular
   (filled with explore on shortfall) instead of a stable SVD mix.
4. Apply per-Top-10 source upper bounds (never minimum quotas), title-token MMR,
   legal boosts, and the final offline guard before snapshot persistence.
5. Deduplicate and filter prior viewable/explicit behavior.
6. Apply the authoritative `items.status = online` filter.
7. Deterministically rerank ordinary candidates with title-token MMR when metadata
   passes the quality gate, then insert active in-scope boosts at fixed positions.
8. Repeat the online filter and, on page one, persist a bounded ordered snapshot
   plus before/after diversity metrics with model/profile/operations
   versions and expiry; later signed cursors read the same snapshot by offset.
9. In one transaction write each recommendation request and ordered served exposures.
10. Return request_id, snapshot_id and item-level provenance.

Rule precedence is `offline > valid boost > dedup/safety > locked mixing policy >
diversity > ordinary ranking`. A valid boost may intentionally repeat a seen item for demonstration;
it can never revive an offline item.

## Event and Profile Flow

An exposure means the server returned an item. An impression means at least 50%
of its card remained in the viewport for 750 ms. The client batches viewable
impressions, retries finitely and uses `sendBeacon` on page hide/unload.
After the viewability threshold it accumulates dwell until viewport exit/page
hide, and reports a successful Web Share or link copy as `share`. The server
derives `visit_index` and an idempotent `revisit` event across distinct requests.

Client events do not contain a trusted user ID. The session supplies identity;
the server verifies request, item and position against that user's exposure.
`event_id` is an idempotency key. In one transaction the server writes the event,
updates the user-item state and increments `profiles.version`. Click and like add
positive item affinity; dwell uses bounded duration buckets, and share/revisit
are positive signals. Positive affinity is capped per item and not_interested
dominates later positive events. The next
personalized request reads this state synchronously. `app.cli retrain-events`
uses a half-open authoritative receive-time window, maps impression/click/like/
not_interested to context or signed feedback, rejects invalid catalog mappings,
and regenerates chronological train/validation/test splits. It records a new data
version, training run and model version. Evaluation/load failure leaves the prior
artifact pointer and active DB version unchanged. The CLI path is synchronous. An
administrator-only HTTP endpoint can enqueue the same flow on an in-process daemon
thread and persists queued/running/succeeded/failed state. It is suitable for this
single-process Demo, but process loss can interrupt work and it must not be described
as a durable external worker queue.

## Operations Flow

Only an administrator session can mutate item status or boost campaigns. Status
change and its before/after audit row share a transaction. Every item-returning
path, including direct item lookup and fallback, calls the same online filter.
Offline marks matching snapshot rows invalid immediately; restoring an item does
not reactivate it in old snapshots. Batch status operations validate all IDs first,
commit atomically, deduplicate IDs, bind retries to an administrator idempotency
key and write one audit row per item under a shared `batch_id`. New boost/restore/model/profile state appears
only in a refreshed snapshot. Dashboard data comes from users, recommendation requests, exposures, events,
items, model versions and operations, never from frontend constants. Per-feed
breakdown distinguishes served exposures, viewable impressions, served CTR and
viewable CTR, alongside clicks, likes and negative feedback.
Latency percentiles and hourly/daily trends use the same server-side facts.

## Core Entities

- `users`, `auth_sessions`: identity, roles and opaque session lifecycle.
- `items`, `item_stats_snapshots`: catalog state and versioned likes/views provenance.
- `feed_snapshots`, `feed_snapshot_items`: bounded TTL candidate snapshots and diversity facts.
- `recommendation_requests`, `exposures`, `events`: request, served and viewable facts.
- `user_item_state`, `profiles`: synchronous online preference state.
- `boost_campaigns`, `operations`, `operation_batches`: scoped rules, atomic batches and append-only audit.
- `model_versions`, `training_runs`: data/model versions, windows, metrics, artifacts and publication attempts.

SQLite connections enable foreign keys, WAL and a busy timeout. Feed facts,
behavior/profile updates and operations/audits each use explicit transactions.

## Failure Modes

- Missing/corrupt model: popular or explore fallback with a response reason.
- Failed training/publication: retain `current.json` and the active DB version.
- Empty candidates: relax seen filtering, still apply offline and deduplication.
- Offline between candidate generation and persistence: final transaction-time
  status filter removes it.
- Duplicate/out-of-order event: idempotent by event_id; server receive time drives
  metrics while client time remains diagnostic.
- Database unavailable/locked past timeout: return an explicit error; do not
  fabricate a response or metric.

Every HTTP response carries an API request ID. The access logger emits one JSON
record with request ID, method, path, status and duration, independently from the
feed request ID used for recommendation exposure lineage.

## Security Boundary

Passwords use salted `hashlib.scrypt`. Only a SHA-256 digest of each random session
token is stored; the plaintext token is an HttpOnly, SameSite cookie. User IDs are
never trusted from normal-user request bodies. Admin authorization is checked in
the service layer for every `/api/v1/admin` route. Secrets come from environment
variables and raw data/artifacts are ignored.
