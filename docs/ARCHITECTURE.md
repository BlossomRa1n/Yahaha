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
process -> chronological splits + summaries -> SVD/baselines -> versioned artifact
                                                            |
                                                            v
browser -> FastAPI auth/feed/event/admin -> recommendation engine -> NumPy artifact
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
support manual configuration comparison; test is evaluated once for the final report.
This MVP does not claim automatic hyperparameter search.

The learning model is implicit-feedback TruncatedSVD over the train-only sparse
user-item matrix. It is simple enough for the deadline, genuinely learned,
explainable as user/item latent affinity, deterministic with a fixed seed, and
directly usable online. Popular and seeded random baselines use the same
evaluation cohort and negative sets.

Artifacts never use pickle. `manifest.json` references NumPy arrays and a
popularity file with hashes, shapes, data version, training configuration,
metric definitions, results and anonymized query-level SVD bad cases. Publication validates every file before an
atomic `artifacts/current.json` replacement. A failed training or load does not
replace the last usable pointer; serving falls back to popular/explore.

## Online Request Flow

1. Resolve an opaque session cookie to exactly one active server-side user.
2. Load one fixed model version for the request.
3. Build candidates for personalized, popular or explore.
4. Apply online profile weights and negative feedback.
5. Insert active in-scope boost campaigns.
6. Apply the authoritative `items.status = online` filter after all merging.
7. Deduplicate, filter prior exposures for ordinary candidates, and paginate.
8. In one transaction write the recommendation request, ordered exposures and
   server-generated impression events.
9. Return the persisted request_id and item-level provenance.

Rule precedence is `offline > valid boost > user seen/not_interested > ordinary
ranking`. A valid boost may intentionally repeat a seen item for demonstration;
it can never revive an offline item.

## Event and Profile Flow

Client events do not contain a trusted user ID. The session supplies identity;
the server verifies request, item and position against that user's exposure.
`event_id` is an idempotency key. In one transaction the server writes the event,
updates the user-item state and increments `profiles.version`. Click and like add
positive item affinity; not_interested applies a strong negative weight. The next
personalized request reads this state synchronously. `app.cli export-events`
writes a staging-only `user,item,timestamp,event_type,weight` snapshot for mapped
dataset users. The current benchmark deliberately does not consume it: a future
job must establish a new chronological cutoff and regenerate validation/test
before merging online periods. The MVP does not claim automatic or online training.

## Operations Flow

Only an administrator session can mutate item status or boost campaigns. Status
change and its before/after audit row share a transaction. Every item-returning
path, including direct item lookup and fallback, calls the same online filter.
Dashboard data comes from users, recommendation requests, exposures, events,
items, model versions and operations, never from frontend constants. Per-feed
breakdown includes requests, exposures, clicks, likes and negative feedback.
Latency percentiles and hourly/daily trends use the same server-side facts.

## Core Entities

- `users`, `auth_sessions`: identity, roles and opaque session lifecycle.
- `items`: official metadata and authoritative online/offline state.
- `recommendation_requests`, `exposures`, `events`: immutable request facts.
- `user_item_state`, `profiles`: synchronous online preference state.
- `boost_campaigns`, `operations`: scoped delivery rules and append-only audit.
- `model_versions`: data version, metrics, artifact path and publication state.

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
