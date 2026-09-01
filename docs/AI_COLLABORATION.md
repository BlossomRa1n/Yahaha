# AI Collaboration and Review Log

This log records delegated work as evidence, not as proof of completion. The lead
agent marks implementation complete only after inspecting diffs and rerunning
tests.

## Audit Round

### Data and algorithm agent

Prompt scope: read-only review of the assessment; identify official data format,
leakage-safe split, CPU model/baselines, artifact contract, metrics and risks.
Allowed files: none. Required evidence: commands and source facts.

Lead review: accepted global timestamp boundaries, train-only popularity and the
TruncatedSVD recommendation. The agent also caught that untimed likes/views are
a present-day snapshot and must not be used as an offline feature.

### Backend agent

Prompt scope: read-only review of authentication, SQLite facts, feed/event/profile,
operations, audit, dashboard and API contracts. Allowed files: none. Required
evidence: commands and source facts.

Lead review: accepted the opaque session, transaction and final offline-filter
invariants. Rejected a mandatory SQLAlchemy/Vite split because the blank project
and environment make a standard-library SQLite layer plus same-origin static UI
lower risk for the two-day scope.

### Frontend and delivery agent

Prompt scope: read-only review of pages, states, API typing, browser/API tests,
clean-start delivery, README and demonstration. Allowed files: none. Required
evidence: commands and source facts.

Lead review: accepted the workflow and error-state matrix. Replaced the proposed
React/Vite client with static modular JavaScript to remove an unnecessary build
dependency while preserving the same API and UX surface.

## Implementation Rounds

The entries below summarize the implementation prompts; they are not claimed to
be verbatim transcripts. No numerical AI contribution percentage was measured.
Agent-produced diffs and summaries were treated as inputs to review, not as proof
of completion.

### Ownership and integration contract

| Role | Owned files | Must not change |
|---|---|---|
| Data/algorithm | `recsys/**`, `tests/test_data.py`, `tests/test_model.py` | Backend, Web and delivery documents |
| Backend/service | `app/**`, `tests/test_api.py` | Recommender training code, Web and delivery documents |
| Frontend/delivery | `web/**`, `README.md`, `docs/DEMO.md`, `docs/VERIFICATION.md`, `tests/test_web_contract.py`; `docs/AI_COLLABORATION.md` for closeout only | `app/**`, `recsys/**`, dependency metadata and frozen contracts |
| Lead architect | Frozen contracts, shared architecture/requirements, integration and final evidence | No specialist claim accepted without diff/test/runtime review |

Before parallel implementation, the lead froze `/api/v1` paths, the SQLite fact
model, event schema, model artifact layout and rule precedence in `docs/API.md`
and `docs/ARCHITECTURE.md`. Specialists modified separate files; cross-module
changes were coordinated through the lead or direct agent messages.

### Data and algorithm implementation

Prompt summary:

- Build a deterministic official-data parser, global chronological split, quality
  summary and one-command pipeline.
- Train popular/random baselines and a CPU learnable model without leakage; use a
  versioned, hash-checked, no-pickle artifact that the backend can consume.
- Modify only the assigned `recsys/**` and data/model test files. Return exact
  commands, output shapes, metrics, timing and remaining evaluation risks.

Agent output:

- Implemented official pairs/titles/likes-views parsing, deterministic split
  files, histories and summary; untimed likes/views were excluded from offline
  features.
- Implemented train-only TruncatedSVD plus shared-query popular/random baselines,
  `K=10`, 100 deterministic negatives and macro-user evaluation.
- The full agent run exited 0: prepare `1.755s`, total `19.323s`; final current
  model `svd-20260901T121430026505Z-cccf5c24` contains 49,416 users, 16,907 items
  and rank 32. Test SVD metrics were Recall `0.359176`, NDCG `0.208986`, HitRate
  `0.399000` over 5,000 users.
- The agent reported deterministic reruns, publication rejection for corrupt
  artifacts and 15 passing repository tests.

Review and corrections:

- Delivery review independently parsed `summary.json`, `current.json`, the
  referenced manifest and metrics instead of copying metric text from the agent.
- The lead retained strict train-only popularity and the explicit leakage checks.
  The documentation was corrected to say validation supports manual comparison;
  this MVP does not automatically search hyperparameters.
- Warm-item test coverage is only `0.237160` and remains disclosed. The online
  system therefore needs popular/explore fallback for time-new content.
- Online events are not automatically exported into a new offline training run;
  this remains a documented batch-integration gap.

### Backend and recommendation-service implementation

Prompt summary:

- Implement opaque HttpOnly sessions, 3 normal users plus an administrator,
  SQLite facts, three feeds, transactional exposures/events/profile updates,
  Dashboard aggregation and server-authoritative operations.
- Consume the frozen artifact; preserve the last usable version and fall back
  when loading fails. Enforce `offline > boost` in every return path.
- Modify only `app/**` and `tests/test_api.py`; return API tests and real artifact
  smoke evidence, not fixed recommendation JSON.

Agent output:

- Implemented FastAPI routes, scrypt passwords, hashed session tokens, HMAC-bound
  cursors, strict request schemas, WAL/foreign keys and explicit transactions.
- Feed responses persist request, ordered exposures and server impressions in one
  transaction. Client events validate session ownership plus request/item/position,
  normalize favorite to like and update the profile synchronously.
- Implemented SVD/personalized, popular and explore candidates, scoped boosts,
  final online filtering, real Dashboard SQL, direct-item filtering and append-only
  operation audit.
- Backend tests cover auth/logout/403, isolation, Feed facts, event idempotency,
  profile changes, Dashboard, boost/offline/restore and corrupt-model fallback.
  The agent reported 15 passing repository tests and a real-artifact SQLite smoke.

Lead review and interface fixes:

- The original Dashboard response omitted the required popular-content aggregate.
  Frontend review raised this before integration; backend added real
  `top_items` aggregation.
- Request diagnostics initially returned request/exposure items but not linked
  behavior. Backend added the associated events so request_id can trace both.
- A user-debug form requiring an unknown internal UUID was not operable. Backend
  made the existing debug path accept exact usernames and added an admin user list;
  the boost form then selected server-provided user IDs.
- The Web files use `web/index.html` plus `/web/*`; backend mounted these after API
  routes and served `/` without creating a separate frontend server.
- Main-thread API/browser smoke confirmed item `2363` at forced position 0,
  offline overriding the valid boost, direct item `404`, omission from all three
  feeds, restore and audit. This runtime evidence, rather than the agent summary,
  supports the final operations claim.

### Frontend and delivery implementation

Prompt summary:

- Build a same-origin, no-build Web UI against the frozen API: login, three Feed
  tabs, linked behavior, profile, real Dashboard, diagnostics and content ops.
- Use a quiet work-focused responsive interface, explicit loading/empty/error/
  auth states and no hardcoded recommendation or metric data.
- Own only `web/**`, README/demo/verification delivery files and the static Web
  contract test. Unrun commands and metrics must remain marked pending.

Agent output:

- Implemented modular HTML/CSS/JavaScript with HttpOnly-session fetches, segmented
  Feed tabs, per-page request linkage, profile/events, real aggregate rendering,
  request/user diagnostics, boost targeting, status changes and audit.
- The frontend never sends a normal-user identity override and never sends client
  impressions. Cover load/error handling converts the backend's 1-pixel placeholder
  into a visible item placeholder.
- Added README execution/evaluation/completeness sections, a timed demo script,
  an append-only verification ledger and seven static contract tests.
- JavaScript syntax checks exited 0. The first targeted pytest attempt failed
  because pytest was not installed; that failure was retained. After dependencies
  became available, the same target passed `7 passed in 0.01s`; a full rerun passed
  `15 passed, 1 warning in 4.71s`.

Lead review:

- The audit proposal originally suggested React/Vite. The lead selected native
  modules to remove a build/runtime dependency while preserving the API surface.
- The lead exercised the UI at 1440 px and 390 px: alice behavior/profile,
  administrator Dashboard, boost/offline/restore and audit completed with zero
  browser-console warnings/errors on those paths.
- The browser run did not cover the complete A/B/carol journey, forced network and
  empty states, or the full recorded demo. Those are not upgraded to complete.

## Cross-Agent Review Record

| Finding | Raised by | Resolution | Runtime evidence |
|---|---|---|---|
| Present-day likes/views could leak future information | Data audit | Excluded from train features, popularity and negative pools | Summary/manifest flags inspected |
| Dashboard lacked required popular content | Frontend review | Added DB-derived `top_items` | Admin browser Dashboard exercised |
| Request debug could not show behavior linkage | Frontend review | Added events to request diagnostics | API tests; response shape reviewed |
| Admin could not know internal user IDs | Frontend review | Username debug plus admin user list and select | Targeted alice boost exercised |
| Strong push could conflict with offline | Lead contract | Final online filter after all merges | Item 2363 offline won in API/browser |
| Backend cover was a valid but blank 1-pixel image | Frontend review | Detect 1x1 load and show item placeholder | Static contract plus zero-console browser run |
| Validation wording implied automatic tuning | Delivery review | Documented manual comparison and fixed rank | README/metrics reviewed |
| Initial pytest invocation lacked pytest | Frontend run | Retained failure, installed dependencies later, reran | 7 Web and 15 repository tests passed |
| Backend closeout misclassified the full artifact as smoke | Backend closeout | Lead compared the pointer/manifest with the data agent's independent hash and shape validation; rejected the claim | 49,416 users, 16,907 items, rank 32; 5,000-user test cohort |
| Architecture still named BPR and implied automatic tuning | Data closeout | Replaced with TruncatedSVD and explicit manual validation comparison | Architecture now matches model code and metrics |
| Event export was phrased as implemented | Data closeout | Reworded as an unimplemented future batch step | Online events still affect synchronous profile/ranking |

## Evidence Boundary and Remaining Work

- Confirmed by main-thread runtime: real event/Dashboard deltas, boost position,
  offline precedence across direct API and all feeds, restore/audit, responsive
  alice/admin browser paths and a clean browser console.
- Confirmed by specialist runs and inspected artifacts/tests: official split,
  metrics, model publication, static Web contract and repository test suite.
- Still pending: full alice/bob/carol browser journey, forced
  network/empty browser states, meaningful commit-count verification and the
  3–5 minute video.
- No source contribution percentage is claimed because no reliable measurement
  was collected. No agent claim is used as evidence for a commit, deployment or
  recording; clean-environment evidence comes from the lead's later rerun.

## Independent Closeout Round

The lead issued three final prompts with explicit file boundaries and completion
evidence. The backend agent was read-only and used a temporary SQLite database;
the data agent was fully read-only; the delivery agent could edit only this log
and `docs/VERIFICATION.md`.

- Backend goal: rerun login/RBAC, alice-bob differentiation, carol cold start,
  pagination, event idempotency/isolation, profile changes, Dashboard and
  boost/offline/restore. Its temporary run passed 4 API tests and a real-artifact
  smoke with 19,220 items, 15 requests, 86 exposures, 89 events and 3 operations.
  The lead accepted those runtime outcomes but rejected its contradictory final
  sentence calling the artifact a 2,000-user smoke model.
- Data goal: recompute split boundaries and hashes, validate every artifact,
  audit leakage rules, evaluation semantics, requirements and ignored-file risk.
  It passed 4 targeted tests without changing the summary, pointer or manifest,
  and identified the stale BPR/tuning/event-export wording corrected above.
- Delivery goal: distinguish agent reports from lead evidence and retain failed
  runs and pending work. Its Markdown structure check passed; the lead then added
  the later dependency, test, initialization, health and mobile-browser reruns.

After all four module commits existed, the lead cloned the history into an ignored
temporary directory, supplied the same official raw files as external input, and
reran dependency sync, smoke processing/training, DB seed, tests, JavaScript checks
and health. This converted clean-checkout reproduction from pending to passed;
the formal demonstration recording remains unperformed.
