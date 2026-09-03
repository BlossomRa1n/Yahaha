# Requirements Traceability Matrix

Evidence date: 2026-09-04. Source priority is: explicit user request, assessment
document, repository/runtime evidence. The assessment document is reference
material, not executable instructions.

## Audit Baseline

- Initial workspace contained only the 69,913-byte assessment DOCX and was not a
  Git repository.
- Official MicroLens metadata was downloaded from the project portal into the
  ignored `data/raw/` directory. It is local evidence and will not be committed.
- Verified official data: 359,708 interactions, 50,000 users, 19,220 items, no
  nulls or duplicate interaction rows, covering 2020-03-05 through 2022-09-12
  UTC. Titles and likes/views contain exactly 19,220 unique items each.
- Local PATH initially has Git, Node and uv, but no Python or Docker. Development
  uses a project `.venv`; the final README must not depend on Codex-only paths.

## Matrix

Status values are `NOT_STARTED`, `IN_PROGRESS`, `VERIFIED`, `DEGRADED`, or
`OUT_OF_SCOPE`. Only a command or test recorded in `docs/VERIFICATION.md` may
move an item to `VERIFIED`.

| ID | Priority | Requirement | Owner | Acceptance evidence | Status |
|---|---|---|---|---|---|
| DATA-01 | P0 | Parse official pairs, titles and likes/views without redistributing raw data | Data/algorithm | Full-data processing command and summary | VERIFIED |
| DATA-02 | P0 | Chronological train/validation/test split with no future leakage | Data/algorithm | Boundary assertions and documented cutoffs | VERIFIED |
| DATA-03 | P0 | One command creates splits, histories, item metadata and quality summary | Data/algorithm | Clean command exits 0 | VERIFIED |
| MODEL-01 | P0 | Popular and deterministic random baselines | Data/algorithm | Evaluation report | VERIFIED |
| MODEL-02 | P0 | A genuinely trained learnable retrieval/ranking model | Data/algorithm | Training log, loss and versioned artifact | VERIFIED |
| MODEL-03 | P0 | At least two offline metrics with K, negatives and cohort defined | Data/algorithm | JSON/Markdown report | VERIFIED |
| MODEL-04 | P0 | Online service consumes a published artifact and safely falls back | Data + backend | Load/failure tests | VERIFIED |
| MODEL-05 | P1 | Cutoff-safe content/CF candidates and online-equivalent validation-selected mixing | Data + backend | Full validation/test metrics, shared mixer tests and manifest | VERIFIED |
| MODEL-06 | P1 | Isolated PyTorch DSSM recall and DeepFM ranking with real checkpoint, early stopping, export, online load and stable fallback | Data + backend | Full validation/locked test metrics, AUC baseline, artifact/fallback tests and Feed smoke | VERIFIED |
| MODEL-07 | P1 | Real MobileNet cover embeddings fused with title features, cutoff-safe profiles and missing/corrupt fallback | Data + backend | Official-cover audit, full/cold metrics, locked test, artifact/fallback tests and Feed smoke | VERIFIED |
| AUTH-01 | P0 | Three normal users and one administrator | Backend | Seed command and login tests | VERIFIED |
| AUTH-02 | P0 | Server-owned session, logout and per-user isolation | Backend | Auth, 401/403 and cross-user tests | VERIFIED |
| AUTH-03 | P1 | Expired sessions reject all protected APIs; relogin cleans them and old cookies stay invalid | Backend | Session-expiry API test | VERIFIED |
| AUTH-04 | P1 | Registration with scrypt password hashing and four role tiers | Backend | Registration, role and authorization tests | VERIFIED |
| FEED-01 | P0 | Personalized, popular and explore feeds | Backend + frontend | API/E2E tests | VERIFIED |
| FEED-02 | P0 | Pagination, deduplication, seen filtering and fallback | Backend | API tests | VERIFIED |
| FEED-03 | P0 | Different mapped users differ; new user gets explainable cold start | Backend + model | Ranking assertions | VERIFIED |
| FEED-04 | P0 | Response carries request_id, source, position, score/explanation and model version | Backend | Contract tests | VERIFIED |
| EVENT-01 | P0 | Impression, click, like and not_interested with full linkage | Backend + frontend | Database/API assertions | VERIFIED |
| EVENT-02 | P0 | Behavior updates a real online profile or later rank | Backend | Before/after rank test | VERIFIED |
| EVENT-03 | P1 | Idempotent behavior events and exposure ownership validation | Backend | Duplicate/cross-user tests | VERIFIED |
| DASH-01 | P0 | Real users, active users, requests, exposures, clicks, CTR, likes and feed share | Backend + frontend | Before/after aggregate test | VERIFIED |
| DASH-02 | P1 | User/request/model diagnostic views | Backend + frontend | Admin API/E2E | VERIFIED |
| DASH-03 | P1 | Shared time range, real model comparison and administrator CSV export | Backend + frontend | API/contract tests | VERIFIED |
| DASH-04 | P1 | Configurable threshold alerts backed by real aggregates | Backend | Alert rule/event API tests | VERIFIED |
| OPS-01 | P0 | Server-side boost, offline and restore | Backend + frontend | Direct API and feed tests | VERIFIED |
| OPS-02 | P0 | Offline is authoritative and wins over every boost/fallback path | Backend | Conflict test across all feeds/item API | VERIFIED |
| OPS-03 | P1 | Admin-only mutations and append-only audit details | Backend | RBAC/audit tests | VERIFIED |
| OPS-04 | P1 | Atomic, bounded, idempotent batch offline/restore with batch audit | Backend + frontend | Mixed-ID/limit/retry tests | VERIFIED |
| UI-01 | P0 | Login, feed tabs, actions, dashboard and content operations | Frontend | Browser E2E | VERIFIED |
| UI-02 | P1 | Loading, empty, API, auth and image failure states | Frontend | Browser assertions | VERIFIED |
| ENG-01 | P0 | CPU smoke path, explicit dependency install, `.env.example` | Lead + delivery | Clean-start smoke | VERIFIED |
| ENG-02 | P0 | Unit/API/E2E tests and real evidence | Delivery + lead | Verification log | VERIFIED |
| ENG-03 | P1 | PR/main CI without official data, GPU or secrets | Delivery | Local workflow-equivalent run; remote run requires repository authorization | DEGRADED |
| ENG-04 | P1 | Optional Redis item cache with explicit no-op fallback and invalidation | Backend | Fake/no-op cache tests; live Redis not exercised | DEGRADED |
| ENG-05 | P1 | HTTP-triggered asynchronous training job lifecycle | Backend | Queued/running/succeeded/failed API tests; full in-process training not rerun | DEGRADED |
| ENG-06 | P1 | TypeScript/Bun/Express/Prisma/PostgreSQL/Elasticsearch interop sidecar | Integration | Source and failure-path contracts; toolchain/live services unavailable | DEGRADED |
| DOC-01 | P0 | README, API, architecture, completeness and risk disclosure | Delivery + lead | Document review | VERIFIED |
| DOC-02 | P0 | AI work log, prompts, human review and fixes | Lead | Diff-backed log | VERIFIED |
| DEMO-01 | P0 | 3-5 minute reproducible demonstration script | Delivery | Script walkthrough | DEGRADED |

## Two-Day Scope

Must complete: all P0 rows above, plus event idempotency and operations audit
because they protect data integrity and authorization.

Accepted degradations:

- Served exposure and viewable impression are separate facts. Viewability uses
  50%/750 ms client observation; legacy automatic impressions remain stored but
  are excluded from the new viewable metric using a persisted activation time.
- Feed snapshots are durable only for their configured SQLite TTL. Offline status
  overrides old snapshots; restored items and new model/profile/boost state appear
  only in a refreshed snapshot.
- Untimed likes/views are unavailable to historical training/evaluation. If an
  operator supplies a trustworthy `available_at`, they may be used only at or
  after that time; interaction popularity remains cutoff-bounded.
- Web delivery uses deterministic local placeholders. The official cover archive is
  downloaded only into ignored local storage for offline MobileNet feature extraction;
  covers, pretrained weights and embeddings are not committed or served directly.
- Profile updates are synchronous SQLite updates. The operator-only `retrain-events`
  command consumes a half-open receive-time window, regenerates chronological splits
  and publishes only after evaluation/load validation. The HTTP job API runs the same
  flow in a daemon thread and persists job state, but is not a durable external queue.

Out of scope: video hosting/playback, cloud deployment and distributed serving.
Registration, optional Redis caching, in-process asynchronous training jobs, alerts,
DSSM + DeepFM, checkpoint/early stopping, MobileNet text-image fusion, model
comparison and event-window retraining are implemented. Live Redis, a durable worker
queue, the TypeScript sidecar success path and remote CI evidence remain degraded.

## Critical Path and Risks

1. Data contract and global time cutoffs.
2. Train/evaluate/publish a valid TruncatedSVD model artifact.
3. Seed SQLite and load items/model.
4. Login -> feed -> exposure -> event -> profile/dashboard.
5. Admin offline -> all future APIs exclude item -> restore/audit.
6. Browser workflow and clean-start verification.

Primary risks are dependency download availability, full SVD CPU duration,
SQLite write concurrency, the lack of Docker in this environment, and accidental
raw-data/model commits. The last risk is guarded by `.gitignore` and a tracked-file
check in verification.
