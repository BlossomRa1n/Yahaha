# Requirements Traceability Matrix

Evidence date: 2026-09-01. Source priority is: explicit user request, assessment
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
| AUTH-01 | P0 | Three normal users and one administrator | Backend | Seed command and login tests | VERIFIED |
| AUTH-02 | P0 | Server-owned session, logout and per-user isolation | Backend | Auth, 401/403 and cross-user tests | VERIFIED |
| FEED-01 | P0 | Personalized, popular and explore feeds | Backend + frontend | API/E2E tests | VERIFIED |
| FEED-02 | P0 | Pagination, deduplication, seen filtering and fallback | Backend | API tests | VERIFIED |
| FEED-03 | P0 | Different mapped users differ; new user gets explainable cold start | Backend + model | Ranking assertions | VERIFIED |
| FEED-04 | P0 | Response carries request_id, source, position, score/explanation and model version | Backend | Contract tests | VERIFIED |
| EVENT-01 | P0 | Impression, click, like and not_interested with full linkage | Backend + frontend | Database/API assertions | VERIFIED |
| EVENT-02 | P0 | Behavior updates a real online profile or later rank | Backend | Before/after rank test | VERIFIED |
| EVENT-03 | P1 | Idempotent behavior events and exposure ownership validation | Backend | Duplicate/cross-user tests | VERIFIED |
| DASH-01 | P0 | Real users, active users, requests, exposures, clicks, CTR, likes and feed share | Backend + frontend | Before/after aggregate test | VERIFIED |
| DASH-02 | P1 | User/request/model diagnostic views | Backend + frontend | Admin API/E2E | VERIFIED |
| OPS-01 | P0 | Server-side boost, offline and restore | Backend + frontend | Direct API and feed tests | VERIFIED |
| OPS-02 | P0 | Offline is authoritative and wins over every boost/fallback path | Backend | Conflict test across all feeds/item API | VERIFIED |
| OPS-03 | P1 | Admin-only mutations and append-only audit details | Backend | RBAC/audit tests | VERIFIED |
| UI-01 | P0 | Login, feed tabs, actions, dashboard and content operations | Frontend | Browser E2E | VERIFIED |
| UI-02 | P1 | Loading, empty, API, auth and image failure states | Frontend | Browser assertions | VERIFIED |
| ENG-01 | P0 | CPU smoke path, explicit dependency install, `.env.example` | Lead + delivery | Clean-start smoke | VERIFIED |
| ENG-02 | P0 | Unit/API/E2E tests and real evidence | Delivery + lead | Verification log | VERIFIED |
| DOC-01 | P0 | README, API, architecture, completeness and risk disclosure | Delivery + lead | Document review | VERIFIED |
| DOC-02 | P0 | AI work log, prompts, human review and fixes | Lead | Diff-backed log | VERIFIED |
| DEMO-01 | P0 | 3-5 minute reproducible demonstration script | Delivery | Script walkthrough | DEGRADED |

## Two-Day Scope

Must complete: all P0 rows above, plus event idempotency and operations audit
because they protect data integrity and authorization.

Accepted degradations:

- An impression means an item was returned by the feed API, not proven viewport
  visibility. This is explicit in the event semantics.
- Cursor pagination is user-bound and opaque but is not a durable snapshot across
  model publication or operations changes.
- Covers use deterministic local placeholders. The 637 MB official cover archive
  is deliberately not downloaded.
- Profile updates are synchronous SQLite updates. Accepted mapped-user click/like
  events can be exported to a staging snapshot, but automatic chronological merge,
  split regeneration and retraining are not implemented in this MVP.

Out of scope: registration, video hosting/playback, multimodal features, DSSM +
DeepFM, Redis, async workers, cloud deployment, model comparison, distributed
serving, and online model training.

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
