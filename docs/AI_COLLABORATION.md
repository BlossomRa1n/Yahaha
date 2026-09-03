# AI Collaboration and Review Log

This log records delegated work as evidence, not as proof of completion. The lead
agent marks implementation complete only after inspecting diffs and rerunning
tests.

## AI Tool Used

This project used **OpenAI Codex** for repository inspection, implementation,
code review assistance, test execution, debugging and delivery-document updates.
The prompt summaries below record the key instructions given to Codex and its
specialist agents. The candidate retained responsibility for architectural choices,
reviewed the resulting diffs, rejected unsupported claims and accepted changes only
after relevant tests or runtime checks passed.

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

Agent output（历史实现；下述 server impression/cursor 语义已被 2026-09-02
viewable-impression 与持久快照升级取代）:

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
- At that historical checkpoint, online events were not integrated into a new
  offline run. This statement is superseded by the synchronous, windowed
  `retrain-events` implementation and its cutoff/idempotency tests.

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
at that checkpoint the formal demonstration recording had not yet been performed.
It was subsequently recorded and its external link was added to `README.md`,
`docs/DEMO.md`, `docs/DELIVERY_CHECKLIST.md` and `docs/VERIFICATION.md`.

## 2026-09-02 Mandatory Closure Round

The lead assigned three non-overlapping specialist tasks and kept integration ownership:

| Role | Goal and allowed files | Required evidence | Lead review outcome |
|---|---|---|---|
| Dashboard gap | Add per-feed request/exposure/click/like/not-interested/CTR/share aggregation and render it; `app/main.py`, `tests/test_api.py`, `web/app.js`, `web/index.html` only | Targeted API/Web tests and JS syntax | Accepted after reading the SQL/UI diff and rerunning the combined suite; behavior attribution uses the exact exposure tuple |
| Model delivery | Add deterministic anonymized SVD Badcases and make event export semantically honest; `recsys/model.py`, `app/cli.py`, related model/export tests only | Targeted/full pytest, smoke training, artifact inspection | Accepted after a new full official train and artifact inspection; rejected the old duplicate-row weighting claim |
| Observability tests | Test latency quantiles, Dashboard time series and JSON request logs; `tests/test_observability.py` only | Focused tests with exact boundary cases | Tests exposed a trailing zero bucket at an exact `[from,to)` boundary; lead fixed production code and reran the suite |

The lead then reviewed the integrated production diff and executed full pytest, JavaScript syntax,
Python compilation, official-data smoke training, the official E2E script and browser workflows.
One additional UI contract mismatch was found during parent review: the model table error row retained
the old five-column span after the data-version column was added. It was corrected to six and verified
with the final Web checks. Agent summaries were not used as completion evidence without these reruns.

## 2026-09-02 Strategy-Consistency and Session Closeout

- The lead re-read the assessment and the historical model-optimization plan, then
  compared the actual online fixed quota mixer with offline `hybrid_all_sources`.
  The two paths were demonstrably different, so prior five-source metrics were not
  accepted as an online-policy replay.
- A shared pure `mix_candidates` contract was introduced for both training and Feed.
  Candidates now carry eligibility, confidence and support; non-finite candidates and
  zero/unsupported CF rows are removed before rank normalization. Validation alone
  compares the bounded safe/dynamic policies, writes the selected policy to the
  manifest, and test evaluates only that lock.
- Full-data training with sampled-all-items validation rejected `dynamic_confidence_v2`:
  sampled-all-items Recall/NDCG
  `0.283030/0.147777` versus safe `0.382142/0.181380`. The published artifact therefore
  uses `safe_svd_content_v2`; test sampled-all-items Recall/NDCG/HitRate remains
  `0.367097/0.150900/0.479800` with cold coverage `0.998687`.
  The final rerun with Top-10 quota/Jaccard diagnostics published
  `svd-20260902T143346986486Z-ab4c3e04` without changing those metrics.
- Session expiry review found no production defect. A new API test sets the stored
  expiry in the past, proves 401 from auth/feed/admin routes, proves login cleanup and
  old-cookie rejection, and rechecks ordinary/admin isolation.
- CI evidence remains local only. No commit, push, PR or other remote mutation was
  performed in this round.

## 2026-09-02 DSSM/DeepFM and Multimodal Experiment Round

- The implementation kept PyTorch as the single deep-learning framework. DSSM and
  DeepFM use the existing chronological split, cutoff-safe title/profile/popularity
  features, deterministic mixed negatives and independent experiment pointers. Each
  epoch wrote safetensors weights plus optimizer/training state; patience-based early
  stopping selected DSSM epoch 3 and DeepFM epoch 1.
- Validation compared independent DSSM, DSSM+DeepFM, rank-fusion and protected
  reranking candidates. The locked `protected_top10_rerank` policy retained stable
  Top-10 membership while improving order. Test was run only after validation lock;
  it was not used to return to tuning.
- One attempted comparison omitted `--max-eval-users 5000`. The query-set hash gate
  rejected it before expensive training, and its metrics were not compared. Repeated
  non-resume training also exposed a Windows checkpoint-directory collision; unique
  checkpoint directories fixed it without weakening checkpoint validation.
- Multimodal work used the official local cover archive and real pretrained
  MobileNetV3-Small weights. All 19,220 items mapped to readable images; 11 duplicate
  images were reported. PCA-128 was fitted only on 16,907 train-visible items, and
  user visual profiles used only cutoff-safe positive history.
- Validation compared text-only, visual-only and fusion on the same query hashes.
  Static visual weight `0.20` passed the Full and cold gates. Two later fine-search
  rounds failed the specified 1% Full / 3% cold improvement threshold, so the loop
  stopped and retained `0.20` rather than using test results to keep tuning.
- Both artifacts remain isolated experiments. Online loading, combined inference and
  failure fallback were verified, but `artifacts/current.json` still points to
  `safe_svd_content_v2`. No commit, push, branch, PR, production publication or remote
  CI trigger was performed.
