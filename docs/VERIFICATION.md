# Verification Log

这里是完成状态的唯一运行证据账本。代码存在、代理总结或 README 命令不等于验证。
执行者必须记录日期、完整命令、退出码、关键输出和失败/降级；长输出可引用仓库内不含
敏感数据的日志路径。禁止补写未实际运行的成功结果。

状态定义：`PASSED` 表示表中所述范围有实际证据；`PARTIAL` 表示只覆盖了部分验收面；
`PENDING` 表示尚无足够证据。代理运行结果与主代理复核必须明确区分，不能用代理总结
替代主代理的浏览器、API 或干净环境验收。

## Current Production Model (2026-09-04)

- 当前个性化推荐的稳定架构是统一两阶段模型，不是若干互相独立的实验：
  SVD、DSSM、标题内容、视觉、item-item CF、热门和探索共同召回，候选合并去重后
  统一由 DeepFM 排序。DSSM 是正式召回源，DeepFM 是正式排序器。
- `artifacts/current.json` 指向 SVD 基础产物
  `svd-20260902T143346986486Z-ab4c3e04`，用于统一模型中的 SVD 召回和故障回退，
  不代表完整线上模型。
- 服务默认通过 `artifacts/experiment-current.json` 加载当前深度模型
  `deep-20260903T152647940394Z-cb975d73`；文件名中的 `experiment` 是历史命名，
  不表示 DSSM/DeepFM 仍处于独立实验链路。
- `artifacts/multimodal-current.json` 指向
  `multimodal-20260902T180847621178Z-7e09b190`，作为同一统一模型的视觉召回组件。
  深度或视觉产物缺失、损坏或版本不兼容时，已映射 warm 用户回退到个性化 SVD；
  cold 用户或基础模型也不可用时回退热门/探索。

## Environment

| Field | Observed value | Status |
|---|---|---|
| OS | Windows 10.0.26200 / PowerShell 7.6.4 | OBSERVED |
| System Node | 24.20.0 | OBSERVED |
| System Git | 2.53.0.windows.3 | OBSERVED |
| System Python | Not on initial PATH | OBSERVED |
| Docker | Not installed on initial PATH | OBSERVED |
| Project Python / uv | Python 3.12.13 / uv 0.11.21 | OBSERVED |

## Command Evidence

| ID | Requirement | Command / action | Result | Evidence source | Status |
|---|---|---|---|---|---|
| V-01 | Dependency install | `uv sync --cache-dir .uv-cache --group dev` | Resolved 33 packages; checked 30 packages; exit 0 | Main-thread rerun | PASSED |
| V-02 | Official data preparation | `.venv\\Scripts\\python.exe -m recsys.pipeline all --raw-dir data/raw --out-dir data/processed --artifacts-dir artifacts --mode full --rank 32 --seed 20260901` | Exit 0; prepare 1.755s; deterministic summary and cutoffs produced | Data agent run; artifact/summary inspected in delivery round | PASSED |
| V-03 | Full CPU model | Same `all --mode full` command | Exit 0 in 19.323s; baselines, SVD metrics and atomic pointer present | Data agent run; current manifest/metrics independently parsed | PASSED |
| V-04 | Database/seed | `.venv\\Scripts\\python.exe -m app.cli init-db --items data/processed/items.csv --reset` | Exit 0; 19,220 items; alice/bob/carol/admin; full model loaded | Main-thread rerun | PASSED |
| V-05 | Service health | Start uvicorn on 8010 and query `/api/v1/health` | `status=ok`, `database=ok`, full model version, `model_error=null` | Main-thread HTTP check | PASSED |
| V-06 | Unit/API tests | `.venv\\Scripts\\python.exe -m pytest -p no:cacheprovider` | Latest rerun: `15 passed, 1 warning in 6.07s` | Main-thread rerun after dotenv/document fixes | PASSED |
| V-07 | Web static contract | `.venv\\Scripts\\python.exe -m pytest tests\\test_web_contract.py -p no:cacheprovider` | `7 passed in 0.01s` | Frontend/delivery agent | PASSED |
| V-07A | Web stdlib assertions | `.venv\\Scripts\\python.exe -B -c <direct test runner>` | 7 assertions passed before pytest became available | Frontend/delivery agent | PASSED |
| V-08 | JavaScript syntax | `node --check web/api.js`; `node --check web/app.js` | Latest main-thread rerun: both exited 0 | Main thread | PASSED |
| V-09 | Browser workflow | Real browser at 1440 px and 390 px | Alice behavior/profile and admin Dashboard/operations passed; final clean-state mobile run had 12 cards, no horizontal overflow and zero console warnings/errors; A/B, carol and full demo script were not all repeated in browser | Main-thread browser smoke | PARTIAL |
| V-10 | Offline authority | Real API plus browser boost/offline/restore checks | Item 2363 forced at position 0; offline overrode boost, direct item returned 404 and all three feeds omitted it; restore and audit succeeded | Main-thread API/browser smoke | PASSED |
| V-11 | Secret/raw audit | Inspect tracked files and all committed paths/history | No raw/processed data, artifact, DB, `.env`, DOCX or large model file is tracked; meaningful module commits exist | Main-thread Git audit | PASSED |
| V-12 | Clean environment | Clone committed history, add user-supplied official raw input, repeat README smoke | Locked sync, pipeline, seed, 15 tests, JS and health all exited 0 | Main-thread clean clone | PASSED |
| V-13 | Live event/Dashboard smoke | Feed then click/like against running service | `+1` request, `+5` exposures, `+1` click and `+1` like observed in real aggregates | Main-thread API smoke | PASSED |
| V-14 | Mandatory official-data E2E | `.venv\\Scripts\\python.exe -m scripts.verify_official_e2e` | Exit 0; 19,220 items; alice/bob differ; carol cold start; stable snapshot replay; Dashboard `+6 requests/+45 served/+3 viewable/+1 click/+1 like`; boost/offline/restore/audit/logout passed | Main-thread rerun, 2026-09-02 | PASSED |
| V-15 | Expanded regression | `.venv\\Scripts\\python.exe -m pytest -p no:cacheprovider --basetemp .tmp\\pytest-final-rerun` | Final rerun: `31 passed, 1 warning in 13.79s`; only third-party Starlette/httpx deprecation warning | Main-thread rerun, 2026-09-02 | PASSED |
| V-16 | Fresh smoke output | `.venv\\Scripts\\python.exe -m recsys.pipeline all --raw-dir data/raw --out-dir .tmp/closeout-smoke-final2/processed --artifacts-dir .tmp/closeout-smoke-final2/artifacts --mode smoke --rank 8 --seed 20260902` | Exit 0; 359,708 interactions; chronological `287,767/35,971/35,970` split; 19,220 items; artifact published with evaluation and anonymized Badcases | Main-thread rerun, 2026-09-02 | PASSED |
| V-17 | Browser network recovery | Stop service, refresh Dashboard, restart same service, click retry | Visible connection error and retry control; retry restored real Dashboard; 390 px viewport reset after mobile verification | Main-thread in-app browser, 2026-09-02 | PASSED |
| V-18 | Current-source clean reproduction | Copy only deliverable source to ignored clean directory; offline locked sync; smoke pipeline; DB seed; pytest/JS/compile; start Uvicorn on 8011 and query health | 30 locked packages installed; 359,708 interactions/19,220 items; `26 passed`; health `status=ok`, `database=ok`, smoke model loaded without error | Main-thread rerun, 2026-09-02 | PASSED |
| V-19 | Shared policy sampled-all-items evaluation | `.venv\Scripts\python.exe -m recsys.pipeline train --processed-dir data/processed --artifacts-dir artifacts --mode full --rank 32 --seed 20260901` | Final rerun exit 0; validation rejected dynamic sampled R/N `0.283030/0.147777` against safe `0.382142/0.181380`; published `svd-20260902T143346986486Z-ab4c3e04` with `safe_svd_content_v2`; Top-10 quota 7 model/3 cold content, zero relaxation/duplicates; test sampled R/N/H `0.367097/0.150900/0.479800`, cold coverage `0.998687` | Main-thread run, 2026-09-02 | PASSED |
| V-20 | Mixer/Feed/operations regression | `.venv\Scripts\python.exe -m pytest tests/test_model.py tests/test_data.py tests/test_content_recall.py tests/test_hybrid_recall.py tests/test_popularity.py tests/test_diversity.py tests/test_api.py tests/test_engagement_events.py -q --basetemp .test-tmp/mix-regression -p no:cacheprovider` | `33 passed`; only third-party Starlette/httpx warning | Main-thread run, 2026-09-02 | PASSED |
| V-21 | Session expiry | `.venv\Scripts\python.exe -m pytest tests/test_api.py::test_expired_session_is_rejected_cleaned_on_relogin_and_cannot_be_reused tests/test_api.py::test_auth_logout_admin_forbidden_and_sqlite_pragmas -q --basetemp .test-tmp/session-expiry -p no:cacheprovider` | `2 passed`; expired auth/feed/admin all 401, relogin cleanup, stale Cookie denial and role isolation covered | Main-thread run, 2026-09-02 | PASSED |
| V-22 | Final repository regression | `.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp/final-after-latency -p no:cacheprovider` | `56 passed`; only third-party Starlette/httpx warning | Main-thread run, 2026-09-02 | PASSED |
| V-23 | Final official E2E | `.venv\Scripts\python.exe -m scripts.verify_official_e2e --items data/processed/items.csv --model-pointer artifacts/current.json` | First two reruns failed on obsolete hard requirements for `item_cf` exposure; contract was aligned to the validation-locked policy. Final two reruns exited 0: 19,220 items, safe policy, source counts 14 model/2 content, A/B/cold start/snapshot/events/Dashboard/ops/logout passed. Avoiding unused safe-policy sources reduced same-script personalized mean/max from 522.67/861.60 ms to 361.24/567.00 ms. | Main-thread reruns, 2026-09-02 | PASSED after retained failures |
| V-24 | Final local CI equivalent | `.venv\Scripts\python.exe -m pytest tests/test_web_contract.py tests/test_ci_contract.py ...`; `compileall`; both `node --check`; `.venv\Scripts\python.exe -m scripts.ci_smoke` | Contract `9 passed`; compile and JS exits 0; smoke status passed with 1,500 interactions, 150 items, 3 events, DB/model OK and `safe_svd_content_v2` | Main-thread run, 2026-09-02 | PASSED locally; remote unverified |
| V-25 | DSSM/DeepFM training and optimization | `.venv\Scripts\python.exe -m recsys.pipeline train-deep --mode full --max-eval-users 5000 --epochs 8 --patience 2` plus validation-only protected-rerank search | Real PyTorch DSSM/DeepFM trained on 287,767 interactions; safetensors checkpoints restored/exported; DSSM early-stopped at epoch 5 (best 3), DeepFM at epoch 3 (best 1); locked `deep-20260902T175056309503Z-fff15189` | Full artifact, training history, query hashes and contract tests inspected, 2026-09-02 | PASSED |
| V-26 | Official-cover multimodal component | `.venv\Scripts\python.exe -m recsys.pipeline train-multimodal --batch-size 64 --pca-dim 128 --max-eval-users 5000 --locked-visual-weight 0.20` | 19,220/19,220 covers mapped and parsed, 11 duplicate images, real pretrained MobileNetV3-Small, train-visible-only PCA-128, locked `multimodal-20260902T180847621178Z-7e09b190`; later tuning comparisons are retained in the metrics | Extraction/manifest/query hashes and contract tests inspected, 2026-09-02 | PASSED |
| V-27 | Combined serving validation | Temporary official-data SQLite HTTP smoke plus 25-user direct inference timing | Source `dssm_deepfm_multimodal`; direct P50/P95/P99 75.82/83.40/84.78 ms; first HTTP Feed 422/446 ms, cursor pages about 9 ms; users differed, no duplicates/fallback, MobileNet evidence present | Main-thread smoke, 2026-09-02 | PASSED |
| V-28 | Deep/multimodal contracts and fallback | `.venv\Scripts\python.exe -m pytest tests/test_deep_models.py tests/test_multimodal.py ...` and combined hybrid/deep/multimodal targets | Deep/multimodal contracts `8 passed`; combined affected targets `14 passed`; corruption, incompatible metadata, checkpoint/restore and stable fallback covered | Main-thread targeted runs, 2026-09-02 | PASSED |
| V-29 | Frozen full regression | `.venv\Scripts\python.exe -m pytest --basetemp .test-tmp/final-freeze-summary-20260903 -p no:cacheprovider` | `64 passed, 1 warning in 27.93s`; warning is the known third-party Starlette/httpx deprecation | Main-thread rerun, 2026-09-03 | PASSED |
| V-30 | Frozen compile/Web/local CI | `.venv\Scripts\python.exe -m compileall -q app recsys scripts tests`; `node --check web/app.js`; `node --check web/api.js`; `.venv\Scripts\python.exe -m scripts.ci_smoke` | All exited 0; CI smoke processed 1,500 interactions/150 items, accepted 3 events and loaded `safe_svd_content_v2` | Main-thread rerun, 2026-09-03 | PASSED locally; remote unverified |
| V-31 | Frozen official-data E2E | `.venv\Scripts\python.exe -m scripts.verify_official_e2e --items data/processed/items.csv --model-pointer artifacts/current.json` | First freeze run reached operations but intermittently observed zero Dashboard revisits and failed; identical rerun exited 0 with 19,220 items, A/B, cold start, stable cursor, dwell/share/revisit, Dashboard `+6/+45/+3/+1/+1/+1`, boost/offline/restore/audit and logout. Failure retained as a flaky E2E observation; focused revisit tests remained green. | Main-thread reruns, 2026-09-03 | PASSED after retained transient failure |
| V-32 | Freeze hygiene and remote boundary | `git diff --check`; `git remote -v`; `git ls-files`; inspect SVD/deep/multimodal pointers | Diff check exit 0 (line-ending notices only); origin is `https://github.com/BlossomRa1n/Yahaha.git`; no tracked raw covers/data, checkpoints, weights, DB or secrets; SVD base and unified deep/multimodal pointers reference loadable versions | Main-thread inspection, 2026-09-03 | PASSED locally; no remote write/run |
| V-33 | Unified seven-source two-stage production model | `uv run python -m recsys.pipeline train-deep --mode full --max-eval-users 5000 --epochs 8 --patience 2`, then one locked test run from checkpoint | Real 287,767-interaction DSSM/DeepFM training; SVD/DSSM/content/visual/item-CF/popular/explore union; 17 continuous + 5 categorical fields; 186,695 deterministic cold-start dropout rows; DSSM best epoch 3, DeepFM best epoch 1, both early-stopped; artifact `deep-20260903T045748115694Z-a8df9062` | Sampled-all-items manifest/metrics/checkpoint reload, real-user Feed and mismatch fallback smoke, 2026-09-03 | PASSED; adopted as the sole personalized production path |
| V-34 | Scheme B regression and hygiene | `uv run pytest -q`; `compileall`; `node --check`; `uv run python scripts/ci_smoke.py`; `git ls-files` | `71 passed`; compile/JS exit 0; CI smoke passed; final real-user union 657 and Top-10 included DSSM/visual/content/item-CF primary sources; incompatible multimodal version fell back to stable SVD; no tracked model/data/cover/DB files | Main-thread run, 2026-09-03 | PASSED locally |
| V-35 | Unified sampled-negative protocol | Full pytest, compileall, `scripts.ci_smoke`, diff check, then GPU `train-deep` smoke with validation and locked test | 80 tests passed; SVD/deep/visual use `deterministic_sampled_negatives_v1`; sampled metrics retain all positives and use 100 deterministic negatives; CUDA smoke used RTX 5070 and finished in 35.1 s; artifact `deep-20260903T134019966433Z-440b2b95`; the mismatched 500-user cohort was correctly identified in metadata | Main-thread run, 2026-09-03 | PASSED locally |
| V-36 | Current unified production pointer | Inspect `app/config.py`, `app/main.py`, `app/recommendation.py` and the three artifact pointers | Default personalized requests load `deep-20260903T152647940394Z-cb975d73` and execute seven-source recall plus DeepFM; SVD is the base recall/fallback artifact and the pinned multimodal version supplies visual candidates | Main-thread code, pointer and manifest inspection, 2026-09-04 | PASSED; ACTIVE PRODUCTION PATH |
| V-37 | Multi-user personalization rejection-risk regression | `uv run python -m scripts.verify_official_e2e --items data/processed/items.csv --model-pointer artifacts/current.json`; `uv run python -m pytest -q -p no:cacheprovider --basetemp=.tmp/pytest-34-fix-final` | Official E2E exited 0 with `alice_bob_different=true`; unified source counts included visual/content/DSSM/item-CF; cold start, replay, events, Dashboard, boost/offline/restore, RBAC and logout passed. Full suite: 102 tests passed; only the known third-party Starlette/httpx warning | Main-thread rerun, 2026-09-04 | PASSED |
| V-38 | Remote GitHub CI | GitHub Actions `CI` on `main` commit `970d23a` | Run `33817297423` completed successfully in 1m04s; locked dependency install, 102 Python tests, compile check, JavaScript checks and synthetic data/model/migration/health smoke all passed | [GitHub Actions](https://github.com/BlossomRa1n/Yahaha/actions/runs/33817297423), 2026-09-04 | PASSED |

## Offline Metrics

Do not transcribe values until V-03 succeeds. Record the artifact version, data version, cohort sizes,
seed, negative count and the popular/random/SVD Recall@10, NDCG@10 and HitRate@10 values from the
generated `metrics.json`.

| Artifact version | Data version | Mode | Cohort | Metrics | Evidence | Status |
|---|---|---|---|---|---|---|
| `svd-20260901T121430026505Z-cccf5c24` | `microlens50k-cb7fb01dc9f42b6b` | full | test: 5,000 users, 100 negatives, coverage 0.237160 | Popular R/N/H 0.246499/0.123706/0.281800; random 0.089834/0.043384/0.108800; SVD 0.359176/0.208986/0.399000 | `manifest.json`, `metrics.json` inspected | PASSED |
| `svd-20260902T022510026424Z-cccf5c24` | `microlens50k-cb7fb01dc9f42b6b` | full | test: 5,000 users, 100 negatives, coverage 0.237160 | Popular R/N/H 0.246499/0.123706/0.281800; random 0.089834/0.043384/0.108800; SVD 0.359176/0.208986/0.399000 | Current manifest/metrics and generated Badcases inspected; official E2E consumed this version | PASSED |
| `svd-20260902T143346986486Z-ab4c3e04` | `microlens50k-cb7fb01dc9f42b6b` | full-data SVD base | test: 5,000 users, 100 negatives, warm coverage 0.237160; sampled cold coverage 0.998687 | SVD/content policy warm R/N/H 0.359176/0.208986/0.399000; sampled-all-items 0.367097/0.150900/0.479800; dynamic comparison R/N 0.283030/0.147777 vs safe comparison 0.382142/0.181380 | Training output and manifest/metrics inspected | PASSED; base recall and fallback component |
| `deep-20260902T175056309503Z-fff15189` | `microlens50k-cb7fb01dc9f42b6b` | full DSSM/DeepFM predecessor | test: same locked 5,000-user Full/Warm query hashes; candidate/cold coverage 1.0 | Protected rerank Full R/N/H 0.367097/0.234943/0.479800 vs SVD-base policy 0.367097/0.150900/0.479800; Warm 0.359176/0.216400/0.399000 vs 0.359176/0.208986/0.399000; validation DeepFM AUC 0.727496 vs linear 0.642042 | Manifest, metrics, training history and reload inspected | PASSED |
| `multimodal-20260902T180847621178Z-7e09b190` | `microlens50k-cb7fb01dc9f42b6b` | full visual component | same locked 5,000-user query hashes; visual/cold-item coverage 1.0 | Validation text R/N 0.291515/0.210490, visual 0.253712/0.149744, fusion 0.310377/0.213680; cold fusion R/N 0.331548/0.221732; final test fusion R/N/H 0.276839/0.185855/0.358200 | Manifest, extraction report, metrics and fallback tests inspected | PASSED |
| `deep-20260903T045748115694Z-a8df9062` | `microlens50k-cb7fb01dc9f42b6b` | unified seven-source full model | same locked 5,000-user sampled-all-items/Warm query hashes; candidate/cold coverage 1.0 | Validation unified R/N/H 0.319631/0.205171/0.413400 vs SVD-base policy 0.382142/0.181380/0.486200; Warm 0.362642/0.227026/0.409600 vs 0.359512/0.218799/0.409600; AUC 0.708714 vs linear 0.687647. Locked test sampled R/N/H 0.257060/0.159583/0.342000 | Manifest, metrics, checkpoint reload, real-user online and failure fallback smoke | PASSED |
| `deep-20260903T152647940394Z-cb975d73` | `microlens50k-cb7fb01dc9f42b6b` | current unified seven-source CUDA model | validation: 1,000 users, 100 negatives; candidate/cold coverage 1.0 | Validation unified R/N/H 0.243420/0.148077/0.337000; DSSM 0.257040/0.153110/0.328000; DeepFM AUC 0.667819 vs linear 0.696504 | Current pointer, manifest, metrics and default service loading path inspected | ACTIVE PRODUCTION MODEL |

## End-to-End Acceptance

| Flow | Evidence to record | Result | Status |
|---|---|---|---|
| Login/session/isolation | 3 normal users, admin, refresh, expiry, logout, 401/403 | Automated auth/logout/expiry/old-cookie/403/isolation tests passed; browser login covered alice and admin, not all 3 normal users | PASSED (automated), PARTIAL (browser) |
| Three feeds | request_id, provenance, pagination, A/B difference, cold start | Official-data E2E verified Alice/Bob receive different personalized lists, Carol cold start, stable pagination replay, provenance, and offline removal from all three feeds. Full browser A/B, pagination and Carol sequence remains | PASSED (automated), PARTIAL (browser) |
| Event/profile | DB event linkage and before/after profile/rank | Main API smoke produced +1 request/+5 exposures/+1 click/+1 like; browser showed alice behavior and updated profile | PASSED |
| Dashboard | Numeric baseline and post-action delta | Main thread observed real post-action deltas and admin Dashboard; top items came from DB aggregation | PASSED |
| Boost/offline/restore | Target feed, all-path filter, audit before/after | Item 2363 appeared as forced position 0; offline won over boost, direct API returned 404, three feeds omitted it; restore and audit passed | PASSED |
| Error/fallback | Missing model, empty candidates, offline item, network error | Automated corrupt-model and empty-candidate fallback passed; live offline 404 passed; browser showed connection error and recovered through retry after service restart | PASSED |
| Responsive/console | Desktop and mobile layout, browser console | 1440 px and 390 px workflows completed with zero console warnings/errors | PASSED |
| Clean reproduction | Fresh checkout through smoke model, DB, tests and health | Completed from the four committed revisions | PASSED |
| Demo video | 3–5 minute recording | [江雨鸿-视频演示.mp4（夸克网盘）](https://pan.quark.cn/s/81e4e62f191a)；分享落地页于 2026-09-04 返回 HTTP 200 | DELIVERED |

## Append-Only Run Notes

Add dated entries below. Include failures and subsequent fixes; never replace an earlier failed run.

### 2026-09-02 — Shared mixing and session closeout

- The original offline `hybrid_all_sources` and online fixed buckets were not the
  same algorithm. A shared pure mixer now drives both. Sampled-all-items validation rejected
  the dynamic policy on Recall/NDCG and locked the safe policy; test remained
  equal to the prior SVD/content fallback instead of publishing the degraded five-source list.
- The official E2E initially failed at two separate assertions that demanded
  `item_cf` in served/dashboard sources. Those assertions encoded source count as
  quality and contradicted the locked safe policy. Both failures were retained;
  the script now validates the response policy version, source compatibility and
  zero duplicates. The identical command then passed twice, including after the
  safe-path latency optimization.
- Session expiry tests set the stored expiry in the past and proved 401 on auth,
  Feed and admin routes, cleanup on relogin, stale-cookie rejection and role isolation.
- At this historical checkpoint remote CI was unverified and no Git remote was
  configured. The final freeze later configured the correct `origin` (V-32), but no
  credential was exercised and no commit, push, PR, remote run or publication occurred.

### 2026-09-01 — Static Web implementation

- `node --check web/api.js` exited `0` with no syntax errors.
- `node --check web/app.js` exited `0` with no syntax errors.
- The first `.venv\\Scripts\\python.exe -m pytest tests\\test_web_contract.py -p no:cacheprovider`
  attempt exited `1`: `No module named pytest`. This remains recorded rather than being hidden.
- A direct Python 3 stdlib runner imported `tests/test_web_contract.py`, executed every `test_*`
  function, and exited `0` with `7 static web contract tests passed`.
- After the project dependencies became available, the targeted pytest command was rerun and exited
  `0` with `7 passed in 0.01s`.
- At this point integrated pytest, service and browser workflows had not run. Subsequent pytest and
  API/browser evidence is appended below rather than rewriting this earlier state.

### 2026-09-01 — Full data/model and repository tests

- The data/algorithm agent ran the full `pipeline all` command with rank 32 and seed 20260901;
  exit `0`, prepare `1.755s`, total `19.323s`.
- This frontend/delivery round independently parsed `data/processed/summary.json`,
  `artifacts/current.json`, the referenced manifest and `metrics.json`. The pointer resolves to
  `svd-20260901T121430026505Z-cccf5c24`; arrays contain 49,416 users, 16,907 items and rank 32.
- `.venv\\Scripts\\python.exe -m pytest -p no:cacheprovider` exited `0` with
  `15 passed, 1 warning in 4.71s`. The warning is a Starlette deprecation notice for the current
  httpx TestClient integration; it does not fail the tests but remains a dependency-upgrade risk.
- At the time of this run, service/browser and clean-checkout evidence had not yet been recorded.
  Subsequent API/browser evidence is appended below; clean-checkout reproduction remains PENDING.

### 2026-09-01 — Main-thread live API smoke

- The running service produced a real recommendation request and persisted `+1` request and `+5`
  exposures. Sending linked behavior produced `+1` click and `+1` like in the Dashboard aggregates.
- The operation sequence targeted item `2363`. An active boost placed it at position `0`; after the
  item was set offline, the direct item API returned `404` and personalized, popular and explore
  feeds all omitted it. Offline therefore overrode the still-valid boost.
- Restoring the item succeeded and the status/boost operations were present in the audit trail.
- The handoff did not include the exact orchestration command or full response bodies, so this note
  records only the observed deltas and rule outcomes, not invented command text.

### 2026-09-01 — Main-thread browser smoke

- The same-origin Web client was exercised at desktop `1440px` and mobile `390px` widths.
- Alice Feed behavior and profile updates were visible. The administrator Dashboard, item search,
  targeted boost for item `2363`, offline-over-boost behavior, restore and audit were exercised.
- Browser console evidence was zero warnings and zero errors for these paths.
- This was not a complete replay of every `docs/DEMO.md` step: browser A/B comparison, carol
  cold-start, clean checkout and video recording remain unverified.

### Still pending after integration

- All three normal users in one browser sequence and the 3–5 minute recorded demonstration. The
  same cases are covered by API/E2E automation; video recording is explicitly postponed by the user.

### 2026-09-01 — Lead closeout reruns

- Plain `uv sync --group dev` first exited `1` because the sandboxed user-level uv cache path could
  not be initialized (`os error 183`). The same locked sync was rerun with the ignored repository
  cache: `uv sync --cache-dir .uv-cache --group dev`; it exited `0`, resolving 33 and checking 30
  packages. The failed attempt is retained as environment evidence.
- After the `.env` loading change, the first full pytest rerun exposed one obsolete Web contract
  assertion that still required the README to say `PENDING VERIFICATION`. The assertion was updated
  to require the truthful remaining gaps instead. The final rerun exited `0` with `15 passed,
  1 warning in 6.07s`; `compileall` and both JavaScript syntax checks also exited `0`.
- The lead stopped the prior service, rebuilt `data/app.db`, imported 19,220 items, seeded
  alice/bob/carol/admin and loaded `svd-20260901T121430026505Z-cccf5c24` without a warning.
  `/api/v1/health` returned `status=ok`, `database=ok`, that model version and `model_error=null`.
- A fresh mobile browser pass at 390 px rendered 12 real personalized cards. The DOM audit returned
  `innerWidth=390`, `scrollWidth=375`; the browser console contained no warning or error entries.
- A clean clone from the four committed revisions was created in an ignored temporary directory.
  After copying the three official files as the documented user-supplied input, locked dependency
  sync succeeded, the smoke `pipeline all` command exited `0`, imported all 359,708 interactions,
  recreated the same cutoffs and published a 2,000-user smoke model. DB initialization imported
  19,220 items and seeded alice/bob/carol/admin; pytest returned `15 passed, 1 warning in 5.61s`,
  both JavaScript checks exited `0`, and health returned DB/model status `ok`.
- Git tracked-path and history audits found no raw/processed data, artifact, database, `.env`,
  assessment DOCX or large model file. Repository objects total about 142 KiB before packing. Four
  implementation commits separately cover contracts/config, offline pipeline, backend service and
  frontend/delivery; this evidence update is committed separately.

### 2026-09-01（晚间）— 方案C 落地 + 后续计划 6/7/8/9 实施

- **方案C 完成**：`recsys/data.py` 与 `recsys/artifacts.py` 已彻底移除目录级 `tempfile`
  （`mkdtemp`/`TemporaryDirectory`），改为带 UUID 的 `Path.mkdir()` 暂存目录；相同任务并发时
  不会互删对方的活动目录。
  新增回归测试 `tests/test_no_tempfile_regression.py`（打桩 `tempfile.mkdtemp` 与
  `TemporaryDirectory` 为抛错，断言 `prepare_data`/`write_staged_artifact` 仍成功）。
- 全量 pytest 现为 **19 passed**（原 15 + 回归 2 + export 2），唯一 warning 仍是 Starlette 的
  httpx TestClient 弃用提示（非失败）。`node --check` 对 `web/app.js`、`web/api.js` 均 exit 0。
- **7b 增量**：V-02/V-03/V-12 的证据建立在 `svd-20260901T121430026505Z-cccf5c24`；方案C 修复后
  按相同 seed=20260901 / rank=32 / 相同数据重跑 pipeline，发布的新版本为
  `svd-20260901T150342412251Z-cccf5c24`，`data_version=microlens50k-cb7fb01dc9f42b6b`（与 V-02
  记录一致），SVD test Recall@10=0.359176 / NDCG@10=0.208986 / HitRate@10=0.399（与 V-02/V-03
  记录一致）。版本号差异仅来自版本串内嵌的发布时刻，config digest `cccf5c24` 一致，证明内容
  确定一致。`artifacts/current.json` 现指向新版本。
- **6a 自动化（API 级）**：干净 seed 重建后（19,220 items、alice/bob/carol/admin），
  `/api/v1/health` 返回 `status=ok / database=ok / model_version=<新版本> / model_error=null`。
  真实模型下：alice/bob 个性化 source=model、fallback=None；**A/B 首项不同**；carol
  fallback=cold_start、source=popular；分页 page1/page2 各 5 项、**重叠为空**。
- **8a**：`python -m app.cli export-events` 保持一事件一行，输出
  `user,item,timestamp,event_type,weight`；click=1/like=3 是显式权重列，
  `consumed_by_training=false`，不会混入旧 benchmark。
- **9-1 / 9-2a / 9-3a**：dashboard 新增 p50/p95/p99/max 延迟分位；新增
  `GET /api/v1/admin/dashboard/timeseries`（metric∈requests/served_exposures/
  viewable_impressions/clicks/likes，自动
  hour/day 分桶）+ 前端 SVG 折线 + 指标下拉；新增 JSON 结构化日志（`app/logging.py` +
  请求日志中间件，每请求一条 JSON：request_id/method/path/status_code/duration_ms）。
- **环境差异提示（诚实记录）**：本会话 PowerShell 为 5.1、`git` 不在 PATH，与上表 Environment
  记录的 PowerShell 7.6.4 / Git 2.53.0 来自不同会话。本次实施未做 git 提交，提交仍需在具备 git
  的环境完成。

## Browser Acceptance Checklist（6a，待主线程执行）

> 下列为仍需在真实浏览器完成、无法被 API/pytest 覆盖的验收面。API 级 A/B、carol 冷启动、
> 分页、空候选回退已由 pytest + 上面的 API smoke 覆盖，此处只需复核浏览器呈现与交互。

**前置**：`python -m app.cli init-db --reset`（干净 seed）；按 README 启动 uvicorn；打开
DevTools → Console（全程零 error/警告），窗口分别用 1440px 与 390px 各过一遍。

- [ ] **三用户登录**：alice/bob/carol 分别登录成功；右上 `.session-user` 显示用户名；
  用 `#logout-button` 登出再登录，会话刷新正常。
- [ ] **A/B 差异（浏览器）**：alice 登录 → 「个性化」tab → 记录前 3 张 `.feed-card` 的
  `data-item-id`；登出，bob 登录 → 记录前 3 张；两者**至少首项不同**（API 已证）。
- [ ] **carol 冷启动**：carol 登录 → 「个性化」→ `#fallback-context` 可见、`#feed-fallback`
  为 `cold_start`，卡片 `.source-label` 为 `popular`；`#feed-model-version` 仍显示当前模型版本。
- [ ] **分页去重**：任一用户个性化流，点击 `#load-more-button` 追加一页，观察 `#feed-list`
  中 `.feed-card` 数量增加且**无重复 item_id**（API 已证，此处核 UI）。
- [ ] **行为反馈**：点击封面/标题（记 click）、点「喜欢」、点「不感兴趣」；「不感兴趣」后对应
  卡片从 `#feed-list` 移除；顶部出现成功提示（`.global-alert`）。
- [ ] **画像**：点 `#profile-button` → `#profile-dialog` 打开，`#profile-summary` 显示曝光/点击/
  喜欢/不感兴趣计数，`#profile-events-body` 列出最近事件。
- [ ] **Dashboard 新面板**：admin 登录 → Dashboard → `#metric-grid` 出现「延迟 P50/P95/P99」；
  切换 `#timeseries-metric`（请求/曝光/点击/喜欢），`#timeseries-chart` 内 SVG 折线随切换刷新。
- [ ] **强制断网（错误态）**：DevTools → Network → Offline，切信息流 tab 或点刷新，`#feed-state`
  显示错误面板（含「重试」）；恢复 Online 后点重试，列表恢复。
- [ ] **空态**：候选耗尽/全部下线的空态已由 pytest 覆盖（断言 200 + 空 items 与空态文案），
  浏览器不单独强验（需下架 19,220 项不现实）；断网错误态与空态共用同一 `#feed-state` 面板。
- [ ] **响应式 + 控制台**：1440px 与 390px 各过一遍主流程，`innerWidth` 正确、无横向溢出
  （`scrollWidth <= innerWidth`）、Console 无 error/警告。

### 2026-09-02 — Mandatory closure correction and rerun

- The earlier 2026-09-01 `8a` note said event weights were represented by duplicate rows. That
  description is superseded. `export-events` now emits exactly one row per source click/like with
  explicit `event_type` and `weight` columns, defaults to ignored
  `data/staging/online_events.csv`, and reports `consumed_by_training=false`. The current benchmark
  intentionally does not merge these events without establishing a new chronological cutoff.
- The full official model was regenerated as `svd-20260902T022510026424Z-cccf5c24`. Its full-data
  protocol and Popular/Random/SVD metrics match the prior deterministic run, and both validation and
  test payloads now include five anonymized, ranked SVD Badcases with miss reasons and coverage
  context.
- Main-thread final full regression returned `31 passed, 1 warning in 13.79s`; Python `compileall` and both
  JavaScript syntax checks exited `0`. The warning remains the third-party Starlette/httpx
  deprecation notice.
- `python -m scripts.verify_official_e2e` exited `0` against the current official artifact. It
  imported 19,220 items, proved alice/bob personalized differences, carol `cold_start`, stable
  snapshot replay, request/event linkage, and Dashboard delta `+6 requests/+45 served exposures/
  +3 viewable impressions/+1 click/+1 like`; it also covered all-feed boost,
  offline precedence across three feeds plus item/cover APIs, restore/audit and logout `401`.
- Browser verification showed real model-backed personalized results, profile version/ranking
  response to behavior, trusted Feed/source plus request ID in recent behavior, real Dashboard
  aggregates, per-feed behavior, latency and trends. At 390 px the document width was 375 for a
  390 px viewport. A forced service outage displayed the connection error and retry action; restarting
  the service and clicking retry restored the Dashboard. The viewport was reset afterward.
- A focused P2 regression run initially failed because the new 366-day bound had been applied to
  the overview route but not the time-series route. The failure (`1 failed, 12 passed`) was retained,
  the guard was moved to the correct route, and the identical target then returned
  `13 passed, 1 warning`. The same closeout also gave data/model staging directories UUID suffixes
  so identical concurrent tasks cannot delete each other's active directory, and disabled Uvicorn's
  duplicate access logger in favor of the request-ID-correlated application record.
- The first clean-source `uv sync --offline` attempt could not reuse the original cache because a
  cache-internal `.git` directory had a restricted ACL. Copying only readable cache content to the
  ignored clean directory and specifying the bundled Python interpreter resolved this environment
  issue; locked sync installed 30 packages. In that clean directory, the official-data smoke pipeline,
  19,220-item DB seed, `26 passed`, Python compilation, both JavaScript checks and real Uvicorn health
  on port 8011 all exited successfully. The temporary 8011 service was then stopped.
- The final service was restarted on port 8010 from the current source with `--no-access-log`.
  `/api/v1/health` returned the full official model with `status=ok`, `database=ok` and
  `model_error=null`; server output contained exactly one correlated `app.access` JSON record for
  that request and no duplicate `uvicorn.access` record.

## 2026-09-02 Semantics and Snapshot Upgrade

- Feed request with 12 items persisted 12 served exposures and zero automatic impressions.
  Browser verification then produced exactly 3 viewable impressions for cards meeting 50%/750 ms.
- Cursor tests cover signed user/feed binding, tamper rejection, TTL expiry, deterministic replay,
  profile/model/new-boost isolation, immediate offline invalidation and no restore reactivation.
- Official data isolation smoke processed 359,708 interactions and published a temporary model
  artifact whose metadata records cutoff-safe cumulative/1/7/30-day/decay/growth popularity
  features. The cumulative likes/views snapshot had `available_at=null` and was excluded.
- Forward migration was applied to a temporary copy of the prior SQLite DB; all 19,220 items
  remained and the snapshot/provenance tables and trace columns were present.

## 2026-09-04 Delivery Audit

- The first direct pytest rerun could not scan the user-level Windows temporary
  directory and stopped with `WinError 5`; this was an environment permission
  failure before affected fixtures ran, not a product assertion failure.
- `uv run python -m pytest -q -p no:cacheprovider --basetemp=.tmp/pytest-delivery-20260904-02`
  then exited `0` with **102 passed** and one known Starlette/httpx deprecation
  warning. The base directory was a new ignored workspace path.
- Git history contains six meaningful local commits before the current delivery work.
  A tracked-path audit found no dataset, model/checkpoint, database, `.env`, assessment
  document or secret-bearing generated artifact.
- `origin` is configured as `https://github.com/BlossomRa1n/Yahaha.git`. The remote was
  empty/unverified at the start of this audit; push and remote branch verification are
  recorded only after they actually succeed.
- The TypeScript interop directory is source-only on this machine: Bun, a TypeScript
  compiler, installed packages and a dependency lockfile are absent. Live PostgreSQL,
  Elasticsearch and upstream proxy success paths have not been verified.
- The recording procedure is retained in `docs/DEMO.md`; the candidate later supplied the external
  video link. The final link check is recorded below rather than rewriting this historical note.

### 2026-09-04 — Demonstration video delivery

- Filename: `江雨鸿-视频演示.mp4`.
- External share: <https://pan.quark.cn/s/81e4e62f191a>.
- `HEAD`/public landing-page requests returned HTTP `200`; the HTML title was `夸克网盘分享`.
