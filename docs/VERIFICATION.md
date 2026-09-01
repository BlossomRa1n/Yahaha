# Verification Log

这里是完成状态的唯一运行证据账本。代码存在、代理总结或 README 命令不等于验证。
执行者必须记录日期、完整命令、退出码、关键输出和失败/降级；长输出可引用仓库内不含
敏感数据的日志路径。禁止补写未实际运行的成功结果。

状态定义：`PASSED` 表示表中所述范围有实际证据；`PARTIAL` 表示只覆盖了部分验收面；
`PENDING` 表示尚无足够证据。代理运行结果与主代理复核必须明确区分，不能用代理总结
替代主代理的浏览器、API 或干净环境验收。

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
| V-11 | Secret/raw audit | Inspect tracked files and history | No completed tracked-file/history audit recorded | Pending lead run | PENDING |
| V-12 | Clean environment | Repeat README from fresh checkout | Not run | Pending lead run | PENDING |
| V-13 | Live event/Dashboard smoke | Feed then click/like against running service | `+1` request, `+5` exposures, `+1` click and `+1` like observed in real aggregates | Main-thread API smoke | PASSED |

## Offline Metrics

Do not transcribe values until V-03 succeeds. Record the artifact version, data version, cohort sizes,
seed, negative count and the popular/random/SVD Recall@10, NDCG@10 and HitRate@10 values from the
generated `metrics.json`.

| Artifact version | Data version | Mode | Cohort | Metrics | Evidence | Status |
|---|---|---|---|---|---|---|
| `svd-20260901T121430026505Z-cccf5c24` | `microlens50k-cb7fb01dc9f42b6b` | full | test: 5,000 users, 100 negatives, coverage 0.237160 | Popular R/N/H 0.246499/0.123706/0.281800; random 0.089834/0.043384/0.108800; SVD 0.359176/0.208986/0.399000 | `manifest.json`, `metrics.json` inspected | PASSED |

## End-to-End Acceptance

| Flow | Evidence to record | Result | Status |
|---|---|---|---|
| Login/session/isolation | 3 normal users, admin, refresh, logout, 401/403 | Automated auth/logout/403/isolation tests passed; browser login covered alice and admin, not all 3 normal users | PASSED (automated), PARTIAL (browser) |
| Three feeds | request_id, provenance, pagination, A/B difference, cold start | Automated Feed contract passed; backend agent reported real alice/bob differences; main thread verified offline removal from all three feeds. Full browser A/B, pagination and carol cold-start sequence remains | PARTIAL |
| Event/profile | DB event linkage and before/after profile/rank | Main API smoke produced +1 request/+5 exposures/+1 click/+1 like; browser showed alice behavior and updated profile | PASSED |
| Dashboard | Numeric baseline and post-action delta | Main thread observed real post-action deltas and admin Dashboard; top items came from DB aggregation | PASSED |
| Boost/offline/restore | Target feed, all-path filter, audit before/after | Item 2363 appeared as forced position 0; offline won over boost, direct API returned 404, three feeds omitted it; restore and audit passed | PASSED |
| Error/fallback | Missing model, empty candidates, offline item, network error | Automated corrupt-model fallback and live offline 404 passed; browser empty/network-failure states were not forced | PARTIAL |
| Responsive/console | Desktop and mobile layout, browser console | 1440 px and 390 px workflows completed with zero console warnings/errors | PASSED |
| Clean reproduction/video | Fresh checkout and 3–5 minute recording | Neither was performed | PENDING |

## Append-Only Run Notes

Add dated entries below. Include failures and subsequent fixes; never replace an earlier failed run.

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

- A fresh-checkout/clean-environment reproduction.
- Tracked-file/history audit for raw data, artifacts, `.env`, secrets and meaningful commit count.
- All three normal users in browser, forced network/empty states and a 3–5 minute recorded
  demonstration.

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
