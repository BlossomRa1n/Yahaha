# API and Artifact Contract

All API paths use `/api/v1`. JSON timestamps are ISO-8601 UTC. IDs are serialized
as strings in JSON even when SQLite stores numeric item IDs.

## Authentication

- `POST /auth/login` body: `{ "username": "alice", "password": "demo-pass" }`
- `GET /auth/me`
- `POST /auth/logout`

Login sets an HttpOnly session cookie. Normal requests never accept a user ID as
an identity override.

## Feeds

`GET /feeds/{personalized|popular|explore}?limit=12&cursor=<opaque>`

```json
{
  "request_id": "uuid",
  "snapshot_id": "uuid",
  "feed_type": "personalized",
  "model_version": "svd-20260901T120000Z",
  "profile_version": 3,
  "fallback_reason": null,
  "next_cursor": "opaque-or-null",
  "has_more": true,
  "items": [
    {
      "item_id": "123",
      "title": "Example",
      "cover_url": "/api/v1/items/123/cover",
      "position": 0,
      "snapshot_position": 12,
      "source": "model",
      "score": 0.842,
      "explanation": "Latent affinity plus recent positive feedback",
      "model_version": "svd-20260901T120000Z",
      "is_forced": false
    }
  ]
}
```

The first page creates a bounded persistent candidate snapshot. The signed cursor
contains only snapshot identity, offset, user/feed bindings and expiry. Later pages
read the same ordered snapshot even if the model, profile or boost rules change.
Offline invalidation still applies immediately and can make an old page short.

Each returned item is a served exposure fact. The server atomically stores the
request and ordered `exposures`; it does not infer that the item was visible.

## Behavior

`POST /events/batch`

```json
{
  "events": [
    {
      "event_id": "imp:feed-request-uuid:123:0",
      "event_type": "impression",
      "request_id": "feed-request-uuid",
      "item_id": "123",
      "position": 0,
      "client_timestamp": "2026-09-01T12:00:00Z"
    }
  ]
}
```

Accepted client types are `impression`, `click`, `like`, `favorite` (normalized
to `like`), `not_interested`, `dwell` and `share`; `revisit` is generated only
by the server from a second cross-request viewable visit. A dwell event requires
`dwell_ms` in `[750,600000]`; a client `visit_index` is never trusted. The web client reports impression only after at
least 50% visibility for 750 ms and uses
`imp:{request_id}:{item_id}:{position}` as its stable idempotency key. The response gives
accepted/duplicate counts and the resulting profile version. A mismatched user,
item or position is rejected.

## User Data

- `GET /me/profile`
- `GET /me/events?limit=50` returns each persisted event with its trusted
  `feed_type` and exposure `source` resolved server-side.
- `GET /items/{item_id}` returns only online content.

## Administration

- `GET /admin/dashboard/overview`
- `GET /admin/dashboard/timeseries?metric=requests|served_exposures|viewable_impressions|clicks|likes|shares|revisits|dwell_events&from=&to=`
- `GET /admin/dashboard/export.csv?from=&to=`
- `GET /admin/requests/{request_id}`
- `GET /admin/users/{user_id}/debug`
- `GET /admin/items?q=&status=&limit=&offset=`
- `PATCH /admin/items/{item_id}/status` body:
  `{ "status": "offline|online", "reason": "..." }`
- `PATCH /admin/items/batch/status` body:
  `{ "item_ids": ["1", "2"], "status": "offline|online", "reason": "...", "idempotency_key": "..." }`.
  The unique item count is limited to 100. Validation and updates are atomic;
  an unknown item changes nothing. Successful retries return the original
  `batch_id`, and each item has one operation row linked by that ID.
- `POST /admin/boosts` body includes `item_id`, `audience` (`all|users`),
  `user_ids`, `feed_types`, `position`, `priority`, `starts_at`, `ends_at`, and
  `reason`.
- `GET /admin/operations`
- `GET /admin/models`
- `GET /admin/models/compare?versions=v1&versions=v2` selects 2-10 real model
  versions. It returns training windows, sample/event counts, publication state,
  current-version status, test SVD metrics and deltas only when K, candidate
  universe, negative sampling and aggregation protocol are compatible.
- Threshold alerting over live DB-aggregated metrics (`requests`, `exposures`,
  `impressions`, `clicks`, `likes`, `ctr`, `active_users`, `latency_p95`,
  `offline_items`): `GET /admin/alerts/metrics`, `GET|POST /admin/alerts/rules`,
  `PATCH|DELETE /admin/alerts/rules/{rule_id}`, `GET /admin/alerts/events`,
  `POST /admin/alerts/events/{event_id}/acknowledge`, `POST /admin/alerts/evaluate`.
  Reads require `analyst`, writes require `operator`. Rules fire only when the
  current metric value actually breaches its threshold; evaluation is
  deterministic and only writes `alert_events` (open → resolved transitions).

Status, boost and audit mutations are server-authorized and transactional.
Dashboard overview returns DB-derived totals, latency percentiles, top items and
three `feed_breakdown` rows containing `requests`, `served_exposures`,
`viewable_impressions`, `clicks`, `likes`, `not_interested`, `shares`,
`revisits`, `average_dwell_ms`, `served_ctr`,
`viewable_ctr` and served exposure `share`. Legacy `exposures`/`ctr` aliases
mean served exposures/clicks per served exposure and include
`ctr_denominator=served_exposures`. The response also exposes the viewable
semantics activation timestamp so pre-migration auto-impressions are not silently mixed.
Timeseries uses a half-open
`[from,to)` range and hour buckets up to 48 hours, otherwise day buckets.

Expired cursors return `410 cursor_expired`. Run
`python -m app.cli cleanup-snapshots` to delete expired SQLite snapshots; valid
snapshots are retained.

Feed responses also include `diversity`. `before` and `after` report title-token
adjacent similarity, intra-list diversity, unique token count and duplicates.
The deterministic MMR order is stored in the snapshot; low-quality title metadata
sets `applied=false` and leaves ordinary ranking unchanged.
`diversity.source_mix` reports the actual `mix_policy_version`, warm/cold and
support context, requested/selected quotas, source availability, quota relaxations,
fallback count and source Jaccard diagnostics. These are computed from the request,
not frontend constants.

Personalized Feed is served by the unified 7-source + DeepFM path as its sole
production path (the stable pointer is unchanged; SVD is a recall source):

- SVD, DSSM, title content, cached MobileNet visual profiles, item-item CF, popular
  and explore each generate bounded candidates. Seen/negative/offline rows and
  zero-support CF rows are removed before a deduplicated union is built.
- DeepFM ranks the open union using seven normalized source scores, source multi-hot,
  user/item, popularity/history buckets, cold-item and visual-availability features.
  Source limits are upper bounds only; no source is forced into Top-K.
- Per-Top-10 source caps, title-token diversity, legal boosts and a final offline
  guard run after model scoring. Responses expose the composite deep+multimodal
  model version, primary source and all contributing sources in the explanation.
- The deep manifest pins the exact multimodal artifact. Missing, corrupt or
  incompatible deep/visual artifacts fall back to popular (filled with explore on
  shortfall) rather than a separate SVD mix; no request-time image inference occurs.
- On the sampled-all-items protocol the pure-DeepFM ranking improved validation NDCG
  (0.205 vs 0.181) but regressed Recall (0.320 vs 0.382) and HitRate (0.413 vs 0.486),
  so the sampled Recall release gate failed; the trade-off is accepted, not masked.

## Event-window Retraining

`python -m app.cli export-events --out data/staging/online_events.csv` writes one
row per accepted event. `python -m app.cli retrain-events --start-time ...
--end-time ... --base-processed-dir data/processed --output-root data/retraining
--mode smoke` consumes the half-open server-receive-time window, validates users
and items, deduplicates events, creates a traceable data version, regenerates
chronological splits and publishes a unique model only after evaluation and load
validation. Failed runs remain in `training_runs` and leave `current.json` unchanged.

The same retraining can be queued as a non-blocking HTTP job:
`POST /admin/training/jobs` (`operator`) accepts `start_time`, `end_time`, `mode`,
`max_users`, `max_eval_users`, `rank`, `seed`, `base_processed_dir`, `output_root`,
returns `202` immediately with a `queued` job, and runs `run_online_retraining` in
a background thread (same publish semantics as the CLI). `GET /admin/training/jobs`
and `GET /admin/training/jobs/{job_id}` (`analyst`) report the lifecycle
`queued → running → succeeded|failed` recorded in `training_jobs`.

## Errors

```json
{
  "error": {
    "code": "forbidden",
    "message": "Administrator role required",
    "request_id": "api-request-uuid",
    "details": null
  }
}
```

## Model Artifact

```text
artifacts/<version>/manifest.json
artifacts/<version>/user_ids.npy
artifacts/<version>/item_ids.npy
artifacts/<version>/user_factors.npy
artifacts/<version>/item_factors.npy
artifacts/<version>/popularity.json
artifacts/<version>/content_item_vectors.npz
artifacts/<version>/content_user_vectors.npz
artifacts/<version>/content_config.json
artifacts/<version>/item_cf_neighbors.npz
artifacts/<version>/item_cf_user_history.npz
artifacts/<version>/item_cf_config.json
artifacts/current.json
```

The manifest includes schema version, data version, algorithm, dimensions,
training configuration/time, evaluation protocol/metrics, file hashes and array
shapes. `metrics.json` and `evaluation.md` also include stable anonymized SVD
Badcases with sampled-candidate ranks and coverage limitations. No Python pickle
is accepted.
`manifest.json.mix_policy` contains schema version, the validation-only candidate
comparison, explicit 1% Full/Warm quality gates and the locked policy version.
Serving defaults legacy artifacts without this section to the safe SVD/content
policy and rejects an unknown policy version instead of attempting runtime mixing.

Processed data also includes `stats_snapshot.json` with the cumulative
likes/views snapshot version, `available_at`, source file SHA-256/name and
mtime, row count and quality summary. The database records the actual import
time separately. Missing `available_at` disables this snapshot for historical
training/evaluation. Popularity artifacts use only
interaction events satisfying `event_timestamp <= feature_cutoff` and record
the 1/7/30-day and time-decay feature definitions.

## Experimental Deep and Multimodal Artifacts

Deep and visual models are isolated from the stable pointer:

```text
artifacts/experiment-current.json
artifacts/<deep-version>/manifest.json
artifacts/<deep-version>/dssm.safetensors
artifacts/<deep-version>/deepfm.safetensors
artifacts/<deep-version>/deep_*_embeddings.npy
artifacts/<deep-version>/training.json
artifacts/<deep-version>/metrics.json

artifacts/multimodal-current.json
artifacts/<multimodal-version>/manifest.json
artifacts/<multimodal-version>/visual_item_embeddings.npy
artifacts/<multimodal-version>/visual_user_profiles.npy
artifacts/<multimodal-version>/visual_available.npy
artifacts/<multimodal-version>/extraction.json
artifacts/<multimodal-version>/metrics.json
```

Each pointer is atomically replaced only after manifest/file hash validation and
model warmup. Deep manifests bind the base stable model, data version, cutoff,
feature/config versions, DSSM/DeepFM settings, selected `rank_strategy`,
`stable_rank_weight`, checkpoint history and evaluation query hashes. Multimodal
manifests bind the same base model and cutoff plus image/encoder hash, PCA feature
version, coverage, and `selected_warm_visual_weight` /
`selected_cold_visual_weight`. Loaders reject unknown strategies, non-finite
weights, base-version mismatches, missing files and hash mismatches. The stable
`artifacts/current.json` pointer is not modified by experimental training.
