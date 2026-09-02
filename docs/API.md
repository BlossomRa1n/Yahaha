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
      "source": "model",
      "score": 0.842,
      "explanation": "Latent affinity plus recent positive feedback",
      "model_version": "svd-20260901T120000Z",
      "is_forced": false
    }
  ]
}
```

Each response is the impression fact. The server atomically stores its request,
exposures and one `impression` event per item.

## Behavior

`POST /events/batch`

```json
{
  "events": [
    {
      "event_id": "client-uuid",
      "event_type": "click",
      "request_id": "feed-request-uuid",
      "item_id": "123",
      "position": 0,
      "client_timestamp": "2026-09-01T12:00:00Z"
    }
  ]
}
```

Accepted types are `click`, `like`, `favorite` (normalized to `like`) and
`not_interested`; impression is generated only by the server. The response gives
accepted/duplicate counts and the resulting profile version. A mismatched user,
item or position is rejected.

## User Data

- `GET /me/profile`
- `GET /me/events?limit=50` returns each persisted event with its trusted
  `feed_type` and exposure `source` resolved server-side.
- `GET /items/{item_id}` returns only online content.

## Administration

- `GET /admin/dashboard/overview`
- `GET /admin/dashboard/timeseries?metric=requests|exposures|clicks|likes&from=&to=`
- `GET /admin/requests/{request_id}`
- `GET /admin/users/{user_id}/debug`
- `GET /admin/items?q=&status=&limit=&offset=`
- `PATCH /admin/items/{item_id}/status` body:
  `{ "status": "offline|online", "reason": "..." }`
- `POST /admin/boosts` body includes `item_id`, `audience` (`all|users`),
  `user_ids`, `feed_types`, `position`, `priority`, `starts_at`, `ends_at`, and
  `reason`.
- `GET /admin/operations`
- `GET /admin/models`

Status, boost and audit mutations are server-authorized and transactional.
Dashboard overview returns DB-derived totals, latency percentiles, top items and
three `feed_breakdown` rows containing `requests`, `exposures`, `clicks`, `likes`,
`not_interested`, `ctr` and exposure `share`. Timeseries uses a half-open
`[from,to)` range and hour buckets up to 48 hours, otherwise day buckets.

## Future-training Event Snapshot

`python -m app.cli export-events --out data/staging/online_events.csv` writes one
row per accepted mapped-user click/like with columns
`user,item,timestamp,event_type,weight`. This file is staging metadata and is not
consumed by the current train-only benchmark. A consumer must create a new
chronological split; old validation/test metrics cannot be reused after merging.

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
artifacts/current.json
```

The manifest includes schema version, data version, algorithm, dimensions,
training configuration/time, evaluation protocol/metrics, file hashes and array
shapes. `metrics.json` and `evaluation.md` also include stable anonymized SVD
Badcases with sampled-candidate ranks and coverage limitations. No Python pickle
is accepted.
