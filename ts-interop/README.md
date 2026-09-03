# microlens-ts-interop

An **isolated, self-contained** TypeScript/Bun interop sidecar that lets the
existing MicroLens recommendation MVP (Python / FastAPI / SQLite) participate in
a hive-side stack built on **Bun / Express / Prisma / PostgreSQL / Elasticsearch**.

It is fully **additive**: everything lives under `ts-interop/` and it never
modifies or depends on the Python web app. The two services communicate only
over HTTP.

## Purpose

- **Read path** (`/health`, `/feeds/:feedType`) works even when PostgreSQL,
  Elasticsearch, or the upstream FastAPI service are down (graceful degradation).
- **Write path** (`/ingest/events`) fans events out to PostgreSQL (via Prisma)
  and Elasticsearch (index `microlens-events`).

## Prerequisites (external services — NOT started by this repo)

This repository does **not** start or provision these. You must run them
yourself before the write path will succeed:

1. **PostgreSQL** — reachable via `DATABASE_URL`. Apply the schema with
   `bun run prisma:migrate` (or `prisma migrate dev`) before ingesting.
2. **Elasticsearch** — reachable via `ELASTICSEARCH_URL`.
3. **Upstream FastAPI service** — the existing Python app running at
   `FASTAPI_URL` (only needed for `/feeds/:feedType` and the `upstream` health
   flag).

The read/health path degrades gracefully: with none of the above running,
`GET /health` still returns `200` with `pg: false`, `es: false`,
`upstream: false`.

## Install

```bash
bun install
bun run prisma:generate   # generate the Prisma client from prisma/schema.prisma
```

If you use npm instead of Bun, `npm install` works too (the scripts above use
`bun`; you can run `npx prisma generate` and `npx tsc --noEmit` directly).

## Configure

```bash
cp .env.example .env   # then fill in real values
```

No real credentials are committed — `.env.example` contains placeholders only,
and `.env` is git-ignored.

## Run

```bash
bun run dev          # bun --watch src/server.ts (default port 4000)
# or
bun run start
```

Typecheck:

```bash
bun run typecheck    # tsc --noEmit
```

## API surface

### `GET /health`

```json
{ "status": "ok", "pg": true, "es": true, "upstream": true }
```

Each dependency is probed in its own `try/catch`; a down dependency simply
reports `false`. The endpoint itself never throws and always returns `200`.

### `GET /feeds/:feedType`

Read-through proxy to `FASTAPI_URL/api/v1/feeds/:feedType`. `feedType` is one of
`personalized | popular | explore`. Optional `?limit=` and `?cursor=` query
params are forwarded, and the `Cookie` header is forwarded so the upstream
session auth still works. The upstream JSON body is passed through verbatim, so
each item's `source`, `score`, and `model_version` fields are preserved.

- Upstream reachable → the upstream status code + body are returned.
- Upstream down → `502` with `{"error":{"code":"upstream_unavailable",...}}`.

### `POST /ingest/events`

Accepts an array of events in the same shape as the upstream `ClientEvent`
(`event_type`, `request_id`, `item_id`, `position`, `client_timestamp`, and the
optional `dwell_ms` / `visit_index`). One documented extension: an optional
`user_id` field is accepted because the sidecar has no authenticated session to
derive it from (the upstream derives `user_id` from the login cookie).

```json
{
  "events": [
    {
      "event_id": "evt-1",
      "event_type": "click",
      "request_id": "req-1",
      "item_id": "1001",
      "position": 0,
      "client_timestamp": "2026-09-04T10:00:00Z",
      "user_id": "u-1"
    }
  ]
}
```

For each event it attempts (a) a PostgreSQL write via Prisma and (b) an
Elasticsearch index into `microlens-events`. `favorite` is normalized to `like`
to match upstream persistence semantics. The response is always `200`:

```json
{
  "received": 1,
  "pgWritten": 1,
  "esIndexed": 0,
  "errors": [{ "event_id": "evt-1", "es": "elasticsearch_index_failed" }]
}
```

If PG or ES are down, the failure is recorded per event in `errors`; the route
does not crash.

## What I verified / did not verify

See the "Verification" section in the final delivery report (and NOTES.md for
the exam mapping). In short: `bun` and `tsc` are not installed in the build
environment at the time of writing, so any typecheck claim is only as strong as
the tooling available — if it was run, the exact command and result are stated
there; otherwise this is noted explicitly and not fabricated.

## Files

```
ts-interop/
├── package.json          # Bun runtime/package manager, scripts, deps
├── tsconfig.json         # strict ESNext / NodeNext ESM config
├── .env.example          # placeholders only, no real credentials
├── .gitignore
├── prisma/
│   └── schema.prisma     # PostgreSQL mirror of the core SQLite tables
└── src/
    ├── server.ts         # Express app: /health, /feeds/:feedType, /ingest/events
    ├── prisma.ts         # lazy PrismaClient + checkPostgres()
    └── es.ts             # lazy ES client + checkElasticsearch()
```
