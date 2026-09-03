# Notes: mapping sidecar components to the "工程与部署" (engineering & deployment) bonus item

This sidecar is a bonus deliverable proving the Python/FastAPI/SQLite MVP can
participate in a hive-side TypeScript/Bun/Express/Prisma/PostgreSQL/Elasticsearch
integration. Below, each component is mapped to the engineering/deployment
concern it demonstrates.

| Component | What it demonstrates (engineering & deployment concern) |
| --------- | ------------------------------------------------------- |
| `package.json` | Dependency + toolchain management. Pins the Bun runtime/package manager, runtime deps (`express`, `@prisma/client`, `@elastic/elasticsearch`, `zod`) and dev deps (`typescript`, `@types/*`, `prisma`); exposes `dev`, `typecheck`, `prisma:generate`, `prisma:migrate` scripts. |
| `tsconfig.json` | Reproducible, strict TypeScript build (`strict`, ESNext, NodeNext ESM, `outDir dist`), so the sidecar typechecks independently of the Python app. |
| `prisma/schema.prisma` | Schema-as-code / migration of the SQLite core tables (`users`, `items`, `events`, `exposures`, `recommendation_requests`, `operations`, `model_versions`) into PostgreSQL, with SQLite CHECK constraints mirrored as enums. |
| `src/prisma.ts` | Lazy/optional Prisma client + connectivity probe, enabling **graceful degradation** (health/read path works with no PostgreSQL). |
| `src/es.ts` | Lazy/optional Elasticsearch client + connectivity probe, same graceful-degradation guarantee for Elasticsearch. |
| `src/server.ts` | Service integration layer: an Express API with a health endpoint (observability), a read-through proxy to the upstream FastAPI feed (interop between two services), and an ingest route that fans events out to PostgreSQL + Elasticsearch with per-event error accounting (fault isolation — a down sink never crashes the route). |
| `.env.example` | Configuration management + **secrets hygiene**: placeholders only, no real credentials committed, `.env` git-ignored. |
| `README.md` | Operations documentation: prerequisites, install/run/typecheck, API surface, and an honest statement of what was and was not verified locally. |

## Honesty note on graceful degradation

- `GET /health` and `GET /feeds/:feedType` are designed to run with **no**
  PostgreSQL, Elasticsearch, or upstream service available.
- `POST /ingest/events` requires PostgreSQL and Elasticsearch to actually persist
  data; when they are down it returns a `200` summary with per-event errors
  rather than failing the request. It does **not** buffer/retry events.

## Schema-mirror deviations (intentional, documented)

- `events.user_id` is nullable in PostgreSQL (upstream SQLite is `NOT NULL`)
  because the ingest wire shape (`ClientEvent`) carries no `user_id` — the
  upstream derives it from the session, which the sidecar does not have.
- SQLite `username ... COLLATE NOCASE` uniqueness is mirrored as case-sensitive
  `@unique` (PostgreSQL would need the `citext` extension for NOCASE).
- Numeric `CHECK` ranges (e.g. `dwell_ms BETWEEN 750 AND 600000`) are enforced in
  the ingest route via `zod`, not in PostgreSQL (Prisma does not emit CHECKs).
- Foreign-key relations between the mirrored tables are not declared; the
  sidecar mirrors columns and only writes to `events`.
