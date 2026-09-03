import express from "express";
import type { Request, Response } from "express";
import { z } from "zod";
import { prisma, checkPostgres } from "./prisma.js";
import { es, checkElasticsearch } from "./es.js";

const PORT = Number(process.env.PORT ?? 4000);
const FASTAPI_URL = (process.env.FASTAPI_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

const app = express();
app.use(express.json({ limit: "1mb" }));

// ---------------------------------------------------------------------------
// Request validation (mirrors app/schemas.py `ClientEvent` / `EventBatch`)
// ---------------------------------------------------------------------------

const eventTypeSchema = z.enum([
  "impression",
  "click",
  "like",
  "favorite",
  "not_interested",
  "dwell",
  "share",
]);

const clientEventSchema = z.object({
  event_id: z.string().min(1).max(128),
  event_type: eventTypeSchema,
  request_id: z.string().min(1).max(128),
  item_id: z.string().min(1).max(128),
  position: z.number().int().min(0).max(9999),
  client_timestamp: z
    .string()
    .refine((value) => !Number.isNaN(Date.parse(value)), {
      message: "client_timestamp must be an ISO-8601 datetime",
    }),
  dwell_ms: z.number().int().min(750).max(600_000).nullable().optional(),
  visit_index: z.number().int().min(1).nullable().optional(),
  // user_id is NOT part of the upstream ClientEvent (it is derived from the
  // authenticated session there). The sidecar has no session, so it accepts an
  // optional user_id to populate the mirrored `events.user_id` column when the
  // caller knows it; otherwise the column is left NULL (see prisma/schema.prisma).
  user_id: z.string().min(1).max(128).optional(),
});

const ingestBodySchema = z.object({
  events: z.array(clientEventSchema).min(1).max(100),
});

// ---------------------------------------------------------------------------
// Connectivity probes
// ---------------------------------------------------------------------------

async function checkUpstream(): Promise<boolean> {
  try {
    const resp = await fetch(`${FASTAPI_URL}/api/v1/health`, {
      signal: AbortSignal.timeout(2_000),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

// Health: reports connectivity without throwing if any dependency is down.
app.get("/health", async (_req: Request, res: Response) => {
  const [pg, esHealthy, upstream] = await Promise.all([
    checkPostgres(),
    checkElasticsearch(),
    checkUpstream(),
  ]);
  res.status(200).json({ status: "ok", pg, es: esHealthy, upstream });
});

// Read-through proxy to the FastAPI feed endpoint. The upstream JSON is
// returned verbatim so each item's `source` / `score` / `model_version` fields
// are preserved. Returns a 502 JSON body if the upstream is unreachable.
app.get("/feeds/:feedType", async (req: Request, res: Response) => {
  const feedType = req.params.feedType;
  if (!feedType) {
    res.status(404).json({
      error: { code: "feed_not_found", message: "Unknown feed type" },
    });
    return;
  }

  const upstreamUrl = new URL(
    `${FASTAPI_URL}/api/v1/feeds/${encodeURIComponent(feedType)}`,
  );
  const { limit, cursor } = req.query;
  if (typeof limit === "string" && limit.length > 0) {
    upstreamUrl.searchParams.set("limit", limit);
  }
  if (typeof cursor === "string" && cursor.length > 0) {
    upstreamUrl.searchParams.set("cursor", cursor);
  }

  try {
    const upstream = await fetch(upstreamUrl, {
      headers: {
        cookie: req.headers.cookie ?? "",
        accept: "application/json",
      },
    });
    const payload: unknown = await upstream.json().catch(() => null);
    res.status(upstream.status).json(payload);
  } catch {
    res.status(502).json({
      error: {
        code: "upstream_unavailable",
        message: "Upstream FastAPI service is unavailable",
      },
    });
  }
});

// Ingest: writes each event to PostgreSQL (Prisma) and indexes it into
// Elasticsearch. Failures are collected per event; the route always responds
// 200 with a summary rather than crashing when PG/ES are unavailable.
app.post("/ingest/events", async (req: Request, res: Response) => {
  const parsed = ingestBodySchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(422).json({
      error: {
        code: "validation_error",
        message: "Request validation failed",
        details: parsed.error.issues,
      },
    });
    return;
  }

  const events = parsed.data.events;
  const receivedAt = new Date();
  let pgWritten = 0;
  let esIndexed = 0;
  const errors: Array<{ event_id: string; pg?: string; es?: string }> = [];

  for (const event of events) {
    // Upstream normalizes "favorite" -> "like" before persisting; mirror that
    // so the PostgreSQL enum only ever sees the CHECK-valid set.
    const normalizedType =
      event.event_type === "favorite" ? "like" : event.event_type;

    let pgOk = true;
    try {
      await prisma.event.create({
        data: {
          eventId: event.event_id,
          eventType: normalizedType,
          requestId: event.request_id,
          userId: event.user_id ?? null,
          itemId: event.item_id,
          position: event.position,
          clientTimestamp: new Date(event.client_timestamp),
          dwellMs: event.dwell_ms ?? null,
          visitIndex: event.visit_index ?? null,
          receivedAt,
        },
      });
    } catch {
      pgOk = false;
    }

    let esOk = true;
    try {
      await es.index({
        index: "microlens-events",
        id: event.event_id,
        document: {
          ...event,
          event_type: normalizedType,
          received_at: receivedAt.toISOString(),
        },
      });
    } catch {
      esOk = false;
    }

    if (pgOk) pgWritten += 1;
    if (esOk) esIndexed += 1;
    if (!pgOk || !esOk) {
      errors.push({
        event_id: event.event_id,
        ...(pgOk ? {} : { pg: "postgres_write_failed" }),
        ...(esOk ? {} : { es: "elasticsearch_index_failed" }),
      });
    }
  }

  res.status(200).json({
    received: events.length,
    pgWritten,
    esIndexed,
    errors,
  });
});

// Minimal index for discoverability.
app.get("/", (_req: Request, res: Response) => {
  res.status(200).json({
    name: "microlens-ts-interop",
    endpoints: ["/health", "/feeds/:feedType", "/ingest/events"],
  });
});

app.listen(PORT, () => {
  console.log(`ts-interop sidecar listening on http://localhost:${PORT}`);
});
