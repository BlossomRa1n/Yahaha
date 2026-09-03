import { Client } from "@elastic/elasticsearch";

const node = process.env.ELASTICSEARCH_URL ?? "http://localhost:9200";

// The @elastic/elasticsearch v8 client does NOT open a connection until the
// first request, so this module can be imported (and /health + /feeds served)
// even when Elasticsearch is down.
export const es = new Client({
  node,
  requestTimeout: 2_000,
  maxRetries: 1,
  sniffOnStart: false,
});

/**
 * Connectivity probe for Elasticsearch. Returns false (never throws) when the
 * cluster is down or unreachable.
 */
export async function checkElasticsearch(): Promise<boolean> {
  try {
    return await es.ping();
  } catch {
    return false;
  }
}
