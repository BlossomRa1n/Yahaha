const API_BASE = "/api/v1";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "network_error", requestId = null, details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body !== undefined && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      credentials: "same-origin",
      ...options,
      headers,
      body: options.body === undefined || options.body instanceof FormData
        ? options.body
        : JSON.stringify(options.body),
    });
  } catch (error) {
    throw new ApiError("无法连接服务，请检查服务是否正在运行。", { details: String(error) });
  }

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : null;

  if (!response.ok) {
    const apiError = payload?.error || {};
    const error = new ApiError(apiError.message || `请求失败（HTTP ${response.status}）`, {
      status: response.status,
      code: apiError.code || "http_error",
      requestId: apiError.request_id || null,
      details: apiError.details || null,
    });
    if (response.status === 401) {
      window.dispatchEvent(new CustomEvent("auth:expired", { detail: error }));
    }
    throw error;
  }
  return payload;
}

function queryString(params) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
  });
  const encoded = query.toString();
  return encoded ? `?${encoded}` : "";
}

export const api = {
  login: (username, password) => request("/auth/login", { method: "POST", body: { username, password } }),
  me: () => request("/auth/me"),
  logout: () => request("/auth/logout", { method: "POST" }),
  feed: (feedType, { limit = 12, cursor = null } = {}) => request(`/feeds/${feedType}${queryString({ limit, cursor })}`),
  sendEvents: (events) => request("/events/batch", { method: "POST", body: { events } }),
  profile: () => request("/me/profile"),
  myEvents: (limit = 50) => request(`/me/events${queryString({ limit })}`),
  item: (itemId) => request(`/items/${encodeURIComponent(itemId)}`),
  dashboard: () => request("/admin/dashboard/overview"),
  requestDebug: (requestId) => request(`/admin/requests/${encodeURIComponent(requestId)}`),
  userDebug: (userId) => request(`/admin/users/${encodeURIComponent(userId)}/debug`),
  users: () => request("/admin/users"),
  items: ({ q = "", status = "", limit = 20, offset = 0 } = {}) => request(`/admin/items${queryString({ q, status, limit, offset })}`),
  updateItemStatus: (itemId, status, reason) => request(`/admin/items/${encodeURIComponent(itemId)}/status`, { method: "PATCH", body: { status, reason } }),
  createBoost: (body) => request("/admin/boosts", { method: "POST", body }),
  operations: () => request("/admin/operations"),
  models: () => request("/admin/models"),
};

export function clientEventId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
