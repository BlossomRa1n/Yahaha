const API_BASE = "/api/v1";
const DASHBOARD_OVERVIEW_PATH = "/admin/dashboard/overview";
const DASHBOARD_EXPORT_PATH = "/admin/dashboard/export.csv";

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
  const { responseType = "json", ...fetchOptions } = options;
  const headers = new Headers(fetchOptions.headers || {});
  if (fetchOptions.body !== undefined && !(fetchOptions.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      credentials: "same-origin",
      ...fetchOptions,
      headers,
      body: fetchOptions.body === undefined || fetchOptions.body instanceof FormData
        ? fetchOptions.body
        : JSON.stringify(fetchOptions.body),
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
  if (responseType === "blob") {
    const disposition = response.headers.get("content-disposition") || "";
    const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1] || "dashboard.csv";
    return { blob: await response.blob(), filename };
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
  register: (username, password) => request("/auth/register", { method: "POST", body: { username, password } }),
  me: () => request("/auth/me"),
  logout: () => request("/auth/logout", { method: "POST" }),
  feed: (feedType, { limit = 12, cursor = null } = {}) => request(`/feeds/${feedType}${queryString({ limit, cursor })}`),
  sendEvents: (events) => request("/events/batch", { method: "POST", body: { events } }),
  profile: () => request("/me/profile"),
  myEvents: (limit = 50) => request(`/me/events${queryString({ limit })}`),
  item: (itemId) => request(`/items/${encodeURIComponent(itemId)}`),
  dashboard: ({ from = null, to = null } = {}) => request(`${DASHBOARD_OVERVIEW_PATH}${queryString({ from, to })}`),
  timeseries: (metric, { from = null, to = null } = {}) => request(`/admin/dashboard/timeseries${queryString({ metric, from, to })}`),
  exportDashboard: ({ from = null, to = null } = {}) => request(
    `${DASHBOARD_EXPORT_PATH}${queryString({ from, to })}`,
    { responseType: "blob" },
  ),
  requestDebug: (requestId) => request(`/admin/requests/${encodeURIComponent(requestId)}`),
  userDebug: (userId) => request(`/admin/users/${encodeURIComponent(userId)}/debug`),
  users: () => request("/admin/users"),
  items: ({ q = "", status = "", limit = 20, offset = 0 } = {}) => request(`/admin/items${queryString({ q, status, limit, offset })}`),
  updateItemStatus: (itemId, status, reason) => request(`/admin/items/${encodeURIComponent(itemId)}/status`, { method: "PATCH", body: { status, reason } }),
  updateItemStatusBatch: (body) => request("/admin/items/batch/status", { method: "PATCH", body }),
  createBoost: (body) => request("/admin/boosts", { method: "POST", body }),
  operations: () => request("/admin/operations"),
  models: () => request("/admin/models"),
  compareModels: (versions) => {
    const query = new URLSearchParams();
    versions.forEach((version) => query.append("versions", version));
    return request(`/admin/models/compare?${query.toString()}`);
  },
};

export function clientEventId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
