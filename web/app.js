import { ApiError, api, clientEventId } from "./api.js";

const state = {
  user: null,
  authMode: "login",
  currentView: "feed",
  feedType: "personalized",
  feedItems: [],
  feedCursor: null,
  feedHasMore: false,
  feedLoading: false,
  dashboardRange: { from: null, to: null },
  itemQuery: { q: "", status: "", limit: 20, offset: 0, total: 0 },
  selectedOperationItems: new Set(),
};

const IMPRESSION_RULE = Object.freeze({
  visibleRatio: 0.5,
  dwellMs: 750,
  batchSize: 25,
  maxRetries: 3,
});
const impressionState = {
  observer: null,
  timers: new Map(),
  pending: new Map(),
  reported: new Set(),
  activeDwells: new Map(),
  dwellTotals: new Map(),
  flushInFlight: false,
  retryTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat("zh-CN").format(Number(value));
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}

function datetimeLocalValue(date) {
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000)
    .toISOString()
    .slice(0, 19);
}

function initializeDashboardRange() {
  const end = new Date();
  const start = new Date(end.getTime() - 24 * 60 * 60 * 1000);
  $("#dashboard-range-from").value = datetimeLocalValue(start);
  $("#dashboard-range-to").value = datetimeLocalValue(end);
  state.dashboardRange = { from: start.toISOString(), to: end.toISOString() };
  $("#dashboard-range-applied").textContent = `已应用：${formatDate(start)} 至 ${formatDate(end)}`;
}

function setDashboardRangeBusy(busy) {
  $("#dashboard-range-apply").disabled = busy;
  $("#dashboard-export").disabled = busy;
  $("#dashboard-range-from").disabled = busy;
  $("#dashboard-range-to").disabled = busy;
}

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(1)} ms`;
}

function metricLabel(metric) {
  return {
    requests: "请求",
    exposures: "服务曝光",
    served_exposures: "服务曝光",
    impressions: "可见曝光",
    viewable_impressions: "可见曝光",
    clicks: "点击",
    likes: "喜欢",
  }[metric] || metric;
}

function formatBucketLabel(t, bucket) {
  const date = new Date(t);
  if (Number.isNaN(date.getTime())) return t;
  if (bucket === "hour") return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
  return date.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
}

function displayError(error) {
  if (!(error instanceof ApiError)) return "发生未知错误，请重试。";
  const prefix = error.status === 403 ? "权限不足：" : "";
  const request = error.requestId ? `（Request ID: ${error.requestId}）` : "";
  return `${prefix}${error.message}${request}`;
}

function setGlobalAlert(message, type = "info", timeout = 4500) {
  const alert = $("#global-alert");
  alert.textContent = message;
  alert.dataset.type = type;
  alert.hidden = false;
  window.clearTimeout(setGlobalAlert.timer);
  if (timeout) setGlobalAlert.timer = window.setTimeout(() => { alert.hidden = true; }, timeout);
}

function setPanelState(element, message, { type = "info", retry = null } = {}) {
  element.replaceChildren();
  element.dataset.type = type;
  element.hidden = false;
  element.append(createElement("span", "", message));
  if (retry) {
    const button = createElement("button", "state-action", "重试");
    button.type = "button";
    button.addEventListener("click", retry);
    element.append(button);
  }
}

function hidePanelState(element) {
  element.hidden = true;
  element.replaceChildren();
}

function isAdmin() {
  return state.user?.role === "admin";
}

function impressionStorageKey(name, userId) {
  return `${name}:${userId}`;
}

function clearPersistedImpressionState(userId) {
  if (!userId) return;
  try {
    sessionStorage.removeItem(impressionStorageKey("pendingImpressions", userId));
    sessionStorage.removeItem(impressionStorageKey("reportedImpressionIds", userId));
  } catch {
    // The server session remains authoritative when browser storage is unavailable.
  }
}

function resetFeedSession({ clearPersisted = false } = {}) {
  const userId = state.user?.id;
  stopImpressionObservation();
  window.clearTimeout(impressionState.retryTimer);
  impressionState.retryTimer = null;
  impressionState.pending.clear();
  impressionState.reported.clear();
  impressionState.activeDwells.clear();
  impressionState.dwellTotals.clear();
  if (clearPersisted) clearPersistedImpressionState(userId);
  state.feedItems = [];
  state.feedCursor = null;
  state.feedHasMore = false;
  state.feedLoading = false;
  $("#feed-list")?.replaceChildren();
  if ($("#feed-context")) $("#feed-context").hidden = true;
  if ($("#request-debug-id")) $("#request-debug-id").value = "";
}

function showLogin(message = "") {
  resetFeedSession({ clearPersisted: Boolean(state.user?.id) });
  state.user = null;
  $("#login-view").hidden = false;
  $$(".page-view").forEach((view) => { view.hidden = true; });
  $("#primary-nav").hidden = true;
  $("#session-controls").hidden = true;
  const error = $("#login-error");
  error.textContent = message;
  error.hidden = !message;
  $("#username").focus();
}

function setAuthMode(mode) {
  state.authMode = mode === "register" ? "register" : "login";
  const registering = state.authMode === "register";
  $("#login-title").textContent = registering ? "创建账号" : "登录工作台";
  $("#login-button").textContent = registering ? "注册并登录" : "登录";
  $("#password").autocomplete = registering ? "new-password" : "current-password";
  $("#login-mode").setAttribute("aria-selected", String(!registering));
  $("#register-mode").setAttribute("aria-selected", String(registering));
  $("#login-error").hidden = true;
}

function showAuthenticated(user) {
  const switchedUser = Boolean(state.user?.id && state.user.id !== user.id);
  resetFeedSession({ clearPersisted: switchedUser });
  state.user = user;
  restoreImpressionState(user.id);
  $("#login-view").hidden = true;
  $("#primary-nav").hidden = false;
  $("#session-controls").hidden = false;
  $("#session-user").textContent = `${user.username} · ${isAdmin() ? "管理员" : "普通用户"}`;
  $$(".admin-only").forEach((element) => { element.hidden = !isAdmin(); });
  showView("feed");
}

function showView(viewName) {
  if ((viewName === "dashboard" || viewName === "operations") && !isAdmin()) {
    setGlobalAlert("管理员权限不足，无法打开该页面。", "error");
    viewName = "feed";
  }
  state.currentView = viewName;
  $$(".page-view").forEach((view) => { view.hidden = view.id !== `${viewName}-view`; });
  $$(".nav-button").forEach((button) => button.classList.toggle("is-active", button.dataset.view === viewName));
  window.location.hash = viewName;
  if (viewName === "feed" && state.feedItems.length === 0) loadFeed(true);
  if (viewName === "dashboard") loadDashboard();
  if (viewName === "operations") loadOperationsView();
}

async function establishSession() {
  try {
    const response = await api.me();
    showAuthenticated(response.user);
    flushImpressions();
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 401) {
      setGlobalAlert(displayError(error), "error", 0);
    }
    showLogin();
  }
}

async function handleLogin(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("#login-button");
  const errorElement = $("#login-error");
  errorElement.hidden = true;
  button.disabled = true;
  button.textContent = state.authMode === "register" ? "注册中..." : "登录中...";
  try {
    const action = state.authMode === "register" ? api.register : api.login;
    const response = await action(form.username.value.trim(), form.password.value);
    form.password.value = "";
    showAuthenticated(response.user);
    flushImpressions();
  } catch (error) {
    errorElement.textContent = displayError(error);
    errorElement.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = state.authMode === "register" ? "注册并登录" : "登录";
  }
}

async function handleLogout() {
  try {
    await flushImpressions();
    await api.logout();
  } catch (error) {
    setGlobalAlert(displayError(error), "error");
  } finally {
    showLogin();
  }
}

function updateFeedContext(response) {
  $("#feed-context").hidden = false;
  $("#feed-request-id").textContent = response.request_id || "—";
  $("#feed-model-version").textContent = response.model_version || "—";
  $("#feed-profile-version").textContent = response.profile_version ?? "—";
  const fallback = $("#fallback-context");
  fallback.hidden = !response.fallback_reason;
  $("#feed-fallback").textContent = response.fallback_reason || "";
  $("#request-debug-id").value = response.request_id || "";
}

function scoreText(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return "无模型分数";
  return `分数 ${Number(score).toFixed(4)}`;
}

function buildFeedCard(item) {
  const card = createElement("article", "feed-card");
  card.dataset.itemId = String(item.item_id);
  card._feedItem = item;

  const coverButton = createElement("button", "cover-button");
  coverButton.type = "button";
  coverButton.title = "记录点击";
  coverButton.setAttribute("aria-label", `点击内容 ${item.title || item.item_id}`);
  const coverFallback = createElement("span", "cover-fallback", `ITEM ${item.item_id}`);
  if (item.cover_url) {
    const image = document.createElement("img");
    image.src = item.cover_url;
    image.alt = item.title ? `${item.title} 的封面` : `内容 ${item.item_id} 封面`;
    image.loading = "lazy";
    image.addEventListener("load", () => {
      if (image.naturalWidth <= 1 && image.naturalHeight <= 1) {
        image.hidden = true;
        coverFallback.hidden = false;
      }
    });
    image.addEventListener("error", () => {
      image.hidden = true;
      coverFallback.hidden = false;
    });
    coverFallback.hidden = true;
    coverButton.append(image, coverFallback);
  } else {
    coverButton.append(coverFallback);
  }
  coverButton.addEventListener("click", () => sendBehavior(item, "click", coverButton));

  const body = createElement("div", "feed-card-body");
  const provenance = createElement("div", "provenance-row");
  const source = createElement("span", "source-label", item.source || "unknown");
  const position = createElement("span", "muted", `#${Number(item.position) + 1}`);
  provenance.append(source, position);

  const titleButton = createElement("button", "feed-title-button", item.title || `内容 ${item.item_id}`);
  titleButton.type = "button";
  titleButton.title = "记录点击";
  titleButton.addEventListener("click", () => sendBehavior(item, "click", titleButton));

  const explanation = createElement("p", "explanation", item.explanation || "服务端未提供解释");
  const details = createElement("p", "score-line", `${scoreText(item.score)} · ${item.model_version || "无模型版本"}`);
  if (item.is_forced) {
    details.append(" · ", createElement("strong", "forced-label", "强推"));
  }

  const actions = createElement("div", "feed-actions");
  const like = createElement("button", "icon-text-button", "");
  like.type = "button";
  like.title = "喜欢此内容";
  like.append(createElement("span", "action-icon", "♡"), createElement("span", "", "喜欢"));
  like.addEventListener("click", () => sendBehavior(item, "like", like));
  const dislike = createElement("button", "icon-text-button", "");
  dislike.type = "button";
  dislike.title = "减少类似推荐";
  dislike.append(createElement("span", "action-icon", "⊘"), createElement("span", "", "不感兴趣"));
  dislike.addEventListener("click", () => sendBehavior(item, "not_interested", dislike));
  const share = createElement("button", "icon-text-button", "");
  share.type = "button";
  share.title = "分享此内容";
  share.append(createElement("span", "action-icon", "↗"), createElement("span", "", "分享"));
  share.addEventListener("click", () => shareItem(item, share));
  actions.append(like, dislike, share);

  body.append(provenance, titleButton, explanation, details, actions);
  card.append(coverButton, body);
  return card;
}

function renderFeed() {
  const list = $("#feed-list");
  list.replaceChildren(...state.feedItems.map(buildFeedCard));
  observeFeedCards();
  const loadMore = $("#load-more-button");
  loadMore.hidden = !state.feedHasMore || state.feedItems.length === 0;
  loadMore.disabled = state.feedLoading;
}

function persistedImpressionIds(userId) {
  try {
    return JSON.parse(
      sessionStorage.getItem(impressionStorageKey("reportedImpressionIds", userId)) || "[]",
    );
  } catch {
    return [];
  }
}

function persistImpressionState() {
  const userId = state.user?.id;
  if (!userId) return;
  try {
    sessionStorage.setItem(
      impressionStorageKey("reportedImpressionIds", userId),
      JSON.stringify([...impressionState.reported].slice(-1000)),
    );
    sessionStorage.setItem(
      impressionStorageKey("pendingImpressions", userId),
      JSON.stringify([...impressionState.pending.values()]),
    );
  } catch {
    // Storage is an optimization; server-side uniqueness remains authoritative.
  }
}

function restoreImpressionState(userId) {
  impressionState.pending.clear();
  impressionState.reported.clear();
  persistedImpressionIds(userId).forEach((eventId) => impressionState.reported.add(eventId));
  try {
    const pending = JSON.parse(
      sessionStorage.getItem(impressionStorageKey("pendingImpressions", userId)) || "[]",
    );
    pending.forEach((event) => {
      if (event?.event_id) impressionState.pending.set(event.event_id, event);
    });
  } catch {
    // Ignore malformed client-side retry state.
  }
}

function impressionEvent(item) {
  const eventId = "imp:" + item._requestId + ":" + item.item_id + ":" + Number(item.position);
  return {
    event_id: eventId,
    event_type: "impression",
    request_id: item._requestId,
    item_id: String(item.item_id),
    position: Number(item.position),
    client_timestamp: new Date().toISOString(),
    _attempts: 0,
  };
}

function queueImpression(item) {
  if (!item?._requestId) return;
  const event = impressionEvent(item);
  if (impressionState.reported.has(event.event_id) || impressionState.pending.has(event.event_id)) return;
  impressionState.pending.set(event.event_id, event);
  persistImpressionState();
  window.clearTimeout(impressionState.retryTimer);
  impressionState.retryTimer = window.setTimeout(flushImpressions, 100);
}

function startDwell(item, key) {
  if (!key || impressionState.activeDwells.has(key)) return;
  impressionState.activeDwells.set(key, { item, startedAt: Date.now() });
}

function endDwell(key) {
  const active = impressionState.activeDwells.get(key);
  if (!active) return;
  impressionState.activeDwells.delete(key);
  const total = Math.min(
    600000,
    Number(impressionState.dwellTotals.get(key) || 0) + Math.max(0, Date.now() - active.startedAt),
  );
  impressionState.dwellTotals.set(key, total);
  if (total < IMPRESSION_RULE.dwellMs) return;
  const item = active.item;
  const event = {
    event_id: `dwell:${clientEventId()}`,
    event_type: "dwell",
    request_id: item._requestId,
    item_id: String(item.item_id),
    position: Number(item.position),
    client_timestamp: new Date().toISOString(),
    dwell_ms: Math.round(total),
    _attempts: 0,
  };
  impressionState.pending.set(event.event_id, event);
  persistImpressionState();
  window.clearTimeout(impressionState.retryTimer);
  impressionState.retryTimer = window.setTimeout(flushImpressions, 100);
}

function endAllDwells() {
  [...impressionState.activeDwells.keys()].forEach(endDwell);
}

async function flushImpressions() {
  if (!state.user || impressionState.flushInFlight || impressionState.pending.size === 0) return;
  const batch = [...impressionState.pending.values()]
    .filter((event) => event._attempts < IMPRESSION_RULE.maxRetries)
    .slice(0, IMPRESSION_RULE.batchSize);
  if (!batch.length) return;
  impressionState.flushInFlight = true;
  try {
    await api.sendEvents(batch.map(({ _attempts, ...event }) => event));
    batch.forEach((event) => {
      impressionState.pending.delete(event.event_id);
      impressionState.reported.add(event.event_id);
    });
    persistImpressionState();
    if (impressionState.pending.size) {
      impressionState.retryTimer = window.setTimeout(flushImpressions, 100);
    }
  } catch {
    batch.forEach((event) => {
      event._attempts += 1;
      if (event._attempts >= IMPRESSION_RULE.maxRetries) {
        impressionState.pending.delete(event.event_id);
      }
    });
    persistImpressionState();
    const retryable = [...impressionState.pending.values()]
      .filter((event) => event._attempts < IMPRESSION_RULE.maxRetries);
    if (retryable.length) {
      const attempts = Math.max(...retryable.map((event) => event._attempts));
      impressionState.retryTimer = window.setTimeout(flushImpressions, 500 * (2 ** attempts));
    }
  } finally {
    impressionState.flushInFlight = false;
  }
}

function stopImpressionObservation() {
  impressionState.observer?.disconnect();
  impressionState.timers.forEach((timer) => window.clearTimeout(timer));
  impressionState.timers.clear();
  endAllDwells();
}

function observeFeedCards() {
  stopImpressionObservation();
  if (!("IntersectionObserver" in window) || document.visibilityState !== "visible") return;
  impressionState.observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const card = entry.target;
      const item = card._feedItem;
      const key = item ? impressionEvent(item).event_id : "";
      if (document.visibilityState !== "visible") {
        if (key && impressionState.timers.has(key)) {
          window.clearTimeout(impressionState.timers.get(key));
          impressionState.timers.delete(key);
        }
        if (key) endDwell(key);
        return;
      }
      if (entry.isIntersecting && entry.intersectionRatio >= IMPRESSION_RULE.visibleRatio) {
        if (key && impressionState.reported.has(key)) {
          startDwell(item, key);
          return;
        }
        if (!key || impressionState.timers.has(key)) return;
        const timer = window.setTimeout(() => {
          impressionState.timers.delete(key);
          if (document.visibilityState !== "visible" || !card.isConnected) return;
          queueImpression(item);
          startDwell(item, key);
        }, IMPRESSION_RULE.dwellMs);
        impressionState.timers.set(key, timer);
      } else if (key && impressionState.timers.has(key)) {
        window.clearTimeout(impressionState.timers.get(key));
        impressionState.timers.delete(key);
      } else if (key) {
        endDwell(key);
      }
    });
  }, { threshold: [IMPRESSION_RULE.visibleRatio] });
  $$(".feed-card").forEach((card) => impressionState.observer.observe(card));
}

function beaconPendingImpressions() {
  const batch = [...impressionState.pending.values()]
    .filter((event) => event._attempts < IMPRESSION_RULE.maxRetries)
    .slice(0, IMPRESSION_RULE.batchSize);
  if (!batch.length || !navigator.sendBeacon) return;
  const events = batch.map(({ _attempts, ...event }) => event);
  const accepted = navigator.sendBeacon(
    "/api/v1/events/batch",
    new Blob([JSON.stringify({ events })], { type: "application/json" }),
  );
  if (accepted) {
    batch.forEach((event) => {
      impressionState.pending.delete(event.event_id);
      impressionState.reported.add(event.event_id);
    });
    persistImpressionState();
  }
}

async function loadFeed(reset = false) {
  if (state.feedLoading) return;
  state.feedLoading = true;
  $$("#feed-tabs [role='tab']").forEach((tab) => { tab.disabled = true; });
  const panel = $("#feed-state");
  setPanelState(panel, reset ? "正在生成推荐并记录服务曝光..." : "正在加载更多推荐...");
  $("#load-more-button").disabled = true;
  if (reset) {
    state.feedCursor = null;
    state.feedHasMore = false;
    state.feedItems = [];
    renderFeed();
  }
  try {
    const response = await api.feed(state.feedType, { limit: 12, cursor: reset ? null : state.feedCursor });
    const existing = new Set(state.feedItems.map((item) => String(item.item_id)));
    const newItems = (response.items || [])
      .filter((item) => !existing.has(String(item.item_id)))
      .map((item) => ({ ...item, _requestId: response.request_id, _feedType: response.feed_type }));
    state.feedItems.push(...newItems);
    state.feedCursor = response.next_cursor || null;
    state.feedHasMore = Boolean(response.has_more && response.next_cursor);
    updateFeedContext(response);
    renderFeed();
    if (state.feedItems.length === 0) {
      setPanelState(panel, "当前没有可展示内容。可能是候选耗尽或内容已下线。", { type: "empty", retry: () => loadFeed(true) });
    } else {
      hidePanelState(panel);
    }
  } catch (error) {
    const cursorRefreshRequired = error instanceof ApiError
      && (error.code === "cursor_expired" || error.code === "invalid_cursor");
    if (cursorRefreshRequired) {
      state.feedCursor = null;
      state.feedHasMore = false;
    }
    setPanelState(
      panel,
      cursorRefreshRequired ? "本次 Feed 分页已失效，请刷新以获取最新推荐。" : displayError(error),
      { type: "error", retry: () => loadFeed(cursorRefreshRequired || reset) },
    );
  } finally {
    state.feedLoading = false;
    $$("#feed-tabs [role='tab']").forEach((tab) => { tab.disabled = false; });
    $("#load-more-button").disabled = false;
  }
}

async function sendBehavior(item, eventType, button) {
  if (button.disabled) return;
  button.disabled = true;
  const originalTitle = button.title;
  button.title = "正在上报";
  try {
    const response = await api.sendEvents([{
      event_id: clientEventId(),
      event_type: eventType,
      request_id: item._requestId,
      item_id: String(item.item_id),
      position: Number(item.position),
      client_timestamp: new Date().toISOString(),
    }]);
    if (response.profile_version !== undefined) $("#feed-profile-version").textContent = response.profile_version;
    if (eventType === "not_interested") {
      state.feedItems = state.feedItems.filter((candidate) => String(candidate.item_id) !== String(item.item_id));
      renderFeed();
    }
    if (eventType === "like") button.classList.add("is-selected");
    const duplicate = Number(response.duplicate_count ?? response.duplicates ?? 0) > 0;
    setGlobalAlert(duplicate ? "该行为已记录，无需重复写入。" : "行为已记录，下一次个性化请求将读取更新后的画像。", "success");
  } catch (error) {
    setGlobalAlert(displayError(error), "error", 0);
  } finally {
    button.disabled = false;
    button.title = originalTitle;
  }
}

async function shareItem(item, button) {
  if (button.disabled) return;
  const shareUrl = new URL(window.location.href);
  shareUrl.hash = "feed";
  shareUrl.searchParams.set("item", String(item.item_id));
  try {
    if (navigator.share) {
      await navigator.share({ title: item.title || `内容 ${item.item_id}`, url: shareUrl.href });
    } else {
      await navigator.clipboard.writeText(shareUrl.href);
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      setGlobalAlert("已取消分享。", "info");
      return;
    }
    setGlobalAlert("分享失败，未记录分享行为。", "error");
    return;
  }
  await sendBehavior(item, "share", button);
}

async function loadProfile() {
  const dialog = $("#profile-dialog");
  if (!dialog.open) dialog.showModal();
  const panel = $("#profile-state");
  setPanelState(panel, "正在加载画像和最近行为...");
  try {
    const [profile, eventsResponse] = await Promise.all([api.profile(), api.myEvents(50)]);
    const summary = profile.summary || {};
    const details = [
      ["画像版本", profile.version],
      ["更新时间", formatDate(profile.updated_at)],
      ["可见曝光", formatNumber(summary.impressions)],
      ["点击", formatNumber(summary.clicks)],
      ["喜欢", formatNumber(summary.likes)],
      ["停留时长", `${formatNumber(summary.dwell_ms)} ms`],
      ["分享", formatNumber(summary.shares)],
      ["重复访问", formatNumber(summary.revisits)],
      ["不感兴趣", formatNumber(summary.not_interested)],
      ["正向内容", (profile.positive_items || []).map((item) => `${item.item_id} (${item.weight})`).join(", ") || "暂无"],
      ["负向内容", (profile.negative_items || []).map((item) => `${item.item_id} (${item.weight})`).join(", ") || "暂无"],
    ];
    const list = $("#profile-summary");
    list.replaceChildren();
    details.forEach(([label, value]) => {
      const wrapper = createElement("div");
      wrapper.append(createElement("dt", "", label), createElement("dd", "", value ?? "—"));
      list.append(wrapper);
    });
    renderRows($("#profile-events-body"), eventsResponse.events || [], (event) => [
      formatDate(event.received_at || event.client_timestamp),
      event.event_type,
      event.item_id,
      `${event.feed_type || "unknown"} / ${event.source || "unknown"}`,
      event.request_id,
    ], 5, "暂无行为记录");
    hidePanelState(panel);
  } catch (error) {
    setPanelState(panel, displayError(error), { type: "error", retry: loadProfile });
  }
}

function renderRows(body, rows, cellsForRow, columnCount, emptyMessage) {
  body.replaceChildren();
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = createElement("td", "table-empty", emptyMessage);
    cell.colSpan = columnCount;
    row.append(cell);
    body.append(row);
    return;
  }
  rows.forEach((item) => {
    const row = document.createElement("tr");
    cellsForRow(item).forEach((value) => {
      const cell = document.createElement("td");
      if (value instanceof Node) cell.append(value);
      else cell.textContent = value === null || value === undefined ? "—" : String(value);
      row.append(cell);
    });
    body.append(row);
  });
}

function renderDashboard(overview) {
  const metrics = [
    ["用户", overview.users], ["活跃用户", overview.active_users], ["请求", overview.requests],
    ["服务曝光", overview.served_exposures ?? overview.exposures],
    ["可见曝光", overview.viewable_impressions ?? overview.impressions],
    ["点击", overview.clicks],
    ["服务 CTR", formatPercent(overview.served_ctr ?? overview.ctr)],
    ["可见 CTR", formatPercent(overview.viewable_ctr)],
    ["喜欢", overview.likes], ["分享", overview.shares], ["重复访问", overview.revisits],
    ["平均停留", formatMs(overview.dwell?.average)], ["停留 P95", formatMs(overview.dwell?.p95)],
    ["下线内容", overview.offline_items], ["生效强推", overview.active_boosts],
    ["当前模型", overview.current_model_version || "—"],
  ];
  const latency = overview.latency || {};
  metrics.push(
    ["延迟 P50", formatMs(latency.p50)],
    ["延迟 P95", formatMs(latency.p95)],
    ["延迟 P99", formatMs(latency.p99)],
  );
  const grid = $("#metric-grid");
  grid.replaceChildren();
  metrics.forEach(([label, value]) => {
    const metric = createElement("div", "metric-item");
    metric.append(createElement("span", "metric-label", label), createElement("strong", "metric-value", typeof value === "number" ? formatNumber(value) : value ?? "—"));
    grid.append(metric);
  });
  $("#dashboard-updated").textContent = overview.range
    ? `${formatDate(overview.range.from)} 至 ${formatDate(overview.range.to)}`
    : "当前聚合范围";

  const exposureTotal = Number(overview.served_exposures ?? overview.exposures ?? 0);
  renderRows($("#feed-breakdown-body"), overview.feed_breakdown || [], (feed) => [
    feed.feed_type,
    formatNumber(feed.requests),
    formatNumber(feed.served_exposures ?? feed.exposures),
    formatNumber(feed.viewable_impressions ?? feed.impressions),
    formatNumber(feed.clicks),
    formatNumber(feed.likes),
    formatNumber(feed.not_interested),
    formatNumber(feed.shares),
    formatNumber(feed.revisits),
    formatMs(feed.average_dwell_ms),
    formatPercent(feed.served_ctr ?? feed.ctr),
    formatPercent(feed.viewable_ctr),
    formatPercent(feed.share ?? (exposureTotal ? Number(feed.exposures) / exposureTotal : 0)),
  ], 13, "暂无信息流请求");

  renderRows($("#top-items-body"), overview.top_items || [], (item) => [
    item.title ? `${item.title} (${item.item_id})` : item.item_id,
    formatNumber(item.served_exposures ?? item.exposures),
    formatNumber(item.viewable_impressions ?? item.impressions),
    formatNumber(item.clicks),
    formatNumber(item.likes),
    formatPercent(item.served_ctr ?? item.ctr),
    formatPercent(item.viewable_ctr),
  ], 7, "暂无热门内容数据");

  renderRows($("#source-breakdown-body"), overview.candidate_sources || [], (source) => [
    source.source,
    formatNumber(source.requests),
    formatNumber(source.served_exposures),
    formatPercent(source.share),
  ], 4, "暂无候选来源数据");
}

function renderTimeseries(payload) {
  const chart = $("#timeseries-chart");
  chart.replaceChildren();
  const points = payload?.points || [];
  if (!points.length) {
    chart.append(createElement("div", "empty", "该时间范围内暂无数据。"));
    return;
  }
  const width = 600;
  const height = 200;
  const padL = 38;
  const padR = 12;
  const padT = 16;
  const padB = 26;
  const maxValue = Math.max(1, ...points.map((p) => p.value));
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("class", "timeseries-svg");

  const x = (i) => padL + (points.length === 1 ? 0 : (i * (width - padL - padR)) / (points.length - 1));
  const y = (v) => padT + (1 - v / maxValue) * (height - padT - padB);

  for (let s = 0; s <= 4; s += 1) {
    const value = Math.round((maxValue * s) / 4);
    const gy = padT + (s / 4) * (height - padT - padB);
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", padL); line.setAttribute("x2", width - padR);
    line.setAttribute("y1", gy); line.setAttribute("y2", gy);
    line.setAttribute("class", "timeseries-grid");
    svg.append(line);
    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", padL - 6); label.setAttribute("y", gy + 3);
    label.setAttribute("class", "timeseries-label");
    label.setAttribute("text-anchor", "end");
    label.textContent = String(value);
    svg.append(label);
  }

  const linePath = document.createElementNS(svgNS, "polyline");
  linePath.setAttribute("points", points.map((p, i) => `${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" "));
  linePath.setAttribute("class", "timeseries-line");
  svg.append(linePath);

  points.forEach((p, i) => {
    const dot = document.createElementNS(svgNS, "circle");
    dot.setAttribute("cx", x(i).toFixed(1));
    dot.setAttribute("cy", y(p.value).toFixed(1));
    dot.setAttribute("r", "2.5");
    dot.setAttribute("class", "timeseries-dot");
    svg.append(dot);
  });

  const xIndices = points.length === 1 ? [0] : [0, points.length - 1];
  xIndices.forEach((i) => {
    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", x(i).toFixed(1));
    label.setAttribute("y", height - 6);
    label.setAttribute("class", "timeseries-label");
    label.setAttribute("text-anchor", i === 0 ? "start" : "end");
    label.textContent = formatBucketLabel(points[i].t, payload.bucket);
    svg.append(label);
  });

  chart.append(svg);
  chart.append(createElement("div", "muted", `${metricLabel(payload.metric)} · ${payload.bucket === "hour" ? "按小时" : "按天"} · ${formatDate(payload.range?.from)} 至 ${formatDate(payload.range?.to)}`));
}

async function loadTimeseries() {
  const chart = $("#timeseries-chart");
  const metric = $("#timeseries-metric")?.value || "requests";
  chart.replaceChildren(createElement("div", "muted", "正在加载趋势..."));
  try {
    renderTimeseries(await api.timeseries(metric, state.dashboardRange));
  } catch (error) {
    chart.replaceChildren(createElement("div", "empty", displayError(error)));
  }
}

function renderModels(response) {
  const metricSummary = (metrics) => {
    const svd = metrics?.test?.models?.svd;
    if (svd) {
      return [
        ["Recall@10", svd["recall@10"]],
        ["NDCG@10", svd["ndcg@10"]],
        ["HitRate@10", svd["hitrate@10"]],
      ].map(([label, value]) => `${label} ${Number(value).toFixed(4)}`).join(" · ");
    }
    return Object.entries(metrics || {})
      .filter(([, value]) => typeof value === "number")
      .map(([key, value]) => `${key}: ${Number(value).toFixed(4)}`)
      .join(" · ") || "—";
  };
  renderRows($("#models-body"), response.models || [], (model) => {
    const selector = document.createElement("input");
    selector.type = "checkbox";
    selector.value = model.model_version;
    selector.className = "model-compare-selector";
    selector.setAttribute("aria-label", `选择模型 ${model.model_version}`);
    selector.addEventListener("change", updateModelCompareButton);
    return [
      selector,
      model.model_version,
      model.data_version || "—",
      model.algorithm,
      model.is_current || model.model_version === response.current_model_version
        ? "当前线上"
        : `${model.training_status || "—"} / ${model.publish_status || model.status}`,
      formatDate(model.activated_at || model.created_at),
      metricSummary(model.metrics),
    ];
  }, 7, "暂无已登记模型");
  updateModelCompareButton();
}

function selectedModelVersions() {
  return Array.from(
    document.querySelectorAll(".model-compare-selector:checked"),
    (node) => node.value,
  );
}

function updateModelCompareButton() {
  const button = $("#compare-models-button");
  if (!button) return;
  const count = selectedModelVersions().length;
  button.disabled = count < 2 || count > 10;
  button.textContent = count ? `比较所选版本（${count}）` : "比较所选版本";
}

function comparisonMetric(model, prefix) {
  const entry = Object.entries(model.metrics || {}).find(([key]) => key.startsWith(prefix));
  if (!entry) return "—";
  const [key, value] = entry;
  const delta = model.deltas_from_baseline?.[key];
  const deltaText = delta === undefined || delta === null
    ? ""
    : ` (${delta >= 0 ? "+" : ""}${delta.toFixed(4)})`;
  return `${key} ${Number(value).toFixed(4)}${deltaText}`;
}

function renderModelComparison(payload) {
  const panel = $("#model-comparison-state");
  const container = $("#model-comparison");
  panel.hidden = false;
  panel.dataset.type = payload.protocol_compatible ? "success" : "empty";
  const protocol = payload.evaluation_protocol || {};
  panel.textContent = payload.protocol_compatible
    ? `基准 ${payload.baseline_version} · K=${protocol.k ?? "—"} · ${protocol.cohort_aggregation || "评估口径已匹配"}`
    : payload.compatibility_reason;
  container.hidden = false;
  renderRows($("#model-comparison-body"), payload.models || [], (model) => [
    `${model.model_version}${model.is_current ? " · 当前线上" : ""}`,
    `${model.data_version || "—"}\n${model.training_window?.start || "—"} → ${model.training_window?.end || "—"}`,
    `${formatNumber(model.sample_count || 0)} / ${formatNumber(model.event_count || 0)}`,
    `${model.training_status} / ${model.publish_status}`,
    comparisonMetric(model, "recall@"),
    comparisonMetric(model, "ndcg@"),
    comparisonMetric(model, "hitrate@"),
  ], 7, "没有可比较的模型版本");
}

async function compareSelectedModels() {
  const versions = selectedModelVersions();
  const panel = $("#model-comparison-state");
  panel.hidden = false;
  setPanelState(panel, "正在比较模型版本...");
  try {
    renderModelComparison(await api.compareModels(versions));
  } catch (error) {
    setPanelState(panel, displayError(error), { type: "error" });
  }
}

async function loadDashboard() {
  const panel = $("#dashboard-state");
  setDashboardRangeBusy(true);
  setPanelState(panel, "正在聚合 Dashboard 指标...");
  const [overviewResult, modelsResult] = await Promise.allSettled([
    api.dashboard(state.dashboardRange),
    api.models(),
  ]);
  if (overviewResult.status === "fulfilled") {
    renderDashboard(overviewResult.value);
    $("#dashboard-range-applied").textContent =
      `已应用：${formatDate(overviewResult.value.range?.from)} 至 ${formatDate(overviewResult.value.range?.to)}`;
    hidePanelState(panel);
  } else {
    setPanelState(panel, displayError(overviewResult.reason), { type: "error", retry: loadDashboard });
  }
  if (modelsResult.status === "fulfilled") {
    renderModels(modelsResult.value);
  } else {
    renderRows($("#models-body"), [], () => [], 6, displayError(modelsResult.reason));
  }
  await loadTimeseries();
  setDashboardRangeBusy(false);
}

async function applyDashboardRange(event) {
  event.preventDefault();
  const start = new Date(event.currentTarget.from.value);
  const end = new Date(event.currentTarget.to.value);
  if (
    Number.isNaN(start.getTime())
    || Number.isNaN(end.getTime())
    || start >= end
    || end - start > 366 * 24 * 60 * 60 * 1000
  ) {
    setGlobalAlert("时间范围无效：开始时间必须早于结束时间，且范围不能超过 366 天。", "error");
    return;
  }
  state.dashboardRange = { from: start.toISOString(), to: end.toISOString() };
  await loadDashboard();
}

async function exportDashboardCsv() {
  const button = $("#dashboard-export");
  button.disabled = true;
  try {
    const { blob, filename } = await api.exportDashboard(state.dashboardRange);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    setGlobalAlert(displayError(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function runDiagnostic(kind, id) {
  const output = $("#diagnostic-result");
  output.textContent = "正在查询...";
  try {
    const response = kind === "request" ? await api.requestDebug(id) : await api.userDebug(id);
    output.textContent = JSON.stringify(response, null, 2);
  } catch (error) {
    output.textContent = displayError(error);
  }
}

function statusBadge(status) {
  return createElement("span", `status-badge is-${status}`, status === "online" ? "在线" : "已下线");
}

function itemActions(item) {
  const wrapper = createElement("div", "table-actions");
  const button = createElement("button", item.status === "online" ? "danger-text-button" : "secondary-compact-button", item.status === "online" ? "下线" : "恢复");
  button.type = "button";
  button.title = item.status === "online" ? "下线内容" : "恢复内容";
  button.addEventListener("click", () => openStatusDialog(item));
  wrapper.append(button);
  return wrapper;
}

function renderItems(response) {
  const items = response.items || [];
  state.itemQuery.total = Number(response.total || 0);
  state.itemQuery.limit = Number(response.limit || state.itemQuery.limit);
  state.itemQuery.offset = Number(response.offset || 0);
  renderRows($("#items-body"), items, (item) => {
    const selector = document.createElement("input");
    selector.type = "checkbox";
    selector.value = item.item_id;
    selector.checked = state.selectedOperationItems.has(item.item_id);
    selector.setAttribute("aria-label", `选择内容 ${item.item_id}`);
    selector.addEventListener("change", () => {
      if (selector.checked) state.selectedOperationItems.add(item.item_id);
      else state.selectedOperationItems.delete(item.item_id);
      updateBatchControls();
    });
    return [
    selector,
    item.item_id,
    item.title,
    formatNumber(item.popularity_score ?? item.views),
    statusBadge(item.status),
    formatDate(item.updated_at),
    itemActions(item),
  ];
  }, 7, "没有符合条件的内容");
  const page = Math.floor(state.itemQuery.offset / state.itemQuery.limit) + 1;
  $("#items-page-label").textContent = `第 ${page} 页 · 共 ${formatNumber(state.itemQuery.total)} 条`;
  $("#items-previous").disabled = state.itemQuery.offset <= 0;
  $("#items-next").disabled = state.itemQuery.offset + items.length >= state.itemQuery.total;
  updateBatchControls();
}

function updateBatchControls() {
  const count = state.selectedOperationItems.size;
  const button = $("#batch-action-button");
  button.disabled = count === 0;
  button.textContent = count ? `批量操作（${count}）` : "批量操作";
  const selectAll = $("#select-page-items");
  const visible = $$("#items-body input[type='checkbox']");
  selectAll.checked = visible.length > 0 && visible.every((input) => input.checked);
  selectAll.indeterminate = visible.some((input) => input.checked) && !selectAll.checked;
}

async function loadItems() {
  const panel = $("#operations-state");
  setPanelState(panel, "正在加载内容状态...");
  try {
    const response = await api.items(state.itemQuery);
    renderItems(response);
    hidePanelState(panel);
  } catch (error) {
    setPanelState(panel, displayError(error), { type: "error", retry: loadItems });
  }
}

async function loadAudit() {
  try {
    const response = await api.operations();
    renderRows($("#operations-body"), response.operations || [], (operation) => [
      formatDate(operation.created_at),
      operation.admin_username || operation.admin_user_id,
      operation.action,
      operation.item_id || operation.target_id,
      operation.batch_id || "—",
      `${auditValue(operation.before)} → ${auditValue(operation.after)}`,
      operation.reason,
    ], 7, "暂无操作记录");
  } catch (error) {
    renderRows($("#operations-body"), [], () => [], 7, displayError(error));
  }
}

function openBatchDialog() {
  if (!state.selectedOperationItems.size) return;
  const form = $("#batch-status-form");
  form.reset();
  form.dataset.idempotencyKey = `batch-${clientEventId()}`;
  $("#batch-selection-count").textContent = `已选择 ${state.selectedOperationItems.size} 个内容`;
  $("#batch-status-form [data-form-error]").hidden = true;
  $("#batch-status-dialog").showModal();
}

async function handleBatchStatusSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const submit = $("button[type='submit']", form);
  const errorElement = $("[data-form-error]", form);
  submit.disabled = true;
  errorElement.hidden = true;
  try {
    const result = await api.updateItemStatusBatch({
      item_ids: [...state.selectedOperationItems],
      status: form.status.value,
      reason: form.reason.value.trim(),
      idempotency_key: form.dataset.idempotencyKey,
    });
    $("#batch-status-dialog").close();
    state.selectedOperationItems.clear();
    setGlobalAlert(
      `批次 ${result.batch_id}：成功 ${result.success_count}，失败 ${result.failure_count}，实际变更 ${result.changed_count}。`,
      "success",
    );
    await Promise.all([loadItems(), loadAudit()]);
  } catch (error) {
    errorElement.textContent = displayError(error);
    errorElement.hidden = false;
  } finally {
    submit.disabled = false;
  }
}

function loadOperationsView() {
  loadItems();
  loadAudit();
  loadAdminUsers();
}

function auditValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function openStatusDialog(item) {
  const nextStatus = item.status === "online" ? "offline" : "online";
  const dialog = $("#status-dialog");
  const form = $("#status-form");
  form.reset();
  form.item_id.value = item.item_id;
  form.status.value = nextStatus;
  $("#status-dialog-title").textContent = nextStatus === "offline" ? "下线内容" : "恢复内容";
  $("#status-dialog-copy").textContent = `${item.title || item.item_id}（${item.item_id}）`;
  $("#status-form .danger-button").textContent = nextStatus === "offline" ? "确认下线" : "确认恢复";
  $("#status-form .danger-button").className = nextStatus === "offline" ? "danger-button" : "primary-button";
  $("#status-form [data-form-error]").hidden = true;
  dialog.showModal();
}

async function handleStatusSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const submit = $("button[type='submit']", form);
  const errorElement = $("[data-form-error]", form);
  submit.disabled = true;
  errorElement.hidden = true;
  try {
    await api.updateItemStatus(form.item_id.value, form.status.value, form.reason.value.trim());
    $("#status-dialog").close();
    setGlobalAlert(form.status.value === "offline" ? "内容已由服务端下线。" : "内容已恢复为在线状态。", "success");
    await Promise.all([loadItems(), loadAudit()]);
  } catch (error) {
    errorElement.textContent = displayError(error);
    errorElement.hidden = false;
  } finally {
    submit.disabled = false;
  }
}

function toIso(localValue) {
  if (!localValue) return null;
  const date = new Date(localValue);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

async function loadAdminUsers() {
  const select = $("#boost-form [name='user_ids']");
  if (select.dataset.loaded === "true") return;
  const loadingOption = createElement("option", "", "正在加载用户...");
  loadingOption.value = "";
  loadingOption.disabled = true;
  select.replaceChildren(loadingOption);
  select.disabled = true;
  try {
    const response = await api.users();
    const users = (response.users || []).filter((user) => user.role === "user");
    select.replaceChildren();
    users.forEach((user) => {
      const option = createElement("option", "", `${user.username} · ${user.id}`);
      option.value = user.id;
      select.append(option);
    });
    if (!users.length) {
      const emptyOption = createElement("option", "", "没有可投放的普通用户");
      emptyOption.value = "";
      emptyOption.disabled = true;
      select.append(emptyOption);
    }
    select.dataset.loaded = "true";
  } catch (error) {
    const errorOption = createElement("option", "", displayError(error));
    errorOption.value = "";
    errorOption.disabled = true;
    select.replaceChildren(errorOption);
    select.dataset.loaded = "false";
  } finally {
    select.disabled = false;
  }
}

async function handleBoostSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const errorElement = $("[data-form-error]", form);
  const submit = $("button[type='submit']", form);
  const feedTypes = $$(`input[name="feed_types"]:checked`, form).map((input) => input.value);
  const userIds = [...form.user_ids.selectedOptions].map((option) => option.value).filter(Boolean);
  const startsAt = toIso(form.starts_at.value);
  const endsAt = toIso(form.ends_at.value);
  let validationError = "";
  if (!feedTypes.length) validationError = "至少选择一路信息流。";
  else if (form.audience.value === "users" && !userIds.length) validationError = "指定用户投放必须填写用户 ID。";
  else if (!startsAt || !endsAt || new Date(endsAt) <= new Date(startsAt)) validationError = "结束时间必须晚于开始时间。";
  if (validationError) {
    errorElement.textContent = validationError;
    errorElement.hidden = false;
    return;
  }
  submit.disabled = true;
  errorElement.hidden = true;
  try {
    await api.createBoost({
      item_id: form.item_id.value.trim(),
      audience: form.audience.value,
      user_ids: form.audience.value === "users" ? userIds : [],
      feed_types: feedTypes,
      position: Number(form.position.value),
      priority: Number(form.priority.value),
      starts_at: startsAt,
      ends_at: endsAt,
      reason: form.reason.value.trim(),
    });
    $("#boost-dialog").close();
    form.reset();
    setGlobalAlert("强推规则已由服务端创建。", "success");
    await loadAudit();
  } catch (error) {
    errorElement.textContent = displayError(error);
    errorElement.hidden = false;
  } finally {
    submit.disabled = false;
  }
}

function setDefaultBoostTimes() {
  const form = $("#boost-form");
  const now = new Date();
  const later = new Date(now.getTime() + 24 * 60 * 60 * 1000);
  const localValue = (date) => new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  form.starts_at.value = localValue(now);
  form.ends_at.value = localValue(later);
}

function bindEvents() {
  $("#login-form").addEventListener("submit", handleLogin);
  $("#login-mode").addEventListener("click", () => setAuthMode("login"));
  $("#register-mode").addEventListener("click", () => setAuthMode("register"));
  $("#logout-button").addEventListener("click", handleLogout);
  $$(".nav-button").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
  $$("#feed-tabs [role='tab']").forEach((button) => button.addEventListener("click", () => {
    state.feedType = button.dataset.feed;
    $$("#feed-tabs [role='tab']").forEach((tab) => {
      const selected = tab === button;
      tab.setAttribute("aria-selected", String(selected));
      if (selected) $("#feed-panel").setAttribute("aria-labelledby", tab.id);
    });
    loadFeed(true);
  }));
  $("#load-more-button").addEventListener("click", () => loadFeed(false));
  $("#profile-button").addEventListener("click", loadProfile);
  $("#dashboard-refresh").addEventListener("click", loadDashboard);
  $("#dashboard-range-form").addEventListener("submit", applyDashboardRange);
  $("#dashboard-export").addEventListener("click", exportDashboardCsv);
  $("#compare-models-button").addEventListener("click", compareSelectedModels);
  $("#timeseries-metric").addEventListener("change", loadTimeseries);
  $("#request-debug-form").addEventListener("submit", (event) => { event.preventDefault(); runDiagnostic("request", event.currentTarget.request_id.value.trim()); });
  $("#user-debug-form").addEventListener("submit", (event) => { event.preventDefault(); runDiagnostic("user", event.currentTarget.user_id.value.trim()); });
  $("#item-search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    state.itemQuery.q = event.currentTarget.q.value.trim();
    state.itemQuery.status = event.currentTarget.status.value;
    state.itemQuery.offset = 0;
    loadItems();
  });
  $("#items-previous").addEventListener("click", () => { state.itemQuery.offset = Math.max(0, state.itemQuery.offset - state.itemQuery.limit); loadItems(); });
  $("#items-next").addEventListener("click", () => { state.itemQuery.offset += state.itemQuery.limit; loadItems(); });
  $("#audit-refresh").addEventListener("click", loadAudit);
  $("#status-form").addEventListener("submit", handleStatusSubmit);
  $("#batch-action-button").addEventListener("click", openBatchDialog);
  $("#batch-status-form").addEventListener("submit", handleBatchStatusSubmit);
  $("#select-page-items").addEventListener("change", (event) => {
    $$("#items-body input[type='checkbox']").forEach((input) => {
      input.checked = event.currentTarget.checked;
      if (input.checked) state.selectedOperationItems.add(input.value);
      else state.selectedOperationItems.delete(input.value);
    });
    updateBatchControls();
  });
  $("#new-boost-button").addEventListener("click", () => {
    const form = $("#boost-form");
    form.reset();
    $("#boost-form [data-user-ids]").hidden = true;
    $("#boost-form [data-form-error]").hidden = true;
    setDefaultBoostTimes();
    $("#boost-dialog").showModal();
  });
  $("#boost-form").addEventListener("submit", handleBoostSubmit);
  $("#boost-form [name='audience']").addEventListener("change", (event) => {
    $("#boost-form [data-user-ids]").hidden = event.currentTarget.value !== "users";
  });
  $$('[data-close-dialog]').forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));
  window.addEventListener("auth:expired", () => showLogin("登录已过期，请重新登录。"));
  window.addEventListener("pagehide", beaconPendingImpressions);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      stopImpressionObservation();
      beaconPendingImpressions();
    } else {
      observeFeedCards();
    }
  });
}

try {
  sessionStorage.removeItem("pendingImpressions");
  sessionStorage.removeItem("reportedImpressionIds");
} catch {
  // Ignore stale pre-user-scoped storage when browser storage is unavailable.
}
initializeDashboardRange();
bindEvents();
establishSession();
