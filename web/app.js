import { ApiError, api, clientEventId } from "./api.js";

const state = {
  user: null,
  currentView: "feed",
  feedType: "personalized",
  feedItems: [],
  feedCursor: null,
  feedHasMore: false,
  feedLoading: false,
  itemQuery: { q: "", status: "", limit: 20, offset: 0, total: 0 },
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

function showLogin(message = "") {
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

function showAuthenticated(user) {
  state.user = user;
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
  button.textContent = "登录中...";
  try {
    const response = await api.login(form.username.value.trim(), form.password.value);
    form.password.value = "";
    showAuthenticated(response.user);
  } catch (error) {
    errorElement.textContent = displayError(error);
    errorElement.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "登录";
  }
}

async function handleLogout() {
  try {
    await api.logout();
  } catch (error) {
    setGlobalAlert(displayError(error), "error");
  } finally {
    state.feedItems = [];
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
  actions.append(like, dislike);

  body.append(provenance, titleButton, explanation, details, actions);
  card.append(coverButton, body);
  return card;
}

function renderFeed() {
  const list = $("#feed-list");
  list.replaceChildren(...state.feedItems.map(buildFeedCard));
  const loadMore = $("#load-more-button");
  loadMore.hidden = !state.feedHasMore || state.feedItems.length === 0;
  loadMore.disabled = state.feedLoading;
}

async function loadFeed(reset = false) {
  if (state.feedLoading) return;
  state.feedLoading = true;
  $$("#feed-tabs [role='tab']").forEach((tab) => { tab.disabled = true; });
  const panel = $("#feed-state");
  setPanelState(panel, reset ? "正在生成推荐并记录曝光..." : "正在加载更多推荐...");
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
    setPanelState(panel, displayError(error), { type: "error", retry: () => loadFeed(reset) });
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
      ["曝光", formatNumber(summary.impressions)],
      ["点击", formatNumber(summary.clicks)],
      ["喜欢", formatNumber(summary.likes)],
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
      event.request_id,
    ], 4, "暂无行为记录");
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
    ["曝光", overview.exposures], ["点击", overview.clicks], ["CTR", formatPercent(overview.ctr)],
    ["喜欢", overview.likes], ["下线内容", overview.offline_items], ["生效强推", overview.active_boosts],
    ["当前模型", overview.current_model_version || "—"],
  ];
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

  const exposureTotal = Number(overview.exposures || 0);
  renderRows($("#feed-breakdown-body"), overview.feed_breakdown || [], (feed) => [
    feed.feed_type,
    formatNumber(feed.requests),
    formatNumber(feed.exposures),
    formatNumber(feed.clicks),
    formatPercent(feed.ctr ?? (Number(feed.exposures) ? Number(feed.clicks) / Number(feed.exposures) : 0)),
    formatPercent(feed.share ?? (exposureTotal ? Number(feed.exposures) / exposureTotal : 0)),
  ], 6, "暂无信息流请求");

  renderRows($("#top-items-body"), overview.top_items || [], (item) => [
    item.title ? `${item.title} (${item.item_id})` : item.item_id,
    formatNumber(item.exposures), formatNumber(item.clicks), formatNumber(item.likes),
    formatPercent(item.ctr ?? (Number(item.exposures) ? Number(item.clicks) / Number(item.exposures) : 0)),
  ], 5, "暂无热门内容数据");
}

function renderModels(response) {
  renderRows($("#models-body"), response.models || [], (model) => [
    model.model_version,
    model.algorithm,
    model.status === "active" || model.model_version === response.current_model_version ? "当前" : model.status,
    formatDate(model.activated_at || model.created_at),
    Object.entries(model.metrics || {}).map(([key, value]) => {
      const displayed = typeof value === "number"
        ? value.toFixed(4)
        : typeof value === "object" ? JSON.stringify(value) : value;
      return `${key}: ${displayed}`;
    }).join(" · ") || "—",
  ], 5, "暂无已登记模型");
}

async function loadDashboard() {
  const panel = $("#dashboard-state");
  setPanelState(panel, "正在聚合 Dashboard 指标...");
  const [overviewResult, modelsResult] = await Promise.allSettled([api.dashboard(), api.models()]);
  if (overviewResult.status === "fulfilled") {
    renderDashboard(overviewResult.value);
    hidePanelState(panel);
  } else {
    setPanelState(panel, displayError(overviewResult.reason), { type: "error", retry: loadDashboard });
  }
  if (modelsResult.status === "fulfilled") {
    renderModels(modelsResult.value);
  } else {
    renderRows($("#models-body"), [], () => [], 5, displayError(modelsResult.reason));
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
  renderRows($("#items-body"), items, (item) => [
    item.item_id,
    item.title,
    formatNumber(item.popularity_score ?? item.views),
    statusBadge(item.status),
    formatDate(item.updated_at),
    itemActions(item),
  ], 6, "没有符合条件的内容");
  const page = Math.floor(state.itemQuery.offset / state.itemQuery.limit) + 1;
  $("#items-page-label").textContent = `第 ${page} 页 · 共 ${formatNumber(state.itemQuery.total)} 条`;
  $("#items-previous").disabled = state.itemQuery.offset <= 0;
  $("#items-next").disabled = state.itemQuery.offset + items.length >= state.itemQuery.total;
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
      `${auditValue(operation.before)} → ${auditValue(operation.after)}`,
      operation.reason,
    ], 6, "暂无操作记录");
  } catch (error) {
    renderRows($("#operations-body"), [], () => [], 6, displayError(error));
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
}

bindEvents();
establishSession();
