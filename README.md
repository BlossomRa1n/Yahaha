# MicroLens Recommendation System MVP

基于官方 MicroLens-50K 数据的模块化单体推荐系统：离线时间切分与
TruncatedSVD 训练、FastAPI/SQLite 在线服务、同源静态 Web 工作台、真实曝光与
行为回传、在线画像、Dashboard、强推/下线/恢复和审计。

> 当前状态：核心纵向链路、全量离线训练、API 测试和桌面/移动浏览器流程已实测。
> 只有 `docs/VERIFICATION.md` 中记录了真实退出码和输出摘要的项目才算已验证；
> 干净 checkout 的 smoke 链路已复现；正式演示视频仍明确列为未完成。

## 架构与边界

系统采用单进程 FastAPI 模块化单体、SQLite、NumPy 模型产物和无需构建的原生
HTML/CSS/JavaScript。浏览器与 API 同源，登录使用服务端保存的 HttpOnly
session。数据库是请求、曝光、行为、画像、运营状态和 Dashboard 指标的事实源。

在线规则顺序是：候选召回与排序 -> 普通候选去重/已看过滤 -> 生效强推 -> 最终
online 过滤 -> 持久化请求/曝光。`offline` 始终高于强推、fallback 和直接内容 API。
完整设计见 `docs/ARCHITECTURE.md`，API 和产物格式见 `docs/API.md`。

## 前置条件

- Python 3.11 或更高版本。
- 推荐安装 [uv](https://docs.astral.sh/uv/)；不要求 Docker、GPU、Redis 或 Node。
- CPU smoke 模式建议至少 4 GB 可用内存。当前机器 full `pipeline all` 实测
  `19.323s`，其中 prepare `1.755s`；峰值内存未测，不应把该耗时外推到其他机器。
- MicroLens 官方数据由使用者自行下载，遵守原项目的数据许可与使用条件。

安装依赖：

```powershell
Copy-Item .env.example .env
uv sync --group dev
```

Linux/macOS 将第一条替换为 `cp .env.example .env`。生产或共享环境必须修改
`APP_SECRET`，并通过环境变量管理真实配置；不要提交 `.env`。

## 官方数据准备

项目来源：[MicroLens GitHub](https://github.com/westlake-repl/MicroLens)。从官方
下载入口取得以下文件并放入 `data/raw/`：

```text
data/raw/MicroLens-50k_pairs.csv
data/raw/MicroLens-50k_titles.csv
data/raw/MicroLens-50k_likes_and_views.txt
```

`data/raw/`、`data/processed/` 和 `artifacts/` 均被 `.gitignore` 排除。不得提交、
重新打包或公开分发官方原始数据及其子集。MVP 不下载 637 MB cover 包；Web 使用
服务端生成的确定性占位封面。

一条命令完成处理、smoke 训练、评估和原子发布：

```powershell
uv run python -m recsys.pipeline all --raw-dir data/raw --out-dir data/processed --artifacts-dir artifacts --mode smoke --max-users 2000 --max-eval-users 500 --rank 32 --seed 20260901
```

仅处理数据：

```powershell
uv run python -m recsys.pipeline prepare --raw-dir data/raw --out-dir data/processed --seed 20260901
```

输出包括 `train.csv`、`validation.csv`、`test.csv`、`items.csv`、
`user_history.jsonl` 和 `summary.json`。全局时间边界保证训练、验证、测试顺序；
所有词表、热度、负采样池和模型只使用训练期数据，未定时的 likes/views 不进入
离线特征。当前官方数据摘要为 50,000 用户、19,220 内容、359,708 交互，时间范围
2020-03-05 至 2022-09-12 UTC；切分为 287,767 / 35,971 / 35,970 条。

完整训练：

```powershell
uv run python -m recsys.pipeline train --processed-dir data/processed --artifacts-dir artifacts --mode full --rank 32 --seed 20260901
```

训练成功后生成版本目录、无 pickle 的 NumPy 数组、`manifest.json`、
`metrics.json`、`evaluation.md` 和原子更新的 `artifacts/current.json`。失败训练不得
覆盖当前可用版本。

## 离线评估口径

同一批评估 query 比较 popular、确定性 random 和 TruncatedSVD。当前协议为：

- `K=10`，报告 `Recall@10`、`NDCG@10` 和 `HitRate@10`。
- 每个用户使用该时间切分中的全部 warm positives 和 100 个不重复、固定 seed 的
  train-item negatives。
- 指标按用户 macro average；候选全集仅包含训练用户在训练期可见的内容。
- validation 指标供人工比较配置，test 只做最终报告；热门统计仅来自 train。MVP
  没有自动超参搜索，当前 `rank=32` 是显式选择。

当前已生成并检查 full 产物 `svd-20260901T121430026505Z-cccf5c24`。test cohort
为 5,000 个用户，每个 query 100 个 negatives，warm-item coverage 为 `0.237160`：

| 模型 | Recall@10 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|
| Popular | 0.246499 | 0.123706 | 0.281800 |
| Random | 0.089834 | 0.043384 | 0.108800 |
| TruncatedSVD | 0.359176 | 0.208986 | 0.399000 |

原始精度、validation 指标、cohort 哈希与协议保存在该版本的 `metrics.json` 和
`evaluation.md`；README 仅做四舍五入展示。

## 初始化与启动

将处理后的内容导入 SQLite，并创建测试用户：

```powershell
uv run python -m app.cli init-db --items data/processed/items.csv --reset
```

`--reset` 会删除现有本地数据库及事件，只用于明确需要重建的开发或演示环境。
普通启动不要携带此参数。

启动服务：

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

访问路径：

- Web 工作台：`http://127.0.0.1:8000/`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`
- OpenAPI：`http://127.0.0.1:8000/docs`

前端由 FastAPI 同源托管，不需要 `npm install` 或单独 dev server。模型缺失或损坏
时 Feed 必须带 `fallback_reason` 并降级到热门/探索；不能伪造模型结果。

## 测试账号

| 角色 | 用户名 | 密码 | 用途 |
|---|---|---|---|
| 普通用户 | `alice` | `demo-pass` | 映射官方历史用户 A |
| 普通用户 | `bob` | `demo-pass` | 映射官方历史用户 B |
| 普通用户 | `carol` | `demo-pass` | 无历史冷启动用户 |
| 管理员 | `admin` | `admin-pass` | Dashboard 与运营 |

这些仅是本地 seed 账号。普通用户不显示管理入口，且服务端对所有 `/api/v1/admin`
接口再次鉴权；隐藏按钮不是安全边界。

## 核心验收流程

1. 用 `alice` 浏览个性化、热门、探索三路 Feed，记录页面显示的 request_id、来源、
   位置、解释和模型版本；加载更多应无重复。
2. 用 `bob` 登录并比较个性化结果；用 `carol` 验证可解释的冷启动 fallback。
3. 对内容执行点击、喜欢、不感兴趣，再打开“我的画像”并重新请求个性化 Feed。
4. 用 `admin` 打开 Dashboard，确认请求、曝光、点击、CTR、喜欢和 Feed 占比来自
   数据库且发生变化；使用 request_id 查询完整链路。
5. 创建定向强推，验证目标 Feed；随后下线同一内容，确认所有 Feed 和直接 item API
   均不返回；恢复后在有效规则下重新可见，并检查审计前后状态与原因。

详细 3–5 分钟录屏步骤见 `docs/DEMO.md`。

## 测试

```powershell
uv run pytest
node --check web/api.js
node --check web/app.js
```

仅运行不依赖服务的 Web 静态契约测试：

```powershell
uv run pytest tests/test_web_contract.py -p no:cacheprovider
```

测试代码可使用明确标注的合成 fixture；官方数据端到端证据必须使用本地下载的数据。
每次真实运行的命令、退出码、数据摘要、指标和失败原因记录到
`docs/VERIFICATION.md`。

## 完成度声明

已验证：当前 full 模型产物及其指标文件、15 项全仓 pytest、Web 静态契约断言、
JavaScript 语法、19,220 内容数据库初始化、真实纵向 API 链路，以及 1440px/390px
浏览器中的登录、Feed、行为画像、Dashboard、强推、下线、恢复和审计。浏览器控制台
没有 warning/error。独立干净 checkout 也已按 README 完成锁定依赖、smoke pipeline、
数据库初始化、全套测试、JavaScript 检查和健康检查。

降级项：API 返回即视为 impression，不能证明进入浏览器 viewport；封面为本地占位；
SQLite 同步更新画像；cursor 不是跨模型发布或运营变更的持久快照；事件当前只同步
影响在线画像，尚未实现自动导出并合入下一轮离线训练。

Mock：测试账号和占位封面是 seed/demo 辅助；自动化单元测试可生成合成数据。推荐
列表、Dashboard 数字、事件、画像和运营状态不得使用固定前端 JSON。

明确未做：注册、视频播放/托管、多模态、DSSM+DeepFM、Redis、异步任务、线上训练、
复杂图表、CSV/告警、云部署。

当前风险：全量 CPU 已实测约 19.3 秒但峰值内存未测；SQLite 并发写入能力有限；
恢复后的普通候选只恢复“可参与推荐”，演示需保留有效强推规则来稳定证明重新可见。

若增加一周：增加 Playwright 多浏览器 E2E、事件批量重试队列、时间范围 Dashboard
与趋势图、boost 管理/停用、离线训练消费在线事件、模型版本对比、结构化日志和延迟
分位数监控，再评估 PostgreSQL 与异步训练任务。

## 交付文档

- `docs/REQUIREMENTS.md`：需求追踪矩阵与状态。
- `docs/ARCHITECTURE.md`：数据流、在线链路、权限和失败恢复。
- `docs/API.md`：API、错误和模型产物契约。
- `docs/VERIFICATION.md`：真实命令、退出码和证据。
- `docs/AI_COLLABORATION.md`：AI prompt、人工 review、问题与修复。
- `docs/DEMO.md`：3–5 分钟演示脚本。
