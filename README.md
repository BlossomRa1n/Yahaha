# MicroLens Recommendation System MVP

基于官方 MicroLens-50K 数据的模块化单体推荐系统：离线时间切分、TruncatedSVD、
标题字符 n-gram TF-IDF 和 item-item CF，FastAPI/SQLite 在线服务、五源候选混排、
同源静态 Web 工作台、真实曝光与行为回传、在线画像、Dashboard、运营控制和审计。

> 当前状态：核心纵向链路、全量离线训练、API 测试和桌面/移动浏览器流程已实测。
> 只有 `docs/VERIFICATION.md` 中记录了真实退出码和输出摘要的项目才算已验证；
> 干净 checkout 的 smoke 链路已复现；正式演示视频链接已交付。

- 源码仓库：<https://github.com/BlossomRa1n/Yahaha>
- 本地 Demo：<http://127.0.0.1:8000/>（按下方“初始化与启动”执行）
- 演示视频：[江雨鸿-视频演示.mp4（夸克网盘）](https://pan.quark.cn/s/81e4e62f191a)
- 逐项交付状态：[`docs/DELIVERY_CHECKLIST.md`](docs/DELIVERY_CHECKLIST.md)

## 架构与边界

系统采用单进程 FastAPI 模块化单体、SQLite、NumPy 模型产物和无需构建的原生
HTML/CSS/JavaScript。浏览器与 API 同源，登录使用服务端保存的 HttpOnly
session。数据库是请求、曝光、行为、画像、运营状态和 Dashboard 指标的事实源。

在线规则顺序是：最终下线过滤 > 合法强推 > 权限/已看/去重 > validation 锁定的共享混排策略 >
标题多样性重排 > 普通排序。第一页持久化有界候选快照，后续 Cursor 只按 offset
读取同一快照；下线仍会立即覆盖旧快照、fallback 和直接内容 API。
完整设计见 `docs/ARCHITECTURE.md`，API 和产物格式见 `docs/API.md`。

## 前置条件

- Python 3.11 或更高版本。
- 推荐安装 [uv](https://docs.astral.sh/uv/)；不要求 Docker、Redis 或 Node。GPU 可选：
  `train-deep` 默认 `--device auto`，检测到 CUDA 时使用 GPU，否则回退 CPU。
- CPU smoke 模式建议至少 4 GB 可用内存。全量流程峰值内存尚未测量，不能据此宣称
  full 模式的最低内存；资源不足时先运行 smoke。
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
重新打包或公开分发官方原始数据及其子集。多模态实验在本地使用官方
`MicroLens-50k_covers.zip` 提取离线特征；原始封面、预训练权重和 embedding 不进入
Git。Web 仍使用服务端生成的确定性占位封面，不直接分发官方图片。

### 从零运行完整数据与统一模型

以下命令在仓库根目录按顺序执行。前三个 CSV/TXT 和封面压缩包必须已放入
`data/raw/`。首次运行需要联网安装锁定依赖并下载 torchvision 官方 MobileNetV3-Small
权重；已有缓存时不会重复下载。

```powershell
Copy-Item .env.example .env
uv sync --group dev
uv run python -c "from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small; mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)"
uv run python -m recsys.pipeline all --raw-dir data/raw --out-dir data/processed --artifacts-dir artifacts --mode full --max-eval-users 5000 --rank 32 --seed 20260901
uv run python -m recsys.pipeline train-multimodal --processed-dir data/processed --artifacts-dir artifacts --base-pointer artifacts/current.json --archive data/raw/MicroLens-50k_covers.zip --covers-dir data/raw/MicroLens-50k_covers --batch-size 64 --pca-dim 128 --max-eval-users 5000 --locked-visual-weight 0.20
uv run python -m recsys.pipeline train-deep --processed-dir data/processed --artifacts-dir artifacts --base-pointer artifacts/current.json --multimodal-pointer artifacts/multimodal-current.json --mode full --validation-mode sampled --max-eval-users 5000 --epochs 8 --patience 2 --device auto
uv run python -m app.cli init-db --items data/processed/items.csv --reset
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

最后一条命令会持续运行服务；浏览器打开 `http://127.0.0.1:8000/`。`init-db --reset`
会清空并重建本地演示数据库，只应在首次初始化或明确重建演示环境时执行。

当前机器的实测与资源边界如下，均不包含下载时间：

| 阶段/资源 | 当前实测或已知边界 |
|---|---|
| 基础数据处理 + SVD full | `19.323s`，其中 prepare `1.755s` |
| 统一深度 full + 四组 validation | RTX 5070 Laptop 8 GB CUDA，约 `6.2` 分钟 |
| checkpoint 恢复后的 validation + test | 同一机器约 `5.2` 分钟；这是单独重跑，不应与上一项简单相加 |
| 多模态特征提取 | 当前实现使用 CPU；未单独记录完整耗时，不给出虚假估计 |
| 内存 | smoke 建议至少 4 GB 可用内存；full 峰值未测 |
| 当前磁盘实占 | `data/raw` 约 1.26 GiB、`data/processed` 约 0.02 GiB、`artifacts` 约 3.20 GiB、`.venv` 约 4.40 GiB，合计约 8.9 GiB；训练临时峰值未测，应额外预留空间 |

纯 CPU 可以完成统一深度训练与推理，但尚无 full 深度训练的 CPU 实测耗时。上述 CUDA
时间只代表当前机器，不能外推到其他硬件。若只需快速检查安装和数据链路，使用下方
smoke 命令，不必重训完整统一模型。

一条命令完成处理、smoke 训练、评估和原子发布：

```powershell
uv run python -m recsys.pipeline all --raw-dir data/raw --out-dir data/processed --artifacts-dir artifacts --mode smoke --max-users 2000 --max-eval-users 500 --rank 32 --seed 20260901
```

仅处理数据：

```powershell
uv run python -m recsys.pipeline prepare --raw-dir data/raw --out-dir data/processed --seed 20260901
```

输出包括 `train.csv`、`validation.csv`、`test.csv`、`items.csv`、
`user_history.jsonl`、`stats_snapshot.json` 和 `summary.json`。全局时间边界保证训练、验证、测试顺序；
所有词表、热度、负采样池和模型只使用训练期数据，未定时的 likes/views 不进入
离线特征。当前官方数据摘要为 50,000 用户、19,220 内容、359,708 交互，时间范围
2020-03-05 至 2022-09-12 UTC；切分为 287,767 / 35,971 / 35,970 条。

`stats_snapshot.json` 记录 likes/views 的版本、`available_at`、源文件 SHA-256、
文件名、源文件修改时间和质量摘要；数据库另行记录实际导入时间。默认
`available_at=null`，因此历史训练/回测禁用；
只有确知快照可用时间时才传
`--stats-available-at 2026-09-02T04:00:00Z`。训练期 popular 使用 cutoff 前交互
构造累计、1/7/30 天、时间衰减和增长特征，不使用最终累计 likes/views。

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

当前已生成并检查 full 产物 `svd-20260902T143346986486Z-ab4c3e04`。test cohort
为 5,000 个用户，每个 query 100 个 negatives，warm-item coverage 为 `0.237160`：

| 模型 | Recall@10 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|
| Popular | 0.246499 | 0.123706 | 0.281800 |
| Random | 0.089834 | 0.043384 | 0.108800 |
| TruncatedSVD | 0.359176 | 0.208986 | 0.399000 |
| Content only | 0.231333 | 0.153964 | 0.260000 |
| Item-item CF | 0.172977 | 0.119093 | 0.200000 |
| 锁定混排策略 | 0.359176 | 0.208986 | 0.399000 |

原始精度、validation 指标、cohort 哈希与协议保存在该版本的 `metrics.json` 和
`evaluation.md`；README 仅做四舍五入展示。报告还包含 validation/test 各最多 5 个
稳定匿名化的 SVD Badcase，给出正样本在同一 sampled candidate set 中的实际排名、
未命中原因和 train-only 候选覆盖限制；这些排名不是全目录排名。

统一 sampled-all-items 评估保留所有正样本（含 cold item），并为每个用户采样 100 个
确定性负例；SVD 不可打分的正样本按 miss 计入。SVD-only Recall@10 为
`0.085134`，content-only 为 `0.259842`，SVD+content fallback 为 `0.367097`，
锁定混排策略为 `0.367097`。标题内容产物覆盖 25,903 个 cold positive 中的
25,869 个，cold-item coverage 为 `0.998687`。训练使用线上同一个
`mix_candidates` 纯函数：只在 validation 比较 `dynamic_confidence_v2` 与
`safe_svd_content_v2`，test 只运行选定策略。动态策略的 validation sampled-all-items
Recall/NDCG 为 `0.283030/0.147777`，低于安全策略的 `0.382142/0.181380`，因此该次
基础模型比较选择安全策略；这些指标仅用于透明记录方案取舍。

## DSSM、DeepFM 与多模态（统一生产链路）

统一 7 源 + DeepFM 是个性化 feed 的唯一生产路径：SVD、DSSM、标题内容、视觉、
item-item CF、热门、探索七路召回，过滤去重后由单一 DeepFM 重排。`artifacts/current.json`
仍是稳定 SVD 指针，但只作为召回源之一；深度和视觉模型使用独立的
`experiment-current.json`、`multimodal-current.json`。统一深度/视觉产物不可用时，
已映射的 warm 用户回退到个性化 SVD；cold 用户或基础模型也不可用时回退热门/探索。

```powershell
uv run python -m recsys.pipeline train-multimodal --batch-size 64 --pca-dim 128 --max-eval-users 5000 --locked-visual-weight 0.20
uv run python -m recsys.pipeline train-deep --mode full --validation-mode sampled --max-eval-users 5000 --epochs 8 --patience 2
```

当前统一两阶段实验 `deep-20260903T045748115694Z-a8df9062` 真实训练 DSSM 和
DeepFM。七路召回为 SVD、DSSM、标题内容、视觉、item-item CF、热门和探索；过滤、
去重后，DeepFM 同时消费七路归一化分数、来源 multi-hot、用户/item、历史密度、
热度、cold 和视觉可用性特征。来源配置仅作为生成/Top-10 上限，不强制低质量来源
入选。每 epoch 保存 safetensors、optimizer 和原子 best/latest 指针；DSSM/DeepFM
均由 patience=2 早停并只导出最佳 checkpoint。训练使用确定性 20% item-ID/协同源
dropout 模拟 cold-start，不使用 validation 标签。DeepFM validation AUC 为
`0.708714`，线性基线为 `0.687647`。

该实验在 validation 上 sampled-all-items NDCG `0.205171` 高于稳定策略 `0.181380`，Warm
Recall/NDCG `0.362642/0.227026` 不低于稳定策略 `0.359512/0.218799`；但 sampled-all-items
Recall `0.319631` 低于 `0.382142`、HitRate `0.413400` 低于 `0.486200`。该纯 DeepFM
权衡（提升 NDCG、退化 Recall/HitRate）已被接受，统一链路成为
唯一生产路径。锁定策略只运行一次 test：sampled-all-items Recall/NDCG/HitRate 为
`0.257060/0.159583/0.342000`。统一模型不可用时，warm 用户回退个性化 SVD，
cold 用户或基础模型不可用时回退热门/探索。

锁定多模态实验 `multimodal-20260902T180847621178Z-7e09b190` 使用真实
MobileNetV3-Small ImageNet 权重，19,220 张封面映射/解析成功率和 item/cold-item
覆盖率均为 100%；PCA128 只在 16,907 个 train-visible item 上拟合。validation
锁定图文 late-fusion 视觉权重 `0.20`：相对 text-only，sampled-all-items Recall
`0.291515 -> 0.310377`，NDCG `0.210490 -> 0.213680`；最终 test Recall/NDCG/
HitRate 为 `0.276839/0.185855/0.358200`。在线只加载缓存向量，不做图片推理。

## 初始化与启动

将处理后的内容导入 SQLite，并创建测试用户：

```powershell
uv run python -m app.cli init-db --items data/processed/items.csv --reset
```

`--reset` 会删除现有本地数据库及事件，只用于明确需要重建的开发或演示环境。
普通启动不要携带此参数。

启动服务：

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

访问路径：

- Web 工作台：`http://127.0.0.1:8000/`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`
- OpenAPI：`http://127.0.0.1:8000/docs`

前端由 FastAPI 同源托管，不需要 `npm install` 或单独 dev server。模型缺失或损坏
时 Feed 必须带 `fallback_reason`：已映射 warm 用户优先回退个性化 SVD，cold 用户或
基础模型不可用时回退热门/探索；不能伪造模型结果。

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
   位置、解释和模型版本；加载更多来自同一 TTL 快照，应稳定且无重复。
2. 用 `bob` 登录并比较个性化结果；用 `carol` 验证可解释的冷启动 fallback。
3. 对内容执行点击、喜欢、不感兴趣和分享；让卡片可见后离开视口产生 dwell，重新
   访问同一内容验证服务端推导的 revisit，再打开“我的画像”并刷新 Feed。
4. 用 `admin` 打开 Dashboard，确认服务曝光、可见曝光、点击、服务 CTR、可见 CTR、
   喜欢和 Feed 占比来自数据库且发生变化；使用 request_id 查询完整链路。
5. 创建定向强推，验证目标 Feed；随后下线同一内容，确认所有 Feed 和直接 item API
   均不返回；恢复后在有效规则下重新可见，并检查审计前后状态与原因。

Feed 返回只写服务曝光。卡片至少 50% 可见并持续 750 ms 后，前端才批量上报
`impression` 并开始累计 dwell；离开视口、页面隐藏或卸载时通过有限重试/Beacon
上报累计停留。分享仅在 Web Share 或复制链接成功后记录，取消不记事件。revisit
由服务端按用户、内容和不同 request 的有效 impression 推导，不信任客户端序号。

Dashboard 的三路分组分别聚合请求、服务曝光、可见曝光、点击、喜欢、不感兴趣、
分享、重复访问和平均停留；概览同时展示停留 P95 与请求延迟 P50/P95/P99，趋势面板
可按小时或天查看请求、两类曝光、点击、喜欢、分享、重复访问和停留事件。所有值均来自
SQLite 请求、曝光和事件事实，不从前端常量生成。

Dashboard 的开始/结束时间在页面点击“应用”后，以同一组 UTC `from/to` 参数刷新概览
和趋势；统计范围统一为 `[from,to)`，默认最近 24 小时且最多 366 天。管理员可从页面
下载当前已应用范围的 UTF-8 CSV，对应接口为
`GET /api/v1/admin/dashboard/export.csv?from=...&to=...`。CSV 复用概览聚合结果，
包含总览、三路 Feed、候选来源和热门内容记录，并明确区分 `served_ctr` 与
`viewable_ctr`。

清理过期 Feed 快照：

```powershell
uv run python -m app.cli cleanup-snapshots
```

在线事件会同步更新画像和下一次个性化排序。需要为后续训练准备事件快照时可运行：

```powershell
uv run python -m app.cli export-events --out data/staging/online_events.csv
```

将真实事件按服务端接收时间窗口合入下一轮训练、评估并原子发布：

```powershell
uv run python -m app.cli retrain-events `
  --start-time 2026-09-02T00:00:00Z `
  --end-time 2026-09-03T00:00:00Z `
  --base-processed-dir data/processed `
  --output-root data/retraining `
  --mode smoke --max-users 2000 --max-eval-users 500 --rank 32 --seed 20260902
```

窗口为 `[start_time,end_time)`，使用不可伪造的服务端 `received_at`。已知历史数据全部作为
窗口前训练基线，窗口内正反馈再严格按时间切分；`impression=0`（上下文）、`click=1`、
`like=3`、`share=4`、`revisit=1.5`，dwell 按 750ms/5s/30s 分为
`0.25/0.75/1.5`，`not_interested=-2`。同一用户内容的窗口正向权重封顶 6，出现
不感兴趣时正向信号不进入该窗口训练。数据版本记录窗口、事件 ID 摘要、跳过原因和样本数；只有
验证/测试 cohort 可评估、指标有限且模型产物能被线上加载时才更新 `current.json`。失败运行
写入 `training_runs`，并保持旧 pointer 和旧模型目录不变。完整数据使用 `--mode full`

管理员 Dashboard 的“模型运行与版本对比”从 `model_versions` 和 `training_runs`
读取真实训练窗口、样本量、发布状态及离线指标。选择 2–10 个版本后，只有
`evaluation_protocol` 完全一致时才展示相对首个版本的指标变化；口径不同或缺失时
仍展示绝对值，但明确禁止计算误导性的差值。该接口为管理员专用。

个性化 Feed 统一接收 `model`、`content_profile`、`item_cf`、`popular`、`explore`
五类候选。共享混排先剔除无资格、非有限及零支持 CF 候选，再做来源内 rank
归一化；动态策略按 warm/cold、历史密度、CF 支持度和内容置信度选择配额，来源不足时
确定性放宽。validation 用于比较并记录候选策略，产物保存本次选定的混排策略。
全局去重后再执行标题词元 MMR。TF-IDF 词表仅由训练期可见标题
拟合，固定词表可转换预测时已存在的新内容；用户内容向量和 CF 共现均不读取 cutoff
之后的行为。各产物带 Hash、数据版本和 cutoff，单一产物损坏只禁用对应来源。

Feed 在创建稳定快照前对普通候选执行确定性的标题词元 MMR 重排。原始相关性分数
不被改写，合法强推位置在重排后固定；下线、用户隔离、已看过滤和去重优先于多样性。
快照保存 `diversity_json`，Feed 响应的 `diversity.before/after` 提供相邻标题相似度、
列表内多样性、标题词元覆盖和重复数，旧 cursor 始终重放同一重排结果。

内容运营支持管理员批量下线和批量恢复（每批最多 100 个去重后的内容）。接口采用
全量原子事务，任一未知内容会使整批失败；`idempotency_key` 防止网络重试产生重复
副作用，`batch_id` 关联每个内容的独立审计记录。批量下线会同步使旧 Feed 快照中的
对应条目失效，下线优先级继续高于强推。

输出为 `user,item,timestamp,event_type,weight`，仅包含映射到官方用户的 click/like。
它是明确标记的 **staging snapshot**，当前 train-only benchmark 不会自动读取该文件；
合并线上时段前必须重新确定时间 cutoff 并重新生成 validation/test，不能沿用旧指标。

演示视频见[夸克网盘分享链接](https://pan.quark.cn/s/81e4e62f191a)，录屏步骤见
`docs/DEMO.md`。

## 测试

```powershell
uv run python -m pytest
node --check web/api.js
node --check web/app.js
```

已有本地官方数据和模型产物时，可用临时 SQLite 执行不污染现有数据库的完整验收：

```powershell
uv run python -m scripts.verify_official_e2e --items data/processed/items.csv --model-pointer artifacts/current.json
```

仅运行不依赖服务的 Web 静态契约测试：

```powershell
uv run python -m pytest tests/test_web_contract.py -p no:cacheprovider
```

测试代码可使用明确标注的合成 fixture；官方数据端到端证据必须使用本地下载的数据。

### 持续集成

`.github/workflows/ci.yml` 在 `main` 分支 push 和 Pull Request 上运行。它使用
`uv.lock` 安装锁定依赖并缓存对应版本，执行完整 pytest、Python 编译检查、两份
JavaScript 语法检查，以及不依赖官方数据、GPU、真实密钥或本地数据库的合成数据
处理/训练/迁移/健康检查 smoke。任一步骤失败都会使 CI 失败。

当前 `origin` 配置为 `https://github.com/BlossomRa1n/Yahaha.git`。远端源码和 CI
状态以交付清单中的实测记录为准；本地 CI 等价命令如下。

本地等价命令：

```powershell
uv sync --locked --group dev
uv run python -m pytest -p no:cacheprovider
uv run python -m compileall -q app recsys scripts tests
node --check web/app.js
node --check web/api.js
uv run python -m scripts.ci_smoke
```
每次真实运行的命令、退出码、数据摘要、指标和失败原因记录到
`docs/VERIFICATION.md`。

## 完成度声明

已验证：当前 full 模型产物、深度/多模态隔离实验、指标与 Badcase 报告、全仓
`102 passed`、Web 静态契约断言、
JavaScript 语法、19,220 内容数据库初始化、真实纵向 API 链路，以及 1440px/390px
浏览器中的登录、Feed、行为画像、Dashboard、强推、下线、恢复和审计。浏览器控制台
没有 warning/error。独立干净 checkout 也已按 README 完成锁定依赖、smoke pipeline、
数据库初始化、全套测试、JavaScript 检查和健康检查。

已知边界：统一 7 源 + DeepFM 已是个性化 feed 唯一生产路径（纯 DeepFM，提升 NDCG 但
退化 sampled-all-items Recall/HitRate，该权衡已接受）；稳定 SVD 指针用于召回和 warm
用户故障回退，多模态产物保留独立离线评估结果。viewable impression 由浏览器按
50% 可见且持续 750ms 上报，`sendBeacon`
只能确认浏览器接收发送任务，不能提供服务端确认；封面为本地占位；SQLite 同步更新
画像。dwell 上限为 10 分钟，重复提交在同一 request/item/position 上取最大值；
revisit 只统计跨 request 的可见访问。Feed 使用带 TTL 的持久候选快照，模型、画像和
强推变化不改变旧快照普通顺序，但下线会立即过滤旧快照。事件窗口重训提供同步
`retrain-events` CLI，同时提供 HTTP 异步训练任务（后台线程复用同一重训发布流程，
作业生命周期 queued→running→succeeded/failed 已测试；真实重训路径与 CLI 共用，
未在服务进程内重新跑完整训练）。

Mock：测试账号和占位封面是 seed/demo 辅助；自动化单元测试可生成合成数据。推荐
列表、Dashboard 数字、事件、画像和运营状态不得使用固定前端 JSON。

明确未做：MicroLens 原始视频播放/托管和云部署。远端 GitHub CI 已在 `main` 提交
`970d23a` 上通过：[CI run 33817297423](https://github.com/BlossomRa1n/Yahaha/actions/runs/33817297423)。
注册（scrypt 密码哈希 + HTTP-only session + 角色权限）已完整实现；
DSSM+DeepFM、checkpoint/早停和 MobileNet 图文融合已实现并验证为统一生产路径；
同步事件窗口重训、模型版本对比、Dashboard CSV 和本地 CI workflow 已实现；
阈值告警（9 个实时 DB 聚合指标、规则/事件表与管理员 API）已实现并测试；
Redis 缓存为可选加速（未配置 `REDIS_URL` 或缺少 redis 包时降级为 no-op，公开 item
载荷缓存 + 状态变更失效已测试；真实 Redis 后端未在本机运行验证）；
`ts-interop/` 为 Bun/Express/Prisma/PostgreSQL/Elasticsearch 联调 sidecar（源码与
优雅失败路径契约已检查；当前机器没有 Bun、TypeScript 编译器、已安装依赖或锁文件，
因此不宣称类型检查通过；PG/ES/上游成功写入与代理路径也尚未联调）。

当前风险：原始 SVD full 曾实测约 19.3 秒；统一深度模型全量训练和四权重 validation
约 6.2 分钟，恢复 checkpoint 后的 validation+test 约 5.2 分钟，峰值内存仍未测。
SQLite 并发写入能力有限；
恢复后的普通候选只恢复“可参与推荐”，演示需保留有效强推规则来稳定证明重新可见。

若增加一周：优先增加 Playwright 多浏览器 E2E、长时间压测和跨平台 CI 矩阵，再根据
真实容量数据评估 PostgreSQL 与异步训练任务；不建议为技术名词展示迁移当前架构。

## 交付文档

- `docs/REQUIREMENTS.md`：需求追踪矩阵与状态。
- `docs/ARCHITECTURE.md`：数据流、在线链路、权限和失败恢复。
- `docs/API.md`：API、错误和模型产物契约。
- `docs/VERIFICATION.md`：真实命令、退出码和证据。
- `docs/AI_COLLABORATION.md`：AI prompt、人工 review、问题与修复。
- `docs/DEMO.md`：3–5 分钟演示脚本。
- `docs/DELIVERY_CHECKLIST.md`：按题目 3.3 顺序排列的交付入口与缺口。
