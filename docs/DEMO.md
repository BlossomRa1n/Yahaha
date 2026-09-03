# 3–5 Minute Demonstration Script / 演示视频录制方法

本视频使用真实终端、真实浏览器和真实数据库结果，依次覆盖题目 3.3 要求的五项内容。
全量深度训练可以提前完成；视频中现场运行 CPU smoke，并展示已有完整训练产物和指标，
不要用固定 JSON、截图或口头描述替代真实操作。

交付视频：[江雨鸿-视频演示.mp4（夸克网盘）](https://pan.quark.cn/s/81e4e62f191a)。
2026-09-04 已验证分享落地页返回 HTTP 200。

## 录制前准备

1. 将仓库设为公开，并用无痕窗口确认
   <https://github.com/BlossomRa1n/Yahaha> 可以访问。
2. 从 [MicroLens 官方仓库](https://github.com/westlake-repl/MicroLens) 的官方入口
   提前下载以下文件到一个不属于 Git 仓库的目录，例如 `D:\MicroLens-official`：

   ```text
   MicroLens-50k_pairs.csv
   MicroLens-50k_titles.csv
   MicroLens-50k_likes_and_views.txt
   ```

   多模态部分使用的 `MicroLens-50k_covers.zip` 也必须来自官方入口，但无需在
   3-5 分钟视频中重新下载或解压。
3. 提前完成完整训练，保留 `artifacts/` 中的 SVD、DSSM/DeepFM 和多模态产物。
   视频只展示这些产物的真实指标，并现场运行快速 smoke。
4. 准备两个终端窗口：终端 A 执行命令，终端 B 保持 Uvicorn 运行。
5. 浏览器缩放设为 100%，关闭包含 Token、Cookie、个人信息或真实密钥的页面。
6. 选择一个当前在线、标题容易辨认的内容 ID，用于强推和下线。

> `init-db --reset` 会清空本地演示数据库。只在确认可以丢弃当前演示事件时执行。

## 0:00-0:50 下载源码和官方数据

先在浏览器展示公开仓库首页及提交记录，然后在终端 A 展示从零获取源码的命令：

```powershell
Set-Location D:\
git clone https://github.com/BlossomRa1n/Yahaha.git Yahaha-demo-recording
Set-Location D:\Yahaha-demo-recording
git log --oneline -5
```

如果已经克隆过，不要删除原目录；换一个新的空目录名。随后在浏览器打开 MicroLens
官方仓库和官方数据下载入口，指出数据由官方提供，不会上传到本项目仓库。

展示已从官方入口下载完成的三个文件，并复制到新源码目录：

```powershell
New-Item -ItemType Directory -Force data\raw
Copy-Item D:\MicroLens-official\MicroLens-50k_pairs.csv data\raw\
Copy-Item D:\MicroLens-official\MicroLens-50k_titles.csv data\raw\
Copy-Item D:\MicroLens-official\MicroLens-50k_likes_and_views.txt data\raw\
Get-ChildItem data\raw
git status --short
```

口播：

> 源码来自公开 GitHub 仓库，数据来自 MicroLens 官方入口。原始数据放在
> `data/raw`，该目录被 Git 忽略，不会重新分发数据集。

为了把总时长控制在 5 分钟内，克隆和数据来源展示完成后，切回已经完成依赖安装和
全量模型训练的工作目录：

```powershell
Set-Location D:\Yahaha_codex
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
uv sync --locked --group dev
```

口播说明当前工作目录与刚克隆的仓库代码一致，预先训练产物只用于节省录制等待时间。

## 0:50-1:30 数据处理、训练评估与启动

在终端 A 现场运行一条 CPU smoke 命令：

```powershell
uv run python -m recsys.pipeline all --raw-dir data/raw --out-dir .tmp/demo-smoke/processed --artifacts-dir .tmp/demo-smoke/artifacts --mode smoke --max-users 2000 --max-eval-users 500 --rank 32 --seed 20260901
```

展示命令成功退出，并打开：

- `.tmp/demo-smoke/processed/summary.json`：用户数、内容数、交互数、时间范围和时间切分。
- `.tmp/demo-smoke/artifacts/` 当前 smoke 模型目录中的 `evaluation.md` 或
  `metrics.json`：Recall@10、NDCG@10、
  HitRate@10 和 AUC。
- `artifacts/current.json`、`artifacts/experiment-current.json` 和
  `artifacts/multimodal-current.json`：SVD、统一 DSSM/DeepFM 模型及视觉组件版本。

> smoke 必须写入 `.tmp/demo-smoke`，不能写入正式的 `data/processed` 或
> `artifacts`；否则会覆盖统一生产模型依赖的基础版本指针。

口播：

> 数据严格按时间切成训练、验证和测试集。个性化采用七路召回，DSSM 是正式召回源，
> 候选合并去重后由 DeepFM 统一排序；训练支持 checkpoint、早停和 GPU/CPU 自动选择。

初始化数据库：

```powershell
uv run python -m app.cli init-db --items data/processed/items.csv --reset
```

在终端 B 启动服务：

```powershell
Set-Location D:\Yahaha_codex
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

浏览器访问 <http://127.0.0.1:8000/>，确认页面正常加载。

## 1:30-2:20 两个用户与不同信息流

1. 使用 `alice / demo-pass` 登录。
2. 打开“个性化”，展示前三项内容，并指出 `request_id`、模型版本、来源、位置、
   分数和解释。
3. 点击“加载更多”，说明 Cursor 分页保持顺序稳定且无重复。
4. 切换“热门”，展示全局热度信息流。
5. 退出 Alice，使用 `bob / demo-pass` 登录。
6. 打开“个性化”，与刚才 Alice 的前三项进行可见对比。
7. 切换“探索”，展示未看或低曝光内容。

口播：

> 登录态由服务端 HttpOnly Session 识别。Alice 和 Bob 的历史、画像、推荐结果与
> 行为互相隔离，因此个性化结果不同；热门和探索是两条独立信息流。

## 2:20-3:05 行为上报、画像和指标变化

回到 Alice：

1. 记住当前个性化 Feed 的 `request_id`。
2. 点击一个内容、喜欢另一个内容，并对第三个内容选择“不感兴趣”。
3. 让一张卡片至少 50% 可见并停留 750 ms，触发可见曝光。
4. 打开“我的画像”，展示最近行为、正负偏好、画像版本和关联 request_id。
5. 刷新个性化 Feed，展示“不感兴趣”内容被过滤，或排序/解释产生变化。

随后登录 `admin / admin-pass`，打开 Dashboard：

1. 展示请求数、服务曝光、可见曝光、点击、点赞和 CTR。
2. 与行为前的基线比较，指出真实增量。
3. 在请求链路查询中输入刚才的 request_id，展示用户、返回列表、曝光和行为关联。

口播：

> Feed、曝光和行为都写入 SQLite，Dashboard 从数据库聚合，不是前端固定数字。
> request_id 可以把一次推荐、位置、曝光和后续行为串起来。

## 3:05-4:20 Dashboard 强推、下线和线上验证

保持管理员登录：

1. 在内容运营中搜索预先准备的在线内容 ID。
2. 创建强推：用户选择 Alice，Feed 选择个性化，位置为 0，有效期设为未来 24 小时，
   原因填写 `demo promotion`。
3. 登录 Alice 并刷新个性化 Feed，展示该内容位于指定位置，同时指出
   `is_forced`、来源和 request_id。
4. 切回管理员，将同一内容下线，原因填写 `demo offline verification`。
5. 再登录 Alice，刷新个性化、热门和探索 Feed，确认该内容全部消失。
6. 访问 `http://127.0.0.1:8000/api/v1/items/{item_id}`，确认直接 API 同样返回
   404，证明不是只在前端隐藏。
7. 管理员恢复内容，并打开审计记录，展示管理员、操作时间、原因和前后状态。

口播：

> 强推和下线都由服务端执行。下线优先级高于强推，因此内容下线后所有 Feed、旧快照
> 和直接内容 API 都不能绕过；恢复和每次操作都有审计记录。

## 4:20-5:00 测试、结果与边界

在终端 A 展示测试命令和已有结果：

```powershell
uv run python -m pytest -q -p no:cacheprovider --basetemp=.tmp/pytest-demo-video
```

打开 `docs/VERIFICATION.md`，展示最近一次完整回归、模型版本和关键指标；再打开
`docs/DELIVERY_CHECKLIST.md`，确认交付入口完整。

最后口播：

> 当前已完成从官方数据处理、训练评估、统一两阶段推荐、多人登录、行为回传、
> Dashboard 到内容运营的真实闭环。Web 使用占位封面，不托管原始视频；SQLite 适合
> 当前 MVP；HTTP 异步训练使用进程内线程；TypeScript sidecar 的 PostgreSQL 和
> Elasticsearch 成功路径尚未在本机联调。

## 录制后检查

- 视频时长在 3-5 分钟内，声音清晰，终端文字能够辨认。
- 五项要求都有真实画面，不只口头描述。
- 两个用户的个性化结果在画面中能直接比较。
- Dashboard 能看到行为后的真实增量。
- 强推后内容出现，下线后 Feed 和直接 API 都无法返回。
- 训练/评估命令、退出状态、模型版本和至少两项指标清晰可见。
- 视频中没有真实密钥、Cookie、Token、个人路径信息或官方数据内容。
- 上传后用无痕窗口验证链接无需申请权限即可播放。
