# 3–5 Minute Demonstration Script

本脚本要求录制真实终端和浏览器，不剪接固定 JSON 或伪造指标。长时间全量训练可提前
完成，但视频必须展示可复现命令、产物版本和真实评估文件。

## 0:00–0:40 数据、训练与启动

1. 展示 `data/raw/` 仅存在本地且被 `.gitignore` 排除，不打开或重新分发原始数据。
2. 展示并运行 CPU smoke 命令：

   ```powershell
   uv run python -m recsys.pipeline all --raw-dir data/raw --out-dir data/processed --artifacts-dir artifacts --mode smoke --max-users 2000 --max-eval-users 500 --rank 32 --seed 20260901
   ```

3. 展示 `summary.json` 的用户/内容/交互/时间边界，以及 `evaluation.md` 中 popular、
   random、SVD 的 Recall@10、NDCG@10、HitRate@10。
4. 展示 `artifacts/current.json` 后初始化并启动服务。说明训练失败不会替换可用版本。

## 0:40–1:35 多用户与三路 Feed

1. 登录 `alice`，切换个性化、热门、探索；指出 request_id、模型版本、画像版本、
   item 来源、position、score/explanation。点击“加载更多”，确认无重复。
2. 退出后登录 `bob`，回到个性化 Feed，指出结果与 alice 的可见差异。
3. 登录冷启动账号 `carol`，展示可解释的热门/探索结果和 fallback 信息。
4. 快速输入一次错误密码，证明登录错误态明确；不要在视频暴露 session cookie。

## 1:35–2:20 行为、画像与排序变化

1. 回到 `alice`；点击一个内容、喜欢另一个内容，并对第三个选择“不感兴趣”。
2. 打开“我的画像”，展示事件类型、item、request_id、画像版本及正/负向内容。
3. 重新请求个性化 Feed，展示不感兴趣内容被过滤，或相关排序/解释发生可观察变化。
4. 说明 impression 由服务端在 Feed 响应事务中写入，浏览器不会重复上报。

## 2:20–3:05 Dashboard 与链路诊断

1. 登录 `admin`，打开 Dashboard，展示真实用户、活跃用户、请求、曝光、点击、CTR、
   喜欢、Feed 分布、热门内容和当前模型。
2. 对比行为前记录的数值，刷新后指出实际增量。
3. 粘贴刚才的 request_id，展示请求用户、Feed、模型、返回列表、曝光及行为关联；再
   查询 alice 的画像与最近请求。
4. 可选：用普通用户直接访问管理 API，展示服务端 403，而非只展示隐藏导航。

## 3:05–4:15 强推、下线、恢复与审计

1. 在内容运营中搜索一个在线内容，创建面向 alice 个性化 Feed、当前位置 0、未来
   24 小时有效且原因完整的强推。
2. 登录 alice 并刷新个性化 Feed，指出 `is_forced`、来源和位置。
3. 管理员下线同一内容。再次验证 alice 的三路 Feed，并请求
   `/api/v1/items/{item_id}`，均不能绕过下线。
4. 恢复内容；因为强推仍有效，它应重新出现。展示审计中的管理员、时间、目标、原因
   和前后状态。强调规则优先级 `offline > boost`。

## 4:15–5:00 测试、完成度与风险

1. 展示 `uv run pytest` 的真实退出码与测试摘要，并打开 `docs/VERIFICATION.md`。
2. 展示 `.env.example`、测试账号和 README 干净启动步骤。
3. 如实陈述降级：响应即曝光、占位封面、SQLite、非持久 cursor、无在线训练。
4. 如有失败或未验证项，直接展示为 PENDING/FAILED，不口头宣称完成。

## 录制前核对

- 从干净数据库 seed，不携带上次演示指标；先记录 Dashboard 基线。
- 确认 current model 指针有效，也演练一次缺模型 fallback。
- 预先选择一个在线、可强推且标题易辨认的 item，并记录其 ID。
- 保留每一步 request_id；浏览器缩放 100%，桌面和窄屏均无元素重叠。
- 视频不包含官方原始数据内容、真实密钥、本机隐私路径或未脱敏 cookie。
