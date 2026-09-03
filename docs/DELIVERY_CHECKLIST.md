# Section 3.3 Delivery Checklist

本清单严格按题目 3.3 顺序给出交付入口。`VERIFIED` 只表示仓库中有对应实现且存在
本地实测证据；远端、外部服务和视频不会用本地结果冒充。

| # | 交付项 | 状态 | 入口与证据 |
|---:|---|---|---|
| 1 | 源码仓库 | VERIFIED | <https://github.com/BlossomRa1n/Yahaha>；`main` 已推送，加入视频链接前共有 11 个有效提交；数据、模型、数据库、`.env` 和题目 DOCX 均未入库。 |
| 2 | Demo 地址 | VERIFIED (LOCAL) | `http://127.0.0.1:8000/`；README 包含完整本地启动方式、端口、访问路径和账号。无线上部署地址。 |
| 3 | 启动命令 | VERIFIED | README 的“前置条件”“初始化与启动”“测试”覆盖依赖、数据库、后端/同源前端、CPU smoke 和训练命令；Redis 是可选加速。 |
| 4 | 数据处理脚本 | VERIFIED | `recsys/data.py`、`recsys/pipeline.py`；README 记录官方来源、目录、时间切分、负采样和一键命令。 |
| 5 | 模型与评估报告 | VERIFIED | `recsys/model.py`、`recsys/deep.py`、`recsys/vision.py`；README 给出 baseline、DSSM/DeepFM、多模态、checkpoint、早停、统一 sampled-negative 口径、至少三项指标和 Badcase。模型大文件不入库，由命令本地生成。 |
| 6 | 测试账号与种子数据 | VERIFIED | `uv run python -m app.cli init-db --items data/processed/items.csv --reset`；`alice`、`bob`、`carol` 和 `admin` 账号见 README。 |
| 7 | 数据库与 API 文档 | VERIFIED | `docs/API.md` 及 `app/db.py`；覆盖核心表、登录、Feed、事件、Dashboard、强推、下线、告警和训练任务。 |
| 8 | 系统设计文档 | VERIFIED | `docs/ARCHITECTURE.md`；覆盖离线/在线流、七路召回、DeepFM 排序、混排、fallback、发布、权限和失败恢复。 |
| 9 | README 与环境变量 | VERIFIED | `README.md`、`.env.example`；真实 `.env` 被忽略，变量用途和降级行为有说明。 |
| 10 | 完成度说明 | VERIFIED | README 的“完成度声明”区分已验证、Mock、未完成、风险和一周迭代计划。 |
| 11 | 演示视频 | DELIVERED | [江雨鸿-视频演示.mp4（夸克网盘）](https://pan.quark.cn/s/81e4e62f191a)；2026-09-04 分享落地页 HTTP 200。 |
| 12 | 测试与验证证据 | VERIFIED | `docs/VERIFICATION.md`；最近全仓结果为 102 passed，并记录真实数据、API、浏览器和失败修复证据。 |
| 13 | AI 协作记录 | VERIFIED | `docs/AI_COLLABORATION.md`；记录工具、关键 prompt、人工 review、测试和典型修复。 |

## 提交前安全检查

- 暂存区不得包含 `data/`、`artifacts/`、模型/checkpoint、数据库、`.env`、原始封面或题目文档。
- 每个提交必须表达单一可说明的功能边界，提交前运行 `git diff --cached --check`。
- 推送后以远端 `main` 的 commit 列表和 `git ls-remote` 为准，再把第 1 项更新为 `VERIFIED`。

## 收尾状态

源码、文档、测试账号、验证证据和演示视频链接均已交付。外部视频由候选人负责保持
分享有效并允许评审访问；仓库不保存或重新分发视频文件。
