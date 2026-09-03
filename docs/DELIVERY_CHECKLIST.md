# Section 3.3 Delivery Checklist

本清单严格按题目 3.3 顺序给出交付入口。`VERIFIED` 只表示仓库中有对应实现且存在
本地实测证据；远端、外部服务和视频不会用本地结果冒充。

| # | 交付项 | 状态 | 入口与证据 |
|---:|---|---|---|
| 1 | 源码仓库 | IN PROGRESS | <https://github.com/BlossomRa1n/Yahaha>；本地已有 6 个有效提交，当前改动将按功能拆分提交；数据、模型、数据库、`.env` 和题目 DOCX 均不得入库。 |
| 2 | Demo 地址 | VERIFIED (LOCAL) | `http://127.0.0.1:8000/`；README 包含完整本地启动方式、端口、访问路径和账号。无线上部署地址。 |
| 3 | 启动命令 | VERIFIED | README 的“前置条件”“初始化与启动”“测试”覆盖依赖、数据库、后端/同源前端、CPU smoke 和训练命令；Redis 是可选加速。 |
| 4 | 数据处理脚本 | VERIFIED | `recsys/data.py`、`recsys/pipeline.py`；README 记录官方来源、目录、时间切分、负采样和一键命令。 |
| 5 | 模型与评估报告 | VERIFIED | `recsys/model.py`、`recsys/deep.py`、`recsys/vision.py`；README 给出 baseline、DSSM/DeepFM、多模态、checkpoint、早停、统一 sampled-negative 口径、至少三项指标和 Badcase。模型大文件不入库，由命令本地生成。 |
| 6 | 测试账号与种子数据 | VERIFIED | `uv run python -m app.cli init-db --items data/processed/items.csv --reset`；`alice`、`bob`、`carol` 和 `admin` 账号见 README。 |
| 7 | 数据库与 API 文档 | VERIFIED | `docs/API.md` 及 `app/db.py`；覆盖核心表、登录、Feed、事件、Dashboard、强推、下线、告警和训练任务。 |
| 8 | 系统设计文档 | VERIFIED | `docs/ARCHITECTURE.md`；覆盖离线/在线流、七路召回、DeepFM 排序、混排、fallback、发布、权限和失败恢复。 |
| 9 | README 与环境变量 | VERIFIED | `README.md`、`.env.example`；真实 `.env` 被忽略，变量用途和降级行为有说明。 |
| 10 | 完成度说明 | VERIFIED | README 的“完成度声明”区分已验证、Mock、未完成、风险和一周迭代计划。 |
| 11 | 演示视频 | **PENDING (REQUIRED)** | `docs/DEMO.md` 已准备 3-5 分钟脚本；仍需人工录制、上传并把可访问链接填入本行。 |
| 12 | 测试与验证证据 | VERIFIED | `docs/VERIFICATION.md`；最近全仓结果为 102 passed，并记录真实数据、API、浏览器和失败修复证据。 |
| 13 | AI 协作记录 | VERIFIED | `docs/AI_COLLABORATION.md`；记录工具、关键 prompt、人工 review、测试和典型修复。 |

## 提交前安全门槛

- 暂存区不得包含 `data/`、`artifacts/`、模型/checkpoint、数据库、`.env`、原始封面或题目文档。
- 每个提交必须表达单一可说明的功能边界，提交前运行 `git diff --cached --check`。
- 推送后以远端 `main` 的 commit 列表和 `git ls-remote` 为准，再把第 1 项更新为 `VERIFIED`。

## 唯一必选阻塞项

代码推送成功后，3.3 中仍需要候选人本人完成的是演示视频。仓库脚本不能替代视频链接，
也不能把自动化测试录屏冒充两个用户、行为变化和运营操作的完整演示。
