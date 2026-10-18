# AICS 评测基线（ai_kefu_test）

说明书场景 47 条回归用例 + LLM-as-judge 判分。与 AICS 主代码**同仓同版本**——
代码与基线一起演进，改动检索/生成链路后跑一遍即可对比。

## 目录内容

| 文件 | 说明 |
|---|---|
| `run_tests_manual.py` | 评测执行器：串行走 AICS testchat 真实链路（`X-LLM-Profile: dev`），逐条判分 |
| `test_cases_manual.json` | 47 条用例（类别 / 子类 / 输入 / 标准答案 / 判分标准 / 难度 / 资料依据页码） |
| `results_manual.json` | **当前基线**（gpt-6-sol 裁判，2026-10-18：32 pass / 9 partial / 6 fail） |
| `results_manual_baseline_deepseek-judge_20261017.json` | 历史基线存档（deepseek 裁判，33/4/10）。**两个裁判分数不可比** |
| `baseline_gpt6sol_20261018.log` | 最近一次基线运行的完整日志（含 partial/fail 差异明细） |
| `AI客服测试用例集*.html` / `AI客服实测结果*.html` | 用例集与实测结果的可读版报告 |

## 运行方式

前置：AICS 服务已启动（本地 dev 默认 `http://127.0.0.1:8000`），
且存在被测员工（默认取 `34e43866643e47e1a5e097386ce1a5ef`，即演示租户的「客服二号」）。

```bash
python run_tests_manual.py
```

配置优先级：进程环境变量 > 同目录 `.env` > 脚本默认。`.env` **不入库**（含 API 密钥），
按需自建：

```ini
# 被测服务（本地 dev 默认 8000，生产 8001）
AICS_BASE=http://127.0.0.1:8000
AICS_EMP_ID=34e43866643e47e1a5e097386ce1a5ef
# 裁判模型：必须与被测异构（自己判自己会分数虚高）
JUDGE_URL=https://api.pateway.ai/v1/chat/completions
JUDGE_KEY=<你的 key>
JUDGE_MODEL=gpt-6-sol
```

47 条串行约 14 分钟（中转 key 并发会触发 429 限流，不能并发）。

## 已知模型参数规范差异

裁判模型换新系模型时注意（`run_tests_manual.py` 的 `_adapt_body_for_model` 已处理）：

- `gpt-6-sol` 等新 GPT 系：不认 `max_tokens`（要 `max_completion_tokens`），
  不认 `temperature=0`（只支持默认值 1）；
- deepseek/glm/kimi 等老模型：只认 `max_tokens`，`temperature=0` 有效。

换裁判 = 换判分标准：**历史基线分数全部作废，必须重建基线后再对比。**
