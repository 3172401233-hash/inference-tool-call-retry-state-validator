# inference-tool-call-retry-state-validator
验证工具调用幂等、重试和取消状态

## 背景

模型发起工具调用后连接中断，客户端重试请求。工具服务可能已执行但响应
丢失：直接重跑会产生重复副作用，复用旧响应又可能对应不同参数。本仓库
提供一个纯状态机验证器（Python 3.8.2 兼容），根据会话事件流计算下一步
动作。

**本模块不调用任何真实工具**，只对事件日志做确定性判定。

## 输入

`tool_call_state.evaluate()` 接收：

- `conversation`：有序事件日志，事件类型包括 `request`、`started`、
  `receipt`、`response_delivered`、`response_lost`、`retry`、`cancel`
- `tool_call_id`：待判定的调用 ID
- `digest`：当前参数的规范摘要（`params_digest()`，SHA-256，键序无关）
- `idempotency_key`：客户端当前出示的幂等键
- `receipt`：已知执行回执（含 `idempotency_key`、`params_digest`、`result`）

## 输出（`Verdict`）

- `status`：`pending` / `in_flight` / `response_lost` / `completed` /
  `cancelled` / `conflict`
- `action`：`execute` / `reuse_result` / `recover_result` / `wait` /
  `reject_param_conflict` / `ack_cancel` / `ignore_cancel` / `noop`
- `should_execute`：是否允许真正执行（同一调用至多为 True 一次）
- `reusable_result`：可复用/可恢复的结果
- `conflict_evidence`：冲突证据（参数漂移、重复取消、不可中断取消等）

## 保证

- 相同调用（`tool_call_id` + 幂等键 + 参数摘要）只执行一次，重试复用回执
- 参数或幂等键漂移时拒绝复用旧结果，返回 `conflict` 与漂移证据
- 工具已执行但响应丢失时返回 `response_lost` + `recover_result`，凭幂等键
  恢复结果而不是重跑
- 取消不能打断已开始的不可逆工具（`ignore_cancel`）；重复取消幂等

## 测试

```bash
python -m unittest
```

覆盖响应丢失、参数漂移、重复取消、执行去重等场景。
