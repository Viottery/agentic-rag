# Context And Memory Architecture

本文档定义项目下一阶段的上下文系统。

目标不是简单“保留更多聊天记录”，
而是把系统升级为：

- 有 `conversation` 边界
- 有短期上下文管理
- 有长期记忆沉淀
- 有执行过程可观测记录
- 有多会话并发与单会话串行队列
- 有未来可扩展的异步运行时

## 1. 设计目标

上下文系统需要同时满足五个目标：

1. 区分不同对话组，避免不同任务线程互相污染
2. 控制 prompt 上下文体积，避免历史无限膨胀
3. 让系统记住稳定事实、偏好、约束和决策
4. 让 planner / execution agent / validator 的高层输出也能被沉淀
5. 从一开始就保留异步性，支持多会话并发与同会话排队

## 2. 核心对象

### 2.1 Conversation

`conversation` 是系统的一级上下文边界。

它代表：

- 一个独立任务线程
- 一个独立主题空间
- 一个独立短期上下文窗口
- 一个独立长期记忆子空间

建议字段：

- `conversation_id`
- `title`
- `status`
- `created_at`
- `updated_at`
- `last_turn_id`
- `rolling_summary`
- `active_task_snapshot`

### 2.2 Turn

`turn` 代表会话中的一轮交互。

一轮 turn 不只是：

- 用户输入
- 系统输出

还应包含这一轮的执行结果和执行摘要。

建议字段：

- `turn_id`
- `conversation_id`
- `request_id`
- `user_message`
- `assistant_answer`
- `status`
- `created_at`
- `started_at`
- `finished_at`

### 2.3 Execution Trace

原始执行记录用于调试、回放和后续总结，
但不应原样全部塞回 prompt。

建议字段：

- `turn_id`
- `fast_path_decision`
- `planner_control`
- `subtasks`
- `execution_results`
- `trace_summary`
- `iteration_count`
- `elapsed_seconds`
- `status`

### 2.4 Turn Summary

`turn_summary` 是从原始执行过程压缩出来的“本轮发生了什么”。

它是短期上下文和长期记忆的主要输入，而不是原始 trace。

建议字段：

- `turn_id`
- `conversation_id`
- `goal`
- `actions_taken`
- `key_findings`
- `limitations`
- `open_loops`
- `memory_candidates`

### 2.5 Memory Note

长期记忆使用“整理后的 note”，而不是直接保存原始对话全文。

建议字段：

- `memory_id`
- `scope`
- `conversation_id`
- `kind`
- `title`
- `content`
- `tags`
- `source_turn_ids`
- `importance`
- `created_at`
- `updated_at`

## 3. 记忆分层

### 3.1 短期上下文

短期上下文采用：

- `recent_turns_window`
- `rolling_summary`
- `active_task_snapshot`
- `recent_turn_summaries`

也就是说，进入模型的上下文不是整段原始历史，
而是：

```text
system prompt
  + rolling summary
  + recent turn window
  + recent turn summaries
  + active task snapshot
  + current user message
```

短期上下文的实现原则：

- 最近轮次使用滑动窗口
- token 超限时优先压缩旧 turn
- 旧 turn 的关键信息进入 rolling summary
- 当前未完成任务状态进入 active task snapshot

### 3.2 长期记忆

长期记忆不直接保存所有聊天内容，
而保存整理后的稳定信息。

建议分成两类：

- `conversation memory`
  只在当前 conversation 内召回
- `global memory`
  跨 conversation 共享

建议 `kind` 分类：

- `fact`
- `preference`
- `decision`
- `constraint`
- `project_context`
- `open_question`
- `todo`

### 3.3 原始 trace 的位置

原始 `intermediate_steps`、planner 输出、execution trace 需要保存，
但默认不直接进入长期记忆和后续 prompt。

原则：

- 原始 trace：存档
- turn summary：进入短期上下文
- memory notes：进入长期记忆

## 4. 会话内哪些信息应被记住

除了用户消息和答案，
系统还应记录高层执行信息。

适合进入 turn summary / memory candidates 的内容包括：

- fast gate 的主决策
- planner 的任务拆分结论
- execution agent 实际调用了哪些 skill / service
- 哪些 task 成功、失败、degraded
- validator 给出的关键限制
- 当前轮次得到的稳定结论
- 当前轮次形成的 TODO、决策或约束

不建议直接长期保留的内容包括：

- 冗长原始 thought
- 每一步重复 observation
- prompt 片段本身
- 大量低层 debug 文本

## 5. 存储设计

初版建议使用：

- SQLite 作为主存储
- markdown / json 文件镜像作为人工可读输出

目录建议：

```text
data/context/
  conversations.db
  notes/
    global/
    conversations/<conversation_id>/
```

### 5.1 建议表

`conversations`

- `conversation_id`
- `title`
- `status`
- `rolling_summary`
- `active_task_snapshot_json`
- `last_turn_id`
- `created_at`
- `updated_at`

`turns`

- `turn_id`
- `conversation_id`
- `request_id`
- `user_message`
- `assistant_answer`
- `status`
- `created_at`
- `started_at`
- `finished_at`

`turn_traces`

- `turn_id`
- `conversation_id`
- `fast_path_mode`
- `planner_control_json`
- `subtasks_json`
- `execution_results_json`
- `trace_summary`
- `iteration_count`
- `elapsed_seconds`
- `status`
- `created_at`

`turn_summaries`

- `turn_id`
- `conversation_id`
- `goal`
- `actions_taken_json`
- `key_findings_json`
- `limitations_json`
- `open_loops_json`
- `memory_candidates_json`
- `created_at`

`memory_notes`

- `memory_id`
- `scope`
- `conversation_id`
- `kind`
- `title`
- `content`
- `tags_json`
- `source_turn_ids_json`
- `importance`
- `created_at`
- `updated_at`

`turn_jobs`

- `job_id`
- `turn_id`
- `conversation_id`
- `status`
- `enqueue_time`
- `start_time`
- `finish_time`
- `worker_id`
- `error`

## 6. 异步运行时

### 6.1 总体原则

从现在开始，运行时应默认保留异步性。

需要同时满足：

- 多个 conversation 可以并发处理
- 同一个 conversation 内多个 turn 必须保持顺序
- 不依赖串行的步骤应尽量异步执行

### 6.2 Conversation Queue Manager

建议引入 `conversation queue manager`。

职责：

- 为每个 `conversation_id` 维护一个独立队列
- 保证同一 conversation 同时只处理一个活动 turn
- 允许不同 conversation 在全局 worker pool 中并发运行

逻辑模型：

```text
POST /chat
  -> persist turn + job
  -> enqueue job into conversation queue
  -> queue manager schedules by conversation_id
  -> global worker pool executes active conversations concurrently
```

### 6.3 单会话串行

同一会话中的 turn 默认串行。

原因：

- 保证上下文顺序一致
- 保证 rolling summary 不被并发覆盖
- 保证 memory writeback 顺序一致
- 保证 planner 看到的是稳定的最近状态

因此：

- conversation 内部用队列维护
- conversation 外部允许并发

### 6.4 多会话并发

多个 conversation 可以同时执行：

- 不同用户的不同会话
- 同一用户的不同话题线程
- 后台整理与前台问答

建议由全局 semaphore / worker pool 限制总体并发量。

### 6.5 会话过程异步调用

`/chat` 应支持两种模式：

- `wait`
  提交后等待当前 turn 完成并返回结果
- `background`
  提交后立刻返回 `turn_id + job_id`，前端轮询或订阅结果

长期建议支持：

- `stream`
  通过 SSE / websocket 回传阶段性事件

## 7. Turn 内部异步

单个 turn 内部也应尽量异步化。

### 7.1 可以并行的步骤

未来适合并发的步骤包括：

- 独立子任务的 execution
- 多路 route fan-out 后的 retrieval
- 多个 service 请求
- turn 结束后的 trace 落盘、summary 更新、memory candidate 提取

### 7.2 不应并行的步骤

这些步骤仍应保持顺序：

- fast gate 之后的主路径选择
- 同一 planner 轮次中的 stop/continue 决策
- 同一 conversation 的 rolling summary 提交
- 同一 turn 的最终 answer 提交

### 7.3 执行图方向

主图和子图都应优先暴露 async 入口。

建议：

- FastAPI route 使用 async handler
- 主图优先走 `ainvoke`
- 子任务执行图优先走 `ainvoke_subtask_graph`
- execution agent 在需要时对独立 subtasks 做 `asyncio.gather`

## 8. API 设计方向

建议逐步收敛为以下接口：

`POST /conversations`

- 创建新 conversation

`GET /conversations/{conversation_id}`

- 获取 conversation 元数据和当前状态

`POST /chat`

请求建议包含：

- `conversation_id`
- `question`
- `mode`
- `debug`

其中：

- `mode=wait`
  等待 turn 完成
- `mode=background`
  立即返回 job 信息

`GET /conversations/{conversation_id}/turns/{turn_id}`

- 查询单轮结果

`GET /conversations/{conversation_id}/jobs/{job_id}`

- 查询队列任务状态

`GET /conversations/{conversation_id}/events`

- 可选的 SSE / websocket 事件流

## 9. 模块划分建议

建议增加以下模块：

- `app/context/store.py`
  conversation / turn / trace / memory 的 SQLite 读写
- `app/context/loader.py`
  组装短期上下文与长期记忆召回结果
- `app/context/summarizer.py`
  生成 turn summary 与 rolling summary
- `app/context/memory_writer.py`
  提取并提交 memory notes
- `app/runtime/queue_manager.py`
  conversation 队列与 worker pool
- `app/runtime/job_runner.py`
  真正执行一个 turn 的 async runner

## 10. 接入当前系统的顺序

建议按以下顺序接入：

1. 引入 `conversation_id`
2. 将 `/chat` 改成显式 conversation-aware
3. 增加 turn / trace 持久化
4. 增加 sliding window + rolling summary
5. 增加 turn summary 生成
6. 增加长期 memory note 写入
7. 引入 conversation queue manager
8. 将主图和子图逐步切到 async 路径
9. 最后再做并发 subtasks 和 streaming

## 11. 初版不追求的内容

当前不必一开始就做：

- 记忆 embedding 检索
- 自动永久保存每一段原始 trace
- 同一 conversation 内并发处理多个 turn
- 复杂的记忆冲突解决
- 过度自动化的 memory writeback

初版更重要的是：

- 会话边界清晰
- 短期上下文可控
- 长期记忆可沉淀
- 异步运行时边界明确
- 多会话并发、单会话串行
