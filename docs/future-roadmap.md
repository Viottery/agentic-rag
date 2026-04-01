# Future Roadmap

本文档用于整理 Agentic-RAG 下一阶段的发展方向。

当前 roadmap 的核心变化是：

- 不再围绕旧的固定节点链继续加复杂度
- 不再把 skill 继续做成隐藏执行逻辑的 Python 包装
- 逐步把系统重构为 `fast path + planner + execution agent + shell runtime + skill registry + service/API`

## 0. 当前快照

当前系统已经具备一些关键基础：

- 本地 RAG 已有可运行的检索、层级 metadata、混合召回与 semantic reranker
- Tavily 搜索已具备第一版 web retrieval 流程
- `/chat` 已有 planner 驱动的多步执行雏形
- 回答后处理已具备基础 grounding / verification 能力

当前也已经暴露出一个明确的 runtime TODO：

- 对“检索后还需要计算/执行”的问题，planner 可能重复派发近似 RAG 子任务
- 当 `tool_execute` 返回 degraded 或 mock 结果时，planner 目前缺少足够强的熔断/收束逻辑
- 这会导致同一问题在同一 scope 上重复检索，并重复尝试无效 action，形成空转链路

同时，下一阶段还需要补齐另外一个基础能力：

- 系统尚未真正引入 `conversation` 概念
- 上下文仍以单请求状态为主，而不是会话化上下文
- 多会话并发、单会话串行队列、trace 落盘、rolling summary 和长期记忆仍未进入主结构

但这些能力目前仍然偏“节点化工作流”组织，而不是统一 runtime。

下一阶段的目标不是继续堆旧 graph，而是：

- 缩短简单问题路径
- 明确 planner 与 execution agent 的职责边界
- 让 shell 成为 execution agent 的主要执行通道
- 把 skill 收敛成可检索、可索引、可注入 prompt 的 registry
- 把检索和工具能力下沉为独立 service/API

## 1. 近期主目标

近期核心目标有六条：

1. 增加 fast path gate，让简单任务尽量直接输出
2. 将系统重构为 `planner -> atomic subtasks -> execution agent -> planner feedback`
3. 将 skill 从“执行逻辑”重构为“`manifest + prompt package + invocation guide`”
4. 将本地 RAG 打包为独立 service/API，而不是直接暴露给 planner
5. 建立跨 Linux / Windows 的统一调用协议
6. 设计 shell policy engine，给 execution agent 开放 shell 权限但加上风控

此外，有一个应尽快纳入迁移计划的稳定性 TODO：

7. 为 planner 增加重复子任务抑制与 degraded-action 熔断，避免空转或近似死循环

在此基础上，新的上下文主线应加入：

8. 引入 `conversation`、`turn`、`trace`、`summary`、`memory note` 五类上下文对象
9. 引入多会话并发与单会话串行队列
10. 将主图、子图和后处理链逐步迁移到 async 运行时

## 2. 架构重构主线

### 2.1 Fast Path Gate

这是第一优先级之一。

目标：

- 简单问题不再默认进入 planner loop
- 降低平均响应时间与 token 成本
- 保留复杂任务的 planner 能力，但只在真正需要时使用

fast gate 的建议输出：

- `direct_answer`
- `planner_loop`

解释：

- `direct_answer`
  适用于低风险、无需检索、无需工具、无需多步规划的问题
- `planner_loop`
  适用于多步问题、信息缺口明显的问题、需要检索、shell、service 或多个能力的问题

初期不追求复杂微调。
更务实的做法是：

- 小模型判断
- 轻量规则兜底

### 2.2 Planner

新主循环应改成：

```text
planner
  -> subtask decomposition
  -> execution agent
  -> structured results
  -> planner
```

planner 的职责边界必须明确收缩。

Planner 负责：

- 识别信息缺口
- 拆分单一职责子任务
- 选择把子任务交给哪个 execution agent
- 基于返回结果决定继续、补任务、还是进入回答

Planner 可以知道：

- skill 列表
- skill 的功能描述
- skill 的适用场景

Planner 不负责：

- 直接调 skill
- 拼 shell 命令
- 决定底层调用参数细节
- 直接与 service API 交互
- 在 prompt 里持有过多底层实现知识

Planner 后续还需要补齐两个止损能力：

- 重复检索抑制：
  当新子任务与已完成子任务在 rewrite 后 query、route scope、目标实体上高度相似时，不再重复派发
- degraded/mock 执行熔断：
  当 execution result 已明确标记 `degraded=true` 且无法满足 success criteria 时，planner 应优先收束到回答或能力缺口说明，而不是继续重复 action

也就是说，Planner 决定的是“做什么”和“交给谁做”，不是“怎么调用”。

### 2.3 Atomic Subtasks

子任务必须尽量单一职责。

设计原则：

- 可执行
- 可验证
- 可解释
- 不在子任务内部再做一次大 planner

子任务执行端允许保留局部策略，
但不能再膨胀成新的全局调度器。

对于“检索 + 计算”类问题，还需要明确一条规则：

- 如果 calculation/action skill 尚未真实可用，就不应让 planner 通过重复派发同类 action 来“碰运气”
- 这种场景应由 validator 或 planner 明确回报能力缺口，而不是制造额外迭代

### 2.4 Execution Agent

execution agent 是新主线里的关键主体。

职责：

- 接收原子任务
- 检索合适的 skill
- 通过 shell 与本地环境、service 或外部能力交互
- 返回结构化 task result

核心原则：

- 强执行
- 弱规划
- 高可观测
- 高可控

从运行时角度，execution agent 还应满足：

- 保留 async 执行入口
- 支持未来对独立 subtasks 做 fan-out / fan-in
- 将不依赖回答返回的落盘和整理步骤异步化

## 3. Conversation 与上下文系统

### 3.1 为什么现在必须引入 conversation

如果没有 `conversation`，
系统仍然会停留在“单请求工作流”层面。

这会直接限制：

- 短期上下文管理
- 长期记忆沉淀
- 多话题并发
- 同一话题内部顺序一致性
- turn 级 trace 和 summary 的可追踪性

因此下一阶段必须正式引入：

- `conversation`
- `turn`
- `turn trace`
- `turn summary`
- `memory note`

### 3.2 短期上下文

短期上下文建议采用：

- recent turns 滑动窗口
- rolling summary
- active task snapshot
- recent turn summaries

也就是说，进入模型的上下文不应只是最近原始消息，
而应是：

- 最近轮次原文
- 压缩后的会话摘要
- 最近关键执行摘要
- 当前未完成任务快照

### 3.3 长期记忆

长期记忆建议先使用：

- SQLite 主存储
- markdown/json 文件镜像

长期记忆不保存原始对话全文，
而保存整理后的 memory notes。

### 3.4 原始 trace 的位置

planner / execution / validator 的输出应被记录，
但不应原样全部回注到 prompt。

建议分层：

- 原始 trace：存档
- turn summary：短期上下文输入
- memory note：长期记忆输入

## 4. Async 运行时

### 4.1 总原则

系统设计从现在开始应默认保留异步性。

需要同时满足：

- 多个 conversation 可以并发
- 同一 conversation 的多个 turn 必须串行
- 不依赖顺序的子步骤尽量异步

### 4.2 Conversation Queue Manager

建议新增 `conversation queue manager`：

- 每个 conversation 一个内部队列
- 同会话只允许一个活动 turn
- 不同会话可以在全局 worker pool 中并发

### 4.3 会话过程异步调用

`/chat` 后续应支持：

- 同步等待结果
- 后台提交 turn 并异步完成
- 后续可扩展 SSE / websocket streaming

### 4.4 可异步化的后处理

以下步骤适合做 write-behind：

- trace 落盘
- turn summary 更新
- memory candidate 提取
- memory note 写入

## 5. Skill Registry

### 3.1 为什么要引入 skill registry

skill 是新的核心抽象之一，
但它不再等于“执行代码”。

原因：

- 降低 planner 需要消费的底层实现细节
- 统一本地 RAG、搜索、工具调用的使用说明
- 让 execution agent 可以检索和引用稳定的调用知识
- 后续扩展新能力时，不必不断修改 planner 提示词结构

### 3.2 skill 应该是什么

skill 更适合作为：

- 调用方法描述
- 提示词包装
- 输入输出约定
- 平台适配信息
- 使用建议

可以把 skill 理解为：

- `manifest + prompt package + invocation guide`

skill 本身不提供服务。

### 3.3 skill 至少应包含的字段

建议每个 skill 至少包含：

- `skill_id`
- `name`
- `summary`
- `when_to_use`
- `when_not_to_use`
- `input_schema`
- `output_schema`
- `prompt_files`
- `service_binding`
- `platform_invocation`
- `tags`
- `examples`

### 3.4 第一优先级 skill

第一优先级仍然是本地知识检索相关 skill。

但这里的重点已不是“把本地 RAG 代码塞进 skill”，
而是：

- 为本地 RAG service 提供一个稳定 skill entry
- 让 execution agent 可以通过 skill 获得调用方法与 prompt 组织方式

## 6. Service / API 主线

### 4.1 为什么要 service 化

真正的能力应由独立服务提供。

原因：

- 避免 planner 或 skill 直接耦合底层实现
- 便于后续独立扩展、替换或部署
- 便于 shell runtime 和 cross-platform invocation 收敛到统一入口

### 4.2 本地 RAG service 的定位

本地 RAG 近期的效果目标没有变化：

- 层次化路由
- 多路召回
- semantic reranker
- 更稳定的 evidence selection

变化在于：

- 它不再作为 planner-facing 节点暴露
- 它也不再被写成 skill 本体
- 它应下沉为独立 service/API

service 内部可以封装：

```text
query normalize
  -> hierarchy routing
  -> scoped hybrid recall
  -> candidate fusion
  -> semantic rerank
  -> evidence packaging
```

### 4.3 后续 service 方向

后续可以逐步补齐：

- `local_rag_service`
- `web_search_service`
- `tool_execution_service`
- `file_service`
- `code_execution_service`

但近期重点仍然是先把本地 RAG service 做扎实。

## 7. Shell Runtime 与跨平台策略

### 5.1 shell 是主要执行通道

execution agent 应保留较强的命令行能力。

shell 不应被视为例外能力，
而应被视为：

- execution agent 与本地环境、外界信息和能力交互的主要 substrate

### 5.2 不强行统一 shell

建议保留：

- Linux: `bash`
- Windows: `powershell`

不要为了统一而抹掉各自的生态优势。

### 5.3 统一调用协议，而不是统一命令字符串

skill 不应只存一条 bash 命令。

更合理的做法是：

- skill 提供平台感知的调用模板
- execution agent 根据当前 OS 选择相应模板

### 5.4 尽量通过 CLI + 文件交换

为了避免 bash / powershell 的引号和 JSON 转义问题，
建议统一使用：

- 输入文件
- 输出文件
- 统一 CLI 调用入口

例如：

```text
write request.json
  -> run skill client
  -> read result.json
```

## 8. Shell 风控

如果 shell 是主要交互通道，
风控必须是 runtime 级能力，而不是 prompt 级提醒。

建议至少包括以下层次。

### 6.1 命令分级

将命令按风险分层，例如：

- `L0` 只读本地
- `L1` 受控写入
- `L2` 网络读取
- `L3` 高风险写操作
- `L4` 危险系统操作

默认 execution agent 只拿到低到中风险权限。

### 6.2 作用域限制

限制：

- 文件系统访问范围
- 网络访问范围
- 可继承环境变量
- 可用可执行程序

### 6.3 审计

每条 shell 命令都应记录：

- task id
- agent id
- command
- cwd
- start/end time
- exit code
- stdout/stderr 摘要

### 6.4 破坏性命令拦截

需要 runtime 层直接阻止高风险命令模式，
而不是依赖 LLM 自觉避免。

### 6.5 资源限制

每条 shell 命令都应有：

- timeout
- max output
- max subprocesses
- resource ceilings

## 9. Grounding Validator

validator 的方向没有变化，
仍然应从“轻量 citation 检查器”逐步升级为：

- 工具结果与答案关联性检查器
- 可解释性增强器
- 幻觉抑制器

但在当前迁移顺序里，
validator 不是第一优先级。

当前更高优先级的是：

- execution agent 边界
- shell runtime
- skill registry
- service 化
- shell 风控

## 10. 本地 RAG 主线

### 8.1 当前继续推进的内容

- 稳定 `hierarchy routing`
- 调整 hybrid retrieval 的候选规模
- 让 semantic reranker 只打最值得重排的候选
- 优化 evidence packaging，便于后续 validator 消费

### 8.2 当前明确不优先推进

- 父子索引 / 多层索引对象建设保留在 TODO
- 不作为这一轮 execution-agent runtime 与 service 化的阻塞项

## 11. 迁移顺序

建议按以下顺序推进：

1. 先更新文档与架构认知
2. 引入 `conversation_id` 与 conversation-aware `/chat`
3. 增加 turn / trace 持久化
4. 增加 sliding window + rolling summary
5. 增加长期 memory note 写入
6. 引入 fast gate
7. 明确 planner 与 execution agent 的职责分离
8. 定义统一 skill registry schema
9. 将本地 RAG 重构为独立 service/API
10. 设计统一 skill 调用 CLI
11. 增加 Linux / Windows 平台适配模板
12. 建立 conversation queue manager
13. 将主图和子图切向 async 运行
14. 建立 shell policy engine
15. 最后再升级 grounding validator
16. 逐步清理旧 graph 中的硬编码节点链

## 12. 长远目标

长远来看，这个项目的目标不是停留在“一个会检索的 agentic RAG demo”，
而是成为一个更完整的 agent runtime：

- 有 conversation 与 context system
- 有 fast path
- 有 planner loop
- 有 execution agent
- 有 shell runtime
- 有 skill registry
- 有 service/API
- 有 grounding validator
- 有未来可扩展的工具与执行边界

长期应该坚持的原则是：

- 简单问题走短链路
- 复杂问题走结构化循环
- planner 只做全局决策
- execution agent 只做强执行
- skill 只做调用知识组织
- service 只做真实能力提供
- shell 既开放能力，也有 runtime 风控
