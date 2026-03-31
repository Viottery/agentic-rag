# Agent Runtime Architecture

本文档描述项目下一阶段的目标架构。

这份文档不再把系统理解为一条固定的
`planner -> dispatcher -> query_refiner -> rag/search -> answer -> checker`
串行链路，而是改为：

- `fast path` 优先处理简单问题
- `planner loop` 处理复杂任务
- `skill executor` 负责单一职责执行
- `validator` 负责结果与答案的关联性、可解释性和幻觉抑制

## 1. 总入口

```text
Client
  -> POST /chat
  -> FastAPI route
  -> build initial state
  -> fast path gate
  -> direct answer or planner loop
  -> return final response
```

对应代码入口仍然是：

- `app/api/routes/chat.py`
- `app/agent/graph.py`

但 graph 的职责将从“固定节点编排”逐步转向“runtime orchestration”。

## 2. 新的主流程

### 2.1 Fast Path

适用目标：

- 简单问题尽快返回
- 不需要本地 RAG / 搜索 / 工具
- 不需要多步决策
- 不值得进入 planner loop

目标链路：

```text
user input
  -> fast gate
  -> direct answer model
  -> return
```

可选保留一个极轻量的 sanity check，但它不应演化成新的重链路。

### 2.2 Normal Path

复杂问题走统一的 planner loop：

```text
user input
  -> fast gate
  -> planner
  -> subtask decomposition
  -> skill executor(s)
  -> structured task results
  -> planner
  -> validator
  -> answer synthesizer
  -> validator
  -> END or back to planner
```

目标不是把所有未来任务都提前写成固定 workflow，
而是建立一套稳定的 runtime 边界：

- planner 决定“现在做什么”
- skill executor 决定“如何完成当前原子任务”
- validator 决定“结果是否真的支撑答案”

## 3. 核心设计原则

### 3.1 简单问题不进大循环

系统应该优先判断：

- 是否可以直接回答
- 是否只需要一次 skill 调用
- 是否必须进入 planner loop

建议 fast gate 的最小决策集为：

- `direct_answer`
- `single_skill`
- `planner_loop`

这样可以避免简单问题也被迫经历多轮 planner、检索、校验。

### 3.2 Planner 只做全局决策

Planner 的职责应收敛为：

- 识别当前信息缺口
- 拆分单一职责子任务
- 选择每个子任务的 executor / skill
- 根据执行结果决定继续、补任务或进入回答

Planner 不负责：

- 子任务内部 query rewrite 细节
- 本地检索路由细节
- rerank 细节
- 工具参数级微调
- 最终引用映射细节

一句话说，Planner 决定：

- `what to do`
- `who should do it`
- `what is still missing`

而不决定：

- `how exactly to do it`

### 3.3 子任务必须单一职责

子任务设计原则：

- 原子化
- 可验证
- 输入输出明确
- 尽量避免内部再做 planner 式决策

子任务 executor 允许有“局部策略”，但不允许有“全局规划”。

例如：

- `local_kb_retrieve` 内部可以做 query normalize、scope route、hybrid retrieval、rerank
- 但它不应自己决定“是否改去搜索”或“是否继续拆任务”

这些判断必须回到 Planner。

## 4. Skill Runtime

### 4.1 为什么要用 skill 抽象

系统的核心能力应逐步收敛为 skill，而不是让 planner 直接面向大量底层实现细节。

这样做的价值：

- 降低 planner prompt 的硬编码能力描述
- 统一本地 RAG、搜索、工具调用的执行协议
- 让执行结果更容易被 validator 和 planner 消费
- 后续可平滑扩展更多 executor

### 4.2 当前最重要的 skill

近期最重要的 skill 是：

- `local_kb_retrieve`

它应把当前本地 RAG 的内部流程封装起来：

```text
query normalize
  -> hierarchy routing
  -> scoped hybrid recall
  -> candidate fusion
  -> semantic rerank
  -> evidence selection
  -> structured result
```

上层只看到：

- 输入：任务目标、query、可选 scope hint
- 输出：summary、evidence、sources、route trace、confidence、failure reason

而不需要感知内部到底有多少检索步骤。

后续可以并列增加：

- `web_search_retrieve`
- `tool_execute`
- `file_read`
- `code_run`

## 5. Validator 的新定位

validator 不再只是“引用覆盖率检查器”。

它更适合作为：

- `grounding validator`

职责包括：

- 验证子任务执行结果是否真的满足任务目标
- 验证最终答案是否被已有结果支持
- 标记强支持 / 弱支持 / 无支持结论
- 给 planner 返回下一轮可执行反馈
- 强化答案的可解释性
- 尽量消除 hallucination

它关注的不只是“有没有 citation”，而是：

- 结果和答案是否相关
- 证据是否足够支撑结论
- 哪些部分超出了执行结果边界

## 6. State 设计方向

下一阶段 state 应逐步围绕这些对象重构：

- `request`
- `fast_path_decision`
- `planner_state`
- `subtasks`
- `skill_results`
- `evidence`
- `answer`
- `validation`
- `trace`

相比当前 state，重点变化是：

- 降低固定节点痕迹
- 强化 task / skill / validation 三类结构化对象
- 让 planner 消费的是“结果对象”，而不是底层节点临时状态

## 7. 延迟与成本控制

这套架构的直接目标之一，就是缩短简单问题链路并控制复杂问题成本。

### 7.1 Fast Path 控制成本

简单问题直接输出，避免：

- planner 多轮循环
- 多次 query rewrite
- 无必要的检索与 rerank
- validator 重链路

### 7.2 Normal Path 控制成本

复杂任务的优化方向应集中在：

- 降低 planner 回合数
- 让子任务更原子，减少重复工作
- 让 skill 内部做强执行，而不是频繁回到 planner
- 控制 reranker 候选规模
- 仅在真正需要时追加 validator / 补任务

## 8. 近期迁移路径

建议按以下顺序迁移：

1. 增加 fast gate，将简单问题从主循环中剥离
2. 将本地 RAG 封装为 `local_kb_retrieve` skill
3. 将现有 `rag/search/action` 统一到 skill executor 协议
4. 重写 planner 输出 schema，改为“子任务 + executor/skill”
5. 将 verifier / checker 逐步重构为 grounding validator
6. 逐步减少旧串行 workflow 的硬编码分支

## 9. 当前明确不优先推进的事项

- 父子索引 / 多层索引对象建设保留在 TODO
- 不优先继续扩张旧 graph 上的固定节点链
- 不优先把复杂性继续堆到 planner prompt 上

当前优先级更高的是：

- skill 抽象
- planner loop 重构
- semantic reranker 落地与调优
- grounding validator 的职责重构
