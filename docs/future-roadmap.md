# Future Roadmap

本文档用于整理 Agentic-RAG 下一阶段的发展方向。

当前 roadmap 的核心变化是：

- 不再围绕旧的固定节点链继续加复杂度
- 逐步把系统重构为 `fast path + planner loop + skill runtime + grounding validator`

## 0. 当前快照

当前系统已经具备一些关键基础：

- 本地 RAG 已有可运行的检索、层级 metadata、混合召回与 semantic reranker
- Tavily 搜索已具备第一版 web retrieval 流程
- `/chat` 已有 planner 驱动的多步执行雏形
- 回答后处理已具备基础 grounding / verification 能力

但这些能力目前仍然偏“节点化工作流”组织，而不是统一 skill runtime。

下一阶段的目标不是继续堆旧 graph，而是：

- 缩短简单问题路径
- 降低 planner prompt 负担
- 强化 skill 执行边界
- 把验证能力提升为 grounding validator

## 1. 近期主目标

近期核心目标有五条：

1. 增加 fast path gate，让简单任务尽量直接输出
2. 将本地 RAG 打包为统一 skill 接口
3. 将系统重构为 `planner -> atomic subtasks -> skill execution -> planner feedback`
4. 将验证能力从轻量 citation 检查升级为 grounding validator
5. 在新架构下继续提升本地 RAG 检索质量与 reranker 稳定性

## 2. 架构重构主线

### 2.1 Fast Path Gate

这是第一优先级之一。

目标：

- 简单问题不再默认进入 planner loop
- 降低平均响应时间与 token 成本
- 保留复杂任务的 planner 能力，但只在真正需要时使用

fast gate 的建议输出：

- `direct_answer`
- `single_skill`
- `planner_loop`

解释：

- `direct_answer`
  适用于低风险、无需检索、无需工具、无需多步规划的问题
- `single_skill`
  适用于明显可以由单个 skill 解决的问题，例如一次本地检索
- `planner_loop`
  适用于多步问题、信息缺口明显的问题、需要多个 skill 的问题

初期不追求复杂微调。
更务实的做法是：

- 小模型判断
- 轻量规则兜底

### 2.2 Planner Loop

新主循环应改成：

```text
planner
  -> subtask decomposition
  -> skill executor(s)
  -> structured results
  -> planner
```

planner 的职责边界必须明确收缩。

Planner 负责：

- 识别信息缺口
- 拆分单一职责子任务
- 选择 skill / executor
- 决定继续、补任务、还是进入回答

Planner 不负责：

- 子任务内部 query rewrite 细节
- 检索路由细节
- rerank 细节
- 工具参数级细节
- 最终 grounding 的具体判定细节

也就是说，Planner 决定的是“做什么”，不是“怎么做”。

### 2.3 Atomic Subtasks

子任务必须尽量单一职责。

设计原则：

- 可执行
- 可验证
- 可解释
- 不在子任务内部再做一次大 planner

子任务 executor 允许保留局部策略，
但不能再膨胀成新的全局调度器。

## 3. Skill Runtime

### 3.1 为什么要引入 skill

skill 是新的核心抽象。

原因：

- 降低 planner 需要消费的底层实现细节
- 统一本地 RAG、搜索、工具调用的接口
- 让 validator 消费一致的结果对象
- 后续扩展新能力时，不必不断修改 planner 提示词结构

### 3.2 第一优先级 skill：`local_kb_retrieve`

本地 RAG 应尽快被打包为单一 skill。

内部可以封装：

```text
query normalize
  -> hierarchy routing
  -> scoped hybrid recall
  -> candidate fusion
  -> semantic rerank
  -> evidence packaging
```

但对 planner 暴露的接口应尽量简单：

- 输入：
  - `task_goal`
  - `query`
  - `scope_hint`
  - `top_k`
- 输出：
  - `summary`
  - `evidence`
  - `sources`
  - `route_trace`
  - `confidence`
  - `failure_reason`

这样 planner 不需要了解 `rag_router`、`bm25`、`rerank` 等底层过程。

### 3.3 后续 skill 方向

后续可以逐步补齐：

- `web_search_retrieve`
- `tool_execute`
- `file_read`
- `code_execute`

但近期重点还是先把本地 RAG skill 做扎实。

## 4. Grounding Validator

validator 的职责需要升级。

它不再只是“有没有引用”的检查器，
而应成为：

- 工具结果与答案关联性检查器
- 可解释性增强器
- 幻觉抑制器

建议职责包括：

- 检查子任务执行结果是否满足 success criteria
- 检查答案关键结论是否被已有结果支撑
- 区分强支持 / 弱支持 / 无支持
- 给 planner 返回下一轮可执行反馈
- 在必要时阻止弱支撑答案直接放行

这意味着 validator 会成为 planner loop 的反馈部件，而不只是最后一道样式化检查。

## 5. 本地 RAG 主线

### 5.1 目标不变，但入口改变

本地 RAG 近期的效果目标没有变化：

- 层次化路由
- 多路召回
- semantic reranker
- 更稳定的 evidence selection

变化在于：

- 它不再作为多个 planner-facing 节点暴露
- 而是作为统一 skill 被调用

### 5.2 近期继续推进的内容

- 稳定 `hierarchy routing`
- 调整 hybrid retrieval 的候选规模
- 让 semantic reranker 只打最值得重排的候选
- 优化 evidence packaging，便于 validator 直接消费

### 5.3 当前明确不优先推进

- 父子索引 / 多层索引对象建设保留在 TODO
- 不作为这一轮 skill runtime 与 reranker 演进的阻塞项

## 6. 在线搜索的定位

在线搜索仍然是补充能力，不应默认替代本地 RAG。

它适用于：

- 本地知识不足
- 用户明确要求联网
- 时间敏感问题

在新架构下，在线搜索更适合作为：

- `web_search_retrieve` skill

与本地 RAG 一样，它应输出结构化结果，而不是把过多搜索过程暴露给 planner。

## 7. 性能与延迟方向

这次架构重构的一个重要目标就是降低延迟和成本。

### 7.1 Fast Path 负责降平均成本

通过 fast gate，简单问题不再经历：

- planner
- 多轮 task loop
- 多次检索
- 重 validator 流程

### 7.2 Planner Loop 负责控制复杂问题成本

复杂问题的优化重点是：

- 减少 planner 回合数
- 让子任务更原子
- 提升 skill 单次执行成功率
- 限制 reranker 候选规模
- 只在必要时请求补任务

## 8. 迁移顺序

建议按以下顺序推进：

1. 先更新文档与架构认知
2. 引入 fast gate
3. 定义统一 skill schema
4. 将本地 RAG 打包为 `local_kb_retrieve`
5. 将现有 `rag/search/action` 迁移到 skill executor 协议
6. 重写 planner 输出 schema
7. 将 verifier / checker 重构为 grounding validator
8. 逐步清理旧 graph 中的硬编码节点链

## 9. 长远目标

长远来看，这个项目的目标不是停留在“一个会检索的 agentic RAG demo”，
而是成为一个更完整的 agent runtime：

- 有 fast path
- 有 planner loop
- 有 skill runtime
- 有 grounding validator
- 有未来可扩展的工具与执行边界

长期应该坚持的原则是：

- 简单问题走短链路
- 复杂问题走结构化循环
- planner 只做全局决策
- skill 只做强执行
- validator 只做 grounding 与放行约束
