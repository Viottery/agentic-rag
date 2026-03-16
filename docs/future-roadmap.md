# Future Roadmap

本文档用于整理 Agentic-RAG 下一阶段的研发方向，重点围绕以下目标展开：

- 可信：回答尽量有证据约束、引用标记和验证机制
- 高效：尽量减少无效循环、控制 token 消耗、提高响应速度
- 实用：逐步接入真实搜索与桌面助手能力
- 有趣：让 agent 不只是“能答”，还真正具备任务协作和工具执行能力

## 1. 产品方向

项目的长期目标不是停留在一个“会检索的聊天接口”，而是逐步演进为一个可信、轻量、可扩展的桌面助手。

这个桌面助手需要具备三层能力：

- 知识能力：能够读取本地知识库、搜索外部信息、理解多来源证据
- 执行能力：能够调用工具、操作文件、执行命令、处理实际任务
- 可信能力：能够说明依据、标记引用、承认限制、在预算约束内给出最佳努力回答

## 2. 当前阶段判断

当前系统已经具备一个可工作的 supervisor 风格多 agent 骨架：

- `planner -> dispatcher -> sub-agent -> planner -> answer_generator -> checker`

其中：

- `rag_agent` 已接入真实本地检索
- `search_agent` 目前仍为 mock
- `action_agent` 目前仍为 mock
- 已具备基本的预算限制与降级机制

这说明当前最有价值的下一步，不是简单增加更多节点，而是提升“真实能力”和“可信约束”。

## 3. 下一阶段总目标

下一阶段建议聚焦在四条主线：

### A. 可信回答链路

目标：

- 让回答尽量受 evidence 约束
- 给关键结论附引用
- 在信息不足时明确说明
- 降低“看起来合理但无依据”的生成

这是最关键的一条主线。

### B. 真实搜索能力

目标：

- 接入被广泛认可、适合 agent 场景的搜索 API
- 让系统能够处理“最新信息”问题
- 将搜索结果纳入统一 evidence 结构，而不是单独走旁路

### C. 实用工具系统

目标：

- 让 agent 能完成桌面助手类任务
- 文件读写、编辑、命令行、邮件等都可以逐步接入
- 每类工具都有边界、风险等级和确认策略

### D. 性能与成本优化

目标：

- 降低 planner / checker 的无效调用
- 在不损害可信度的前提下减少 token 消耗
- 为后续真实工具和搜索接入预留预算控制能力

## 4. 最关键的优先事项

如果只选一个最关键的实现，建议优先做：

## 可信回答链路

原因：

- 当前系统已经能“组织流程”，但还不够“约束回答”
- 一旦未来接入真实搜索和更多工具，回答质量和真实性风险会更高
- 如果不先做好引用、验证和证据约束，系统能力越强，潜在误导成本越高

这一阶段的核心目标是：

```text
retrieve/search
  -> normalize evidence
  -> answer with citations
  -> verify support
  -> return best-effort grounded answer
```

## 5. 可信链路的具体实施项

### 5.1 统一 evidence schema

将本地 RAG、搜索结果、工具结果统一整理进同一套证据结构。

建议后续增强字段：

- `source_type`
- `source_name`
- `source_id`
- `title`
- `content`
- `score`
- `source_url`
- `retrieved_at`
- `freshness`
- `quote`
- `support_score`

价值：

- 统一后续 answer_generator 和 verifier 的输入
- 降低不同 agent 之间的数据结构割裂

### 5.2 引用型回答生成

目标：

- 回答中关键结论带引用标记
- 输出不只给内容，也给来源

例如：

```text
该项目当前使用 FastAPI 作为 API 层，并用 LangGraph 编排 agent 工作流。[Project_goal::chunk::000]
```

价值：

- 回答可追溯
- 更适合后续前端展示 citations

### 5.3 轻量 verifier

verifier 不必重做完整推理，而是只做轻量检查：

- 抽取答案中的关键 claims
- 检查每个 claim 是否有对应 evidence 支持
- 输出支持率和未支撑结论

建议输出：

- `supported_claims`
- `unsupported_claims`
- `citation_coverage`
- `final_confidence`

价值：

- 降低幻觉风险
- 比单纯“checker pass/fail”更可解释

### 5.4 可信降级

当前系统已经有预算限制降级。

下一步建议将降级细化为三层：

- 任务级降级：某个 agent 执行失败或超时
- 图级降级：整个 workflow 达到预算上限
- 输出级降级：答案明确说明哪些部分已证实、哪些部分未证实

价值：

- 降级更透明
- 用户体验比简单失败更好

## 6. 真实搜索规划

建议优先接入一个成熟、在 agent 生态中常见的搜索 API。

优先建议：

### Tavily

原因：

- 面向 agent / LLM 场景设计
- 在 LangChain / agent 生态中使用广泛
- 接入成本相对较低

接入后建议流程：

```text
search_agent
  -> real search api
  -> search results normalization
  -> evidence extraction
  -> answer_generator / verifier
```

后续搜索增强项：

- query rewrite
- result dedup
- freshness routing
- snippet cleaning
- page-level extraction

## 7. 工具系统规划

建议按风险和价值分批接入，而不是一次性接很多工具。

### 第一批：低风险高价值工具

- 读文件
- 列目录
- 搜索本地文本
- 写新文件

### 第二批：中风险工具

- 修改现有文件
- 命令行执行
- 文本批处理

### 第三批：高风险工具

- 邮件收发
- 删除文件
- 外部系统写操作

设计原则：

- 所有工具都应有统一注册层
- 每次调用都保留审计记录
- 高风险工具默认 require-confirmation
- 工具返回结果必须结构化，并纳入统一 evidence / trace

## 8. 性能与 token 优化方向

未来如果要做到“快速响应、节约 token”，建议重点优化这些点：

### 8.1 简单问题 fast path

对明显简单的问题不必走完整 planner loop。

例如：

- 已知本地知识问答
- 简单文件读取
- 轻量格式转换

可以走轻量路由，减少 planner/checker 消耗。

### 8.2 轻量 verifier

不要每次都用大模型完整复审一遍答案。

可以优先：

- 规则检查
- 结构检查
- 小模型验证

只有高风险问题再走完整 verifier。

### 8.3 缓存

可以考虑加入：

- 检索结果缓存
- 搜索结果缓存
- prompt 片段缓存
- 相同任务的工具结果缓存

### 8.4 预算细化

当前只有全局预算：

- 最大轮数
- 最大时长

未来可以进一步加入：

- 单节点超时
- 单工具超时
- 单请求 token budget
- 不同任务类型的不同预算策略

## 9. 建议的迭代顺序

### 阶段 1：可信回答 MVP

- 统一 evidence schema
- answer_generator 增加引用输出
- 增加 verifier
- 输出 support/confidence 信息

### 阶段 2：接入真实搜索

- search_agent 接入 Tavily
- 搜索结果进入统一 evidence schema
- 支持时效性问题

### 阶段 3：工具系统第一批

- 读文件
- 写文件
- 本地搜索
- 安全的命令行能力

### 阶段 4：性能优化

- fast path
- 缓存
- 更细预算
- 并行分发探索

## 10. 当前系统与路线的关系

现有系统已经提供了很适合继续演进的边界：

- `planner` 适合作为 supervisor 保留
- `dispatcher` 适合扩展成更完整的 task router
- `rag_agent` 已经是真实能力入口
- `search_agent` 和 `action_agent` 已经预留了替换真实实现的接口
- `answer_generator` 和 `checker` 可以分别升级成“带引用生成器”和“支持度验证器”

所以当前架构不需要推倒重来，下一阶段更像是在现有骨架上逐步“换实心器官”。

## 11. 一句话总结

下一阶段最值得投入的方向，不是让 agent “会更多”，而是先让 agent “说得更有根据”。在此基础上，再接入真实搜索与实用工具，项目才更接近一个可信、快速、节约 token、又真正有用的桌面助手。
