# Agent Product Gap Analysis

本文档从“成熟 agent 产品”而不是“节点型 workflow demo”的视角，
分析当前项目的主要差距、风险点与优先级改进方向。

它与另外两份文档的关系是：

- [docs/langgraph-workflow-visualization.md](/home/viottery/workspace/agentic-rag/docs/langgraph-workflow-visualization.md)
  负责描述目标 runtime 架构
- [docs/future-roadmap.md](/home/viottery/workspace/agentic-rag/docs/future-roadmap.md)
  负责描述迁移主线与实施顺序
- 本文档
  负责从产品化、成熟度和收益风险角度解释“为什么要这样重构”

## 1. 当前判断

当前项目已经不是“只有一个聊天接口”的最小 demo。

它已经具备：

- FastAPI 统一 API 入口
- LangGraph 驱动的多步执行雏形
- 本地 Qdrant RAG
- Tavily 搜索
- MediaWiki 数据采集与增量索引
- 基础的 grounding / verification 思路
- semantic reranker 的第一版接入

但从成熟 agent 产品视角看，当前系统仍然更像：

- 一个“节点编排 + 检索实验”的可调试框架

而不是：

- 一个“有 fast path、skill runtime、grounding validator、稳定产品接口”的 agent runtime

换句话说，当前最大的差距已经不再是“有没有 agent 味道”，
而是“有没有产品级 runtime 边界与能力抽象”。

## 2. 当前做得对的地方

在分析差距之前，需要先明确当前已经做对的部分。

### 2.1 本地 RAG 已经有真实基础

当前本地 RAG 并不是单纯的向量 top-k：

- 层级 metadata 已经进入 payload
- hybrid retrieval 已经形成
- semantic reranker 已开始接入
- scope routing 已有基础接口
- evidence 输出链路已经存在

这意味着本地 RAG 很适合被进一步封装成一个独立 skill，
而不是推翻重来。

### 2.2 数据链路已经具备工程骨架

当前已经形成：

```text
crawl / local files
  -> raw documents
  -> manifest
  -> indexing
  -> qdrant payload
  -> retrieval
```

这条链路已经有：

- 稳定 `document_id`
- hierarchy metadata
- 增量判断
- stale document 删除
- 本地知识树形落盘

这些都说明项目不是“纯 prompt 拼接系统”，
而是已经开始具备知识基础设施。

### 2.3 已经开始重视 grounding

当前系统已经不是“检索后直接吐答案”。

它已经在尝试：

- evidence 汇总
- 回答约束
- 引用映射
- verification
- budget fallback

虽然这些能力还需要重构，
但方向上已经站在了“可信回答”这边。

## 3. 主要差距

下面这些问题，是当前项目距离成熟产品最关键的差距。

### 3.1 缺少 fast path，简单问题成本过高

这是当前最明显的产品体验问题之一。

现在的系统更偏向：

- 凡事先进入多步 agent 链路

这会带来几个问题：

- 简单问题也要走重链路
- token 成本偏高
- 平均响应时间偏长
- 用户会感知到“不必要的思考”

成熟产品通常会优先做一件事：

- 让简单问题尽快返回

因此，当前最缺的不是再加几个节点，
而是明确：

- 哪些问题可以直接回答
- 哪些问题只需要一次 skill
- 哪些问题才值得进入 planner loop

### 3.2 缺少统一 skill runtime，能力边界仍然分散

当前系统的能力仍然更多暴露为：

- 若干 graph 节点
- 若干 prompt 边界
- 若干子 agent

这对产品化不利。

原因是：

- planner 需要知道太多底层能力细节
- prompt 中能力描述会不断膨胀
- 新能力接入时容易继续长出新的专用分支
- validator 和上层 orchestration 难以消费统一的结果对象

成熟产品更需要的是：

- 统一 skill contract

也就是：

- 输入 schema
- 执行边界
- 输出 schema
- 错误与置信度表达

当前最重要的缺口不是“再做一个新 agent”，
而是“把已有能力收口成可规划、可验证、可组合的 skill”。

### 3.3 Planner 还不够“全局”，子任务执行还不够“原子”

这是架构层的核心问题。

当前系统虽然已经有 planner，
但整体仍然带有很强的“节点驱动工作流”痕迹。

这样的问题在于：

- planner 很容易被底层实现细节拖累
- 子任务内部容易继续做局部决策
- 任务拆分容易变得不够原子
- 最终形成“全局和局部都在做决策”的双重复杂度

成熟的 agent runtime 更应该是：

- Planner 只做全局判断
- 子任务只做单一职责执行

也就是说：

- Planner 决定“做什么”
- Skill 决定“怎么做完当前原子任务”

当前系统的主要差距，不是 planner 不够聪明，
而是职责边界还不够收敛。

### 3.4 验证层还停留在轻量 citation / checker 视角

从产品视角看，当前“验证”还偏轻。

当前重点更接近：

- 有没有 citation
- 覆盖率够不够
- 有没有明显 unsupported paragraph

这当然有价值，但还不足以支撑成熟 agent 产品。

成熟产品需要 validator 检查的是：

- 工具或 skill 返回的结果是否真的回答了子任务
- 最终答案是否真的被这些结果支撑
- 哪些结论只是推断
- 哪些内容超出了证据边界
- 是否应该回到 planner 继续补任务

也就是说，验证层应从：

- `citation/verification/checker`

升级成：

- `grounding validator`

### 3.5 API 与状态仍偏内部调试态

当前 `/chat` 仍返回比较胖的状态对象。

这在架构演化阶段是合理的，
但从产品角度看仍有明显差距：

- 外部契约不稳定
- 内部节点状态泄漏过多
- 响应模型不够收敛
- 不利于长期兼容

成熟产品通常至少会区分：

- 用户响应
- trace / debug / observability 输出

当前项目还处在“调试友好优先”的阶段，
没有完全进入“产品接口优先”的阶段。

### 3.6 Runtime 还不是产品级任务系统

当前 graph 能跑，
但还不是成熟 runtime。

仍缺少：

- task checkpoint
- 恢复机制
- 中断 / 取消
- 更清晰的任务状态查询
- 更细的超时和重试治理
- 多会话和多用户层面的运行治理

这意味着当前更适合：

- 架构验证
- 小规模开发迭代

而还不适合：

- 长任务
- 多用户
- 严格 SLA
- 强治理工具接入

## 4. 根本原因

这些差距并不意味着当前方向错了。

相反，它们主要来自项目当前阶段的有意识取舍：

- 先把主链路搭起来
- 先验证 RAG 和搜索能力
- 先把 evidence / verification 思路跑通
- 先保留足够多的中间状态便于调试

这本身是合理的。

真正的问题不是：

- “为什么现在还不成熟”

而是：

- “下一阶段要不要继续堆节点和 prompt 复杂度”
- “还是应该先收敛 runtime 与 skill 抽象”

本文的结论非常明确：

- 下一阶段不应继续主要围绕旧 graph 复杂度扩张
- 而应优先转向 `fast path + skill runtime + grounding validator`

## 5. 优先级建议

### 5.1 第一优先级：引入 fast gate

这是平均时延、成本和体验最直接的优化杠杆。

重点不是追求极其复杂的分类器，
而是先明确最小决策集：

- `direct_answer`
- `single_skill`
- `planner_loop`

只要这一步成立，系统的平均成本就会显著下降。

### 5.2 第二优先级：把本地 RAG 打包成 skill

当前最值得先收口的能力就是本地 RAG。

原因：

- 它已经是项目最真实、最稳定的能力
- 内部步骤已经够丰富，值得封装
- 对 planner 的价值最大

建议尽快形成：

- `local_kb_retrieve`

让它统一负责：

- routing
- hybrid retrieval
- rerank
- evidence packaging

### 5.3 第三优先级：重写 Planner 的输出契约

Planner 不该继续主要围绕底层节点工作。

建议改为直接输出：

- 子任务
- executor / skill
- success criteria
- planner note

这样可以让系统从“节点型 workflow”真正转向“任务型 runtime”。

### 5.4 第四优先级：把验证层升级为 grounding validator

这一层决定系统是不是“会答但不稳”，
还是“能对自己的答案负责”。

建议 validator 逐步承担：

- 结果与子任务目标对齐检查
- 结果与最终答案对齐检查
- 支撑强弱标记
- 回流 planner 的修正建议

### 5.5 第五优先级：继续稳固本地 RAG 质量

虽然架构主线已经变化，
但 retrieval quality 仍然非常重要。

当前应继续推进：

- 语料清洗
- 层次化路由稳定性
- hybrid retrieval 候选控制
- semantic reranker 调优
- evidence packaging 质量

原因很简单：

- 如果 skill 本身质量不稳，再好的 planner 也会被拖累

## 6. 当前不建议优先投入的方向

### 6.1 不建议继续把主要精力放在旧 planner prompt 微调上

当前最缺的已经不是：

- planner 再多几个技巧

而是：

- planner 的职责边界收敛
- planner 消费更好的 skill 抽象

在 runtime 结构没收敛之前，
继续大调 planner prompt 的收益通常有限。

### 6.2 不建议继续扩张大量专用节点分支

如果现在继续为各种任务堆：

- 新节点
- 新分支
- 新特判

短期会显得能力更多，
长期则会带来：

- 路由复杂
- 维护困难
- prompt 膨胀
- 结果难统一验证

### 6.3 不建议把父子索引放进当前主线阻塞项

父子索引是合理的未来方向，
但当前不应该成为 skill runtime 和 grounding validator 的阻塞项。

它更适合继续保留在 TODO，
等到主架构收敛之后再推进。

## 7. 建议的阶段化路线

### Phase 1：文档与架构认知统一

目标：

- 先统一系统设计语言

内容：

- 明确 fast path
- 明确 planner 边界
- 明确 skill runtime
- 明确 grounding validator

### Phase 2：最小 runtime 重构

目标：

- 让新架构先跑起来

内容：

- 引入 fast gate
- 定义 skill schema
- 打包 `local_kb_retrieve`
- 改写 planner 输出结构

### Phase 3：验证层升级

目标：

- 让系统能判断“结果是否真的支撑答案”

内容：

- 将 verifier / checker 重构为 grounding validator
- 增强结果-答案关联性检查
- 增强 planner feedback

### Phase 4：扩展 skill 生态与 runtime 能力

目标：

- 从本地 RAG runtime 扩展到更完整的 agent runtime

内容：

- `web_search_retrieve`
- `tool_execute`
- 更成熟的任务治理
- 更稳定的 API 与 trace 输出

## 8. 一句话结论

当前项目最缺的已经不是“再多一些 agent 节点”，
而是：

- 更短的简单问题路径
- 更稳定的 skill 抽象
- 更清晰的 planner 边界
- 更强的 grounding validator

因此，下一阶段最关键的事情不是继续堆旧 workflow，
而是把系统真正重构成一个：

- `fast path + planner loop + skill runtime + grounding validator`

的产品级 agent runtime。
