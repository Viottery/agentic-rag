# Agent Product Gap Analysis

本文档从“成熟 agent 产品”而非“工作流 demo / 技术验证项目”的视角，整理当前项目的主要差距、风险点与优先级改进方向。

它的定位不是替代现有的：

- `docs/langgraph-workflow-visualization.md`
- `docs/future-roadmap.md`

而是补上一层更偏产品化、工程成熟度和落地顺序的分析。

## 1. 当前判断

当前项目已经完成了一套相当清晰的 agent 能力骨架：

- FastAPI 提供统一 API 入口
- LangGraph 提供 supervisor 风格编排
- 本地 Qdrant RAG 已经可用
- Tavily 搜索已具备第一版 web-RAG 能力
- 回答链路已经接入 citation mapping、verification、checker
- MediaWiki 知识采集、索引与增量 manifest 已经形成最小闭环

这说明项目已经不是“只有一个聊天接口”的最小 demo，而是进入了：

`workflow validation + retrieval architecture exploration`

阶段。

但从成熟 agent 产品视角看，当前系统仍然更像：

- 一个可观察、可调试、正在演化中的 agent 框架

而不是：

- 一个稳定、可恢复、可审计、可持续优化的 agent 产品

换句话说，当前项目的主要问题不是“没有 agent 味道”，而是“产品化关键层还没补齐”。

## 2. 当前做得好的地方

在分析差距之前，需要先明确当前已经做对的部分，因为这些会直接影响后续迭代策略。

### 2.1 主干架构是清晰的

当前图结构并没有急着把所有未来任务硬编码成专用 workflow，而是先建立了稳定的节点角色：

- planner
- dispatcher
- query_refiner
- rag_router / rag_agent
- search_agent
- action_agent
- answer_generator
- citation_mapper
- verifier
- checker

这是一种比较健康的起点，因为后续可以逐步替换节点内部实现，而不必推翻整条主链路。

### 2.2 本地知识库能力已经不是“纯向量搜索”

当前本地检索已经开始朝更成熟的 retrieval pipeline 演化：

- embedding 检索
- BM25 / lexical 检索
- RRF 融合
- document-focused 二次筛选
- knowledge hierarchy metadata
- coarse-to-fine routing 的预备接口

这比“只做一次向量 top-k”要成熟得多。

### 2.3 数据链路已经开始结构化

`crawl-mediawiki -> raw txt -> manifest -> index manifest -> Qdrant payload -> retrieval`

这条链路已经具备：

- 原始源数据落盘
- 本地层次目录
- 内容哈希
- 索引增量判断
- document_id 稳定化
- hierarchy metadata 写入 payload

这是后续做知识库治理和产品化质量提升的基础。

### 2.4 已经开始重视可信回答

当前系统不是“检索完立刻吐答案”，而是加入了：

- answer draft
- citation mapping
- verification
- checker
- budget-limited fallback

这意味着项目的方向已经不是单纯追求“能答”，而是希望回答可解释、可回看、可控。

## 3. 主要问题与成熟度差距

下面这些问题，是当前项目距离成熟 agent 产品最关键的差距。

### 3.1 执行层仍然缺失，系统还不能稳定“做事”

这是当前最明显的产品级缺口。

当前情况：

- `action_agent` 仍然是 mock
- `search_agent` 虽然接入了 Tavily，但缺失更完整的工具治理层
- 项目里还没有统一的 tool runtime

这意味着系统目前主要擅长：

- 检索
- 搜索
- 汇总
- 生成回答

但还不擅长：

- 稳定执行动作
- 读写本地资源
- 调用受控工具
- 执行多步任务
- 对执行失败进行可恢复处理

而成熟 agent 产品的关键，不只是“会回答”，更是“会在受控边界内完成任务”。

当前缺少的不是单个 action node，而是一整层执行基础设施：

- 工具注册与发现
- 参数 schema 与校验
- 权限分级
- 执行审计
- 风险边界
- 超时 / 重试 / 取消
- 结构化错误返回

如果这一层不补齐，系统很难从 RAG/QA 产品走向真正的 agent 产品。

### 3.2 RAG 质量还不够产品级，尤其是中文知识库场景

从产品效果看，当前最值得优先投资的部分，其实不是 planner，而是 retrieval quality。

当前主要问题：

- embedding 模型使用 `sentence-transformers/all-MiniLM-L6-v2`
- 对中文大规模 wiki 语料并不理想
- chunking 仍然是字符窗口切块
- MediaWiki 清洗仍较粗糙
- 文本中保留了不少目录、按钮文案、编辑标记等页面噪声

这会直接影响：

- 召回质量
- chunk 纯度
- rerank 稳定性
- 引用映射质量
- 最终回答可读性

对于像 `data/raw/prts-wiki/干员/1765-真理.txt` 这样的页面，可以看到正文中仍然混入大量：

- 目录项
- “编辑”
- “看图模式”
- “下载立绘”
- “试听语音”

这类内容对知识检索几乎没有正向价值，但会稀释语义密度。

因此，当前本地 RAG 的主要瓶颈不是“有没有路由”，而是：

- 语料是否干净
- embedding 是否适合中文
- chunk 是否足够面向结构
- rerank 是否真正语义相关

如果这些基础问题不解决，后续再叠更多 planner prompt 或复杂 graph，也很难根本改善用户体验。

### 3.3 API 仍处于调试态，缺少产品接口收敛

当前 `/chat` 的返回内容仍然是完整 `AgentState`。

这在 workflow validation 阶段很合理，但从产品角度看存在明显问题：

- 返回结构过胖
- 内部状态直接暴露
- 缺少面向用户的稳定响应模型
- 缺少面向前端/客户端的长期兼容契约

同时，仓库里已经存在：

- `app/api/schemas/request.py`
- `app/api/schemas/response.py`

但当前 route 并没有真正收敛到这些 schema。

这说明 API 层目前仍处于：

- 内部调试优先

而不是：

- 产品接口优先

成熟 agent 产品通常至少需要区分两层输出：

- 用户可消费的稳定响应
- 内部诊断/trace/debug 信息

否则一旦内部状态结构演化，外部接口也会被迫一起变化。

### 3.4 编排能跑，但还不是生产级 runtime

当前的 LangGraph supervisor workflow 已经足以验证主链路，但离成熟 runtime 还有明显差距。

当前缺少的关键能力包括：

- 持久化 checkpoint
- 失败恢复
- 长任务中断与重试
- streaming 输出
- 并行 fan-out / fan-in 的真实实现
- 多会话隔离
- 任务状态查询
- 节点级 SLA / timeout / retry policy

当前系统更像“一次性执行图调用”，而不是“任务系统”。

对于成熟 agent 产品，runtime 至少要能支持：

- 任务运行过程可追踪
- 长任务可恢复
- 工具失败不会直接让整轮不可控
- 执行预算可按节点、按任务、按用户维度治理

否则系统一旦接入真实工具与更复杂任务，运行稳定性会迅速变差。

### 3.5 评估体系仍偏 smoke test，不足以支撑持续优化

当前测试已经覆盖了一些很重要的内容：

- mediawiki helper
- hierarchy metadata
- structure summary
- rag route fallback
- hybrid retriever helper
- e2e chat smoke flow

但这些更偏：

- 行为可用性验证
- 结构完整性验证

而不是：

- 质量回归验证

成熟 agent 产品需要持续衡量的通常包括：

- retrieval recall / precision
- rerank 质量
- citation coverage
- hallucination rate
- latency
- token cost
- fallback rate
- tool failure rate
- answer satisfaction proxy

没有这些，系统后续很难知道：

- 哪些改动真的提升了效果
- 哪些只是让工作流更复杂
- 哪些策略实际上在增加成本或退化质量

### 3.6 安全、权限和治理能力仍处于空白或非常早期阶段

如果把项目目标理解为未来的完整 agent runtime，那么当前最薄弱、但后续一定要补的层是治理能力。

当前缺少的典型产品能力包括：

- 工具权限模型
- 高风险操作隔离
- 用户/租户级别的访问边界
- 调用审计
- 速率限制
- 配额治理
- 敏感信息与密钥暴露控制
- prompt injection / tool injection 的系统级防御

目前节点 prompt 中已经有相当明确的“把输入视为不可信数据”的意识，这很好。

但成熟产品不能只靠 prompt 层防御，还需要：

- runtime 层约束
- tool schema 层约束
- 权限与审批层约束

### 3.7 性能、成本与多用户场景尚未进入工程治理

当前阶段的实现强调清晰和可调试，这是合理的。

但从成熟产品视角看，系统还没有系统性地处理这些问题：

- embedding / retrieval / generation 的整体时延
- 大知识库下 scroll / BM25 的成本
- 大量用户并发时的负载模型
- 多轮 planner loop 的 token 成本
- 搜索与抽取调用的成本控制
- 模型降级策略

这意味着当前系统更适合：

- 学习
- 演示
- 单人/小规模迭代

还不适合：

- 稳定对外服务
- 多用户并发
- 严格成本约束场景

## 4. 造成这些差距的根本原因

这些问题并不意味着当前方向错了。

相反，它们主要来自项目当前阶段的有意取舍：

- 先把 agent 主链路和能力边界搭起来
- 先验证 state / evidence / citation / checker 的工作方式
- 先让本地 RAG 和搜索能力接入可运行形态
- 暂时不把高副作用工具和复杂 runtime 一次性做全

这套策略本身是合理的。

问题不在于“为什么现在还不成熟”，而在于：

- 下一阶段要不要继续往 graph 和 prompt 上加复杂度
- 还是先把检索、执行层、接口层、评估层这些产品基础设施补齐

本文的结论非常明确：

下一阶段的主要工作，不应再以“继续堆更多 agent 味道”为主，而应以“把系统变成更像产品的 agent” 为主。

## 5. 优先级建议

如果从收益与风险的平衡来看，建议按下面顺序推进。

### 5.1 第一优先级：把本地 RAG 质量做扎实

这是当前最值得优先投入的方向。

重点包括：

- 改进 MediaWiki 清洗质量
- 为正文、目录、表格、注释等内容分层处理
- 用更适合中文的 embedding 模型替换当前默认模型
- 补 query-document semantic reranker
- 让 chunking 更面向结构而不是字符窗
- 建立 retrieval eval 基线

为什么它应该排第一：

- 这是当前系统最真实、最稳定、最常用的能力
- 它直接影响搜索、引用、回答和用户主观体验
- 检索质量不稳时，后续 planner 与 verification 都会被拖累

### 5.2 第二优先级：建设真实的 tool runtime

建议不要再把“action_agent 做几个特殊逻辑”当作主方向，而是直接思考统一执行层。

优先落地的工具类型可以是：

- 受控搜索工具
- 文件读取工具
- 本地文本检索工具
- 命令执行工具

需要同时补齐：

- schema
- permission
- timeout
- retry
- audit
- structured result

这样做之后，`action_agent` 才能从 mock 边界变成真正的执行边界。

### 5.3 第三优先级：收敛 API 与会话层

建议尽快区分两类输出：

- 用户响应模型
- 调试/trace 模型

同时逐步引入：

- `session_id`
- 会话上下文
- streaming
- task status / trace 查询

这是从“内部调试接口”走向“产品接口”的必要一步。

### 5.4 第四优先级：建立评估与观测闭环

建议补齐一套最小但持续可运行的评估系统：

- 固定问题集
- retrieval 召回评估
- citation 覆盖评估
- final answer 质量样本评估
- latency / token / fallback 统计

同时补 observability：

- 每轮 planner 决策日志
- 每个工具调用日志
- evidence 数量与来源统计
- verifier/checker 失败原因统计

否则后续优化很难形成闭环。

### 5.5 第五优先级：再推进更复杂的 graph 能力

在前面几层更稳以后，再继续做下面这些会更划算：

- parallel fan-out / fan-in
- retrieval subgraph
- search subgraph
- 长任务恢复
- task family workflow

这时复杂度的收益才会更明显。

## 6. 不建议当前阶段优先投入的方向

下面这些方向不应该完全不做，但不建议作为当前主线。

### 6.1 不建议继续把主要精力放在 planner prompt 微调上

当前 planner 已经足够完成工作流验证。

在 retrieval 和 tool runtime 还不够稳的情况下，继续大量调 planner：

- 收益有限
- 可解释性会下降
- 难以形成稳定评估

### 6.2 不建议过早扩张大量专用 workflow

如果现在就为很多任务写专用链路，短期会显得能力变多，但长期很容易变成：

- 路由复杂
- 维护成本高
- 复用差
- 难以统一审计与治理

### 6.3 不建议把产品化问题留到“能力做完之后再说”

对 agent 系统来说，下面这些不是收尾项，而是主干项：

- API 契约
- 工具权限
- 评估
- 观测
- 恢复机制

越晚补，返工越大。

## 7. 建议的阶段化落地路线

为了让路线更可执行，可以把后续工作拆成四个阶段。

### Phase 1：质量基础设施

目标：

- 先把“找对证据”做稳

建议内容：

- 清洗 MediaWiki 噪声
- 改造中文 embedding
- 引入 reranker
- 建立 retrieval eval
- 优化 chunking

成功标志：

- 检索结果显著更干净
- 相同问题的返回更稳定
- 引用覆盖率和回答质量明显提升

### Phase 2：执行基础设施

目标：

- 让系统从“会回答”变成“能在受控边界内执行”

建议内容：

- 建立统一 tool runtime
- 将 `action_agent` 从 mock 改为真实执行边界
- 引入权限与审计
- 补工具失败恢复与超时治理

成功标志：

- 至少一批真实工具稳定接入
- 工具调用结果结构化可追踪
- 高风险能力有边界

### Phase 3：接口与 runtime 产品化

目标：

- 让系统从内部实验框架变成可对接产品的服务

建议内容：

- 收敛 API 响应
- 增加 session / trace / streaming
- 引入 checkpoint 和恢复
- 完善任务状态管理

成功标志：

- API 对前端稳定可用
- 长任务具备恢复与观测能力
- 不再需要把完整内部状态直接暴露给客户端

### Phase 4：高级编排与平台化

目标：

- 在基础设施稳固后，再释放更复杂的 agent 能力

建议内容：

- fan-out / fan-in 并行图
- retrieval/search subgraph
- task family workflow
- 多用户成本治理
- 更强的安全和权限体系

成功标志：

- 复杂任务吞吐和稳定性明显提升
- 新能力接入不再依赖堆专用分支

## 8. 一句话结论

当前项目最缺的不是“更像 agent”，而是“更像产品”。

下一阶段最关键的不是继续堆更多 planner / graph 复杂度，而是优先补齐：

- 更可靠的本地 RAG 质量
- 真正的 tool runtime
- 收敛的 API / session 层
- 评估与观测闭环
- 运行时治理能力

只有这些基础设施到位之后，项目才能从一个结构优秀的 agent 实验框架，逐步成长为真正成熟的 agent 产品。

## 9. 如何正确抽象一次失败案例

在分析具体失败案例时，需要特别避免把结论写成：

- 没有识别某个具体角色
- 没有给某个具体页面加规则
- 没有为某个具体问题写特判

这种结论虽然容易推动短期修补，但会把系统带向不断堆规则的方向。

更好的方式是把单次失败上升为通用能力层问题。

### 9.1 不应抽象成“某个实体没消歧”

例如：

- “没有识别夜刀和异格夜刀”
- “没有对某个 wiki 页面做特殊处理”

这些都属于过于具体的现象层结论。

真正应抽象的问题是：

- 系统缺少通用的实体解析与候选解释能力
- 系统缺少在存在歧义时触发澄清或保守回答的策略

这类问题不仅会出现在游戏角色名上，也会出现在：

- 人名
- 公司名
- 版本名
- 产品简称
- 文件名
- 工具名
- 同名 API / 库

因此应优先建设：

- entity resolution
- ambiguity detection
- clarification policy

而不是为每个实体单独加规则。

### 9.2 不应抽象成“某个搜索词改错了”

例如：

- “不该搜 operator comparison”
- “不该把中文翻译成英文”

这类结论也仍然过于具体。

真正的问题是：

- 问题表示在多节点之间没有被稳定保持
- 系统没有统一的 query representation policy
- planner / query_refiner / search_agent 对用户意图的表示不一致

这意味着后续应建设的不是“更多 rewrite prompt”，而是：

- 面向任务类型的 query planning
- 面向语言与知识源的 query normalization
- 面向工具能力的 query contract

### 9.3 不应抽象成“某个文档没命中正确 chunk”

更高层的抽象应是：

- 系统具备 document retrieval，但缺少 evidence targeting
- 系统能找到“相关文档”，但未必能找到“最适合回答当前问题的证据片段”

对于成熟 agent，这一区别非常重要。

真正需要补齐的能力是：

- section-aware retrieval
- field-aware retrieval
- evidence-type routing
- query-document semantic reranking

### 9.4 更合理的失败抽象模板

建议后续把失败案例统一整理成下面这种格式：

1. 用户任务类型是什么
2. 系统在哪一层误解了任务
3. 哪个能力层缺失导致误解持续放大
4. 当前 fallback 为什么没有及时止损
5. 应补的是哪种通用能力，而不是哪条特判

按照这个模板，类似失败案例通常会被抽象成以下几类通用能力问题：

- `entity resolution` 不足
- `task framing` 不足
- `evidence targeting` 不足
- `tool escalation policy` 不足
- `answerability judgment` 不足
- `representation stability` 不足

这比“再多加几条 domain-specific heuristic”更有长期价值。

## 10. Memory Strategy

当前项目已经有一定的“工作记忆”雏形，即：

- `AgentState`
- `subtasks`
- `aggregated_context`
- `evidence`
- `trace_summary`

但从成熟 agent 产品视角看，后续确实应开始显式设计 memory 系统。

重点在于：

- 要建设 memory
- 但不要把 memory 当成当前主要问题的替代解法

### 10.1 为什么现在要开始考虑 memory

随着系统从单次问答走向复杂 agent runtime，下面这些能力都会逐渐依赖 memory：

- 多轮澄清
- 连续任务上下文保持
- 用户口径记忆
- 工具执行轨迹回溯
- 长任务分阶段恢复
- 失败后继续执行

如果没有 memory 分层设计，后续系统通常会出现：

- 状态越来越胖
- prompt 越来越长
- 关键上下文与低价值上下文混在一起
- 多轮任务难以稳定恢复

### 10.2 需要区分三类 memory

#### A. Working Memory

这是一轮任务内部的短时工作记忆。

当前项目已经有部分实现，但仍不够结构化。

它应主要保存：

- 当前问题表示
- 已识别实体及候选解释
- 当前任务类型
- 已生成比较维度或分析框架
- 已尝试的检索路径
- 已知失败原因
- 当前可用证据摘要

这是当前最值得优先加强的一层。

因为很多失败并不是“系统忘了”，而是：

- 系统从一开始就没有把该记住的工作状态结构化下来

#### B. Session Memory

这是一段会话内的跨轮记忆。

它应主要保存：

- 用户已经确认过的实体解释
- 默认比较口径
- 上一轮澄清结果
- 会话内已经接受的任务约束
- 会话内的近期结论与未完成任务

如果项目后续希望支持：

- “上一句说的夜刀是异格”
- “按泛用性而不是纯伤害比较”
- “继续刚才的分析”

那么这一层就应进入实际设计。

相比长期记忆，这一层的收益更直接，也更能解决复杂 agent 真实使用中的上下文稳定性问题。

#### C. Long-Term Memory

这是跨会话、长期保留的记忆。

它可能包括：

- 用户长期偏好
- 常用术语
- 常见任务模板
- 持续项目上下文
- 历史知识构建结果

但对当前项目阶段而言，这一层还不是最优先的。

原因是：

- 当前主要问题仍是任务理解、证据检索和放行治理
- 在这些基础能力还不稳时，长期记忆容易放大错误
- 系统可能会更稳定地记住错误的实体解释、错误的任务框架和错误的工具偏好

### 10.3 当前推荐顺序

建议按以下顺序建设：

1. 强化 `working memory`
2. 设计并接入 `session memory`
3. 最后再考虑 `long-term memory`

这意味着下一阶段更应该优先建设：

- 结构化中间状态
- 会话级澄清结果复用
- 任务恢复与 checkpoint

而不是优先做“长期用户画像”。

### 10.4 Memory 不是特判系统

需要特别强调：

memory 的目标不是把每一次错误都记成规则。

真正应该被 memory 保存的是：

- 结构化决策
- 澄清结果
- 已证实的事实
- 当前会话约束

而不是：

- 为某个具体领域临时写的特判
- 为某个具体用户问题保存的脆弱 prompt 痕迹

否则 memory 会退化成另一种更难维护的 heuristic cache。

## 11. 中文模型与轻量检索模型建议

当前项目默认 embedding 模型是：

- `sentence-transformers/all-MiniLM-L6-v2`

从中文 wiki 语料和本地 RAG 的场景看，这个选择并不理想。

后续确实应考虑替换为更适合中文检索的 embedding 模型，并尽早引入 reranker。

### 11.1 选型原则

对于当前项目，更重要的不是“最强模型”，而是：

- 中文效果明显更稳
- 体量适中
- 易于用 `sentence-transformers` / Hugging Face 接入
- 能与 reranker 组合
- 推理成本与当前项目规模匹配

### 11.2 推荐优先级

#### 方案 A：最稳妥的轻量升级

- Embedding: `BAAI/bge-small-zh-v1.5`
- Reranker: `BAAI/bge-reranker-base`

适用情况：

- 希望尽量少改现有代码
- 希望先把中文检索从“明显不合适”升级到“够用且轻量”
- 本地资源有限

为什么适合当前项目：

- `bge-small-zh` 的官方卡片已明确建议切到 `bge-small-zh-v1.5`
- v1.5 版本专门改善了 similarity distribution
- 对中文检索是直接对口的
- 体量比 `base` / `large` 更轻
- `bge-reranker-base` 官方标注为中英轻量 reranker，部署和推理都相对容易

#### 方案 B：当前阶段更推荐的平衡方案

- Embedding: `BAAI/bge-base-zh-v1.5`
- Reranker: `BAAI/bge-reranker-base` 或 `BAAI/bge-reranker-v2-m3`

适用情况：

- 希望在效果与成本之间取得更平衡的结果
- 本地知识库是中文为主
- 允许比 small 更高一点的推理成本

为什么我更推荐它作为主线候选：

- 官方 C-MTEB 表格里，`bge-base-zh-v1.5` 在中文基准上的整体表现明显高于 small
- 但资源成本又明显比 large 和更重的多语模型更好控制
- 对你这种以中文 wiki + 本地知识库为主的项目，通常比 current model 更合适，也比一上来就上 M3 更务实

#### 方案 C：为后续多语言 / 长文 / 混合检索预埋能力

- Embedding: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`

适用情况：

- 后续明确要做多语言
- 想支持更长文档
- 想探索 dense + sparse + multi-vector 的统一路线
- 准备把 retrieval pipeline 做得更完整

它的优势在于：

- 官方模型卡明确支持 100+ 语言
- 最大长度到 8192 tokens
- 同时覆盖 dense / sparse / multi-vector 三类常见检索方式

但它并不是当前最轻量的方案。

所以对你这个项目阶段，我更把它看作：

- 中期目标模型

而不是：

- 当前最先落地的低风险替换

### 11.3 对当前项目的实际建议

如果要一个最务实的落地顺序，我建议：

1. 先把当前 embedding 从 `all-MiniLM-L6-v2` 切到 `BAAI/bge-base-zh-v1.5`
2. 再引入 `BAAI/bge-reranker-base`
3. 等 retrieval eval 跑起来后，再判断是否升级到 `bge-reranker-v2-m3`
4. 如果后续明确要做多语言与长文混合检索，再考虑整体迁移到 `bge-m3`

这样做的好处是：

- 改动路径平滑
- 收益可验证
- 不需要一开始就把检索架构与模型栈同时大改

### 11.4 为什么不建议当前直接追求“最强大模型”

当前更大的瓶颈并不只在 embedding。

还包括：

- MediaWiki 清洗噪声
- section-aware retrieval 缺失
- query representation 漂移
- reranker 缺失
- answerability judgment 不成熟

如果这些不先解决，即使直接换到更重的模型，收益也会被大幅稀释。

因此更合理的策略是：

- 先用更适合中文的轻量模型把底座做对
- 再通过 reranker 和 retrieval policy 把系统做稳
- 最后再决定是否进一步升级到更重的模型

### 11.5 与当前代码的接入成本

就当前代码而言，接入 BGE 系列的技术阻力并不高。

原因是现有实现已经基于：

- `sentence-transformers`

而 BGE 官方卡片也直接给出了：

- Sentence-Transformers 用法
- Hugging Face Transformers 用法

因此从工程角度看，第一步通常只需要：

- 替换默认 embedding model name
- 为 query 侧补上更适合 retrieval 的 instruction 策略
- 在 reranker 层新增 top-k 重排

相比大规模重构，这是成本较低、收益较高的一步。

### 11.6 参考模型卡与官方资料

后续如果要继续细化选型，建议优先参考以下官方资料：

- FlagEmbedding 官方仓库：
  `https://github.com/FlagOpen/FlagEmbedding`
- BGE v1 / v1.5 官方说明：
  `https://bge-model.com/bge/bge_v1_v1.5`
- `BAAI/bge-base-zh-v1.5`：
  `https://huggingface.co/BAAI/bge-base-zh-v1.5`
- `BAAI/bge-small-zh-v1.5`：
  `https://huggingface.co/BAAI/bge-small-zh-v1.5`
- `BAAI/bge-m3`：
  `https://huggingface.co/BAAI/bge-m3`
- `BAAI/bge-reranker-base`：
  `https://huggingface.co/BAAI/bge-reranker-base`
