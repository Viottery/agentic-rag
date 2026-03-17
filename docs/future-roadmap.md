# Future Roadmap

本文档用于整理 Agentic-RAG 下一阶段的发展方向。

这份 roadmap 分成两层：

- 近期：围绕 RAG 的实际实现与工程落地
- 长远：围绕整个 agent 系统的能力边界、运行模型与演化目标

这两层的关系是：

- 近期工作解决“现在怎样把检索系统做对、做稳、做可信”
- 长远设计解决“这个项目最终要演化成什么样的 agent 助手系统”

当前阶段暂时不新增新的主目标，只把已有方向整理得更清晰。

## 0. 当前快照

在进入下一阶段之前，当前系统已经完成了这一轮关键骨架建设：

- `rag_agent` 已接入真实本地知识库检索
- `search_agent` 已接入 Tavily，并具备第一版 web-RAG 路径：
  - search
  - extract
  - chunk
  - rerank
- `answer_generator -> citation_mapper -> verifier -> checker` 已形成基础可信回答链路
- workflow 已具备预算限制、结构化输出失败重试和兜底能力
- 已开始区分“证据存在”与“证据是否适合被引用”

因此，下面的规划不是从零设计，而是在现有 supervisor-style graph 上继续结构化和扩展。

下一阶段的一个明确原则是：

- 尽量减少仅靠手工规则堆起来的启发式设计
- 优先采用更稳定的检索 pipeline、排序 pipeline、路由元数据设计和图编排能力
- 把启发式保留在兜底和安全边界上，而不是让它成为主实现

## 1. 近期：RAG 相关实现

近期工作的核心目标有四条：

1. 本地 RAG 从单层检索升级为层次化检索与路由
2. 在线搜索 RAG 成为本地 RAG 的补充能力，并针对简单任务优化路径
3. 在不牺牲可信度的前提下，逐步让检索与回答链路更贴近成熟 retrieval pipeline
4. 利用 LangGraph 的并行 fan-out / fan-in 让复杂检索与搜索任务可并行执行

### 1.1 本地 RAG：层次化知识库与逐层路由

这是近期最重要的主线。

当前本地 RAG 的主要问题不是“不能检索”，而是：

- 检索范围还是单层向量搜索
- 文件夹结构与知识域结构尚未显式建模
- Qdrant 中缺少能够支撑逐层路由的结构化元数据
- planner / query_refiner 还无法稳定地先缩小范围、再进入细检索

近期目标是把本地知识库从“文档集合”升级为“带结构的知识域系统”。

#### 目标形态

本地知识库应当同时具备两层组织方式：

- 文件系统中的层级结构
- 向量数据库中的对应层级结构与元数据索引

每个知识层级都应附带结构化描述，例如：

- `kb_id`
- `parent_kb_id`
- `path`
- `title`
- `summary`
- `scope`
- `keywords`
- `document_types`
- `update_frequency`
- `language`

这样做的目的不是单纯“补 metadata”，而是为了支撑分层路由：

```text
user question
  -> coarse route to knowledge domain
  -> narrow to sub-domain / folder group
  -> retrieve candidate documents
  -> retrieve chunk-level evidence
  -> answer with citations
```

#### 近期实施重点

- 为本地知识库定义更清晰的层次化 metadata schema
- 让 index pipeline 在写入 Qdrant 时保留层级信息
- 在 retrieval 前引入粗路由步骤，而不是直接全局向量检索
- 让 `rag_agent` 支持“先选范围，再检索内容”的两阶段逻辑
- 让 `query_refiner` 或后续专门的 retrieval router 能利用知识域描述，而不是只做 query 表面改写

#### 多路召回与重排序

本地 RAG 的近期规划不应停留在“路由完再做一次向量检索”，而应逐步升级到更成熟的 retrieval pipeline：

- 多路召回
  - 结构化 metadata 过滤召回
  - 向量召回
  - 关键词 / BM25 类召回
  - 必要时保留基于标题、路径、summary 的浅层召回
- 召回结果融合
  - 去重
  - 分源归一化
  - 候选集合截断
- 重排序
  - 对候选文档级结果做一次重排
  - 对 chunk 级结果再做更细粒度重排

这样做的目标不是把 pipeline 变复杂，而是减少“单一相似度搜索一把梭”的不稳定性。

这里需要特别区分两类“重排序”：

- 召回融合后的规则化/算法化排序
  - 例如向量召回、BM25 召回、标题命中、路径命中、RRF 融合、document-focused 二次筛选
  - 这类排序可以显著提升“找对文档”的稳定性
- 真正的 query-document semantic reranking
  - 输入是“用户问题 + 候选文档/候选 chunk”
  - 输出是一个直接表示二者语义相关性的分数
  - 常见实现形态是 cross-encoder 或专门的 reranker model

后者应当被视为本地 RAG 近期规划中的重要增强方向，而不是可有可无的优化项。  
原因在于：

- 多路召回更擅长“把可能相关的内容召回上来”
- 真正的 semantic reranker 更擅长“从这些候选里挑出最适合回答当前问题的段落”

对于实体型问题、人物介绍类问题、背景说明类问题，这种差异尤为明显。  
系统可能已经成功命中了正确 document，但如果没有 query-document semantic reranking，仍然会在该 document 内选中不适合回答问题的 chunk。

因此，本地 RAG 更完整的目标形态应理解为：

```text
question
  -> route to knowledge scope
  -> multi-recall inside selected scope
  -> candidate fusion
  -> document-level rerank
  -> chunk-level semantic rerank
  -> evidence selection
  -> answer
```

这里的 `chunk-level semantic rerank` 指的是传统意义上的“同时考虑用户问题与候选文本语义相关性”的重排序，而不是仅靠标题匹配、词项匹配或召回通道融合。

#### 当前边界说明

当前实现可以逐步具备：

- 向量召回
- BM25/lexical 召回
- 标题与路径参与候选排序
- 召回结果融合
- document-focused 二次筛选

但这仍然不等同于“已经具备真正的 semantic reranker”。  
在文档、设计和评估中应明确区分：

- `hybrid retrieval / fusion ranking`
- `query-document semantic reranking`

前者已经是当前演化方向的一部分，后者仍应作为后续设计目标继续推进。

更具体地说，近期本地 RAG 应逐步形成这样的处理顺序：

```text
question
  -> route to knowledge scope
  -> multi-recall inside selected scope
  -> candidate fusion
  -> rerank
  -> chunk selection
  -> evidence output
```

#### 设计原则

本地 RAG 这条线尽量遵守以下原则：

- 优先使用结构化 metadata 和明确的索引设计，而不是在 prompt 中临时猜范围
- 优先使用可解释的召回与重排 pipeline，而不是不断堆临时规则
- 启发式只作为 fallback，不作为主要检索策略

#### 近期不追求的事

- 暂时不追求特别复杂的多跳推理型 retrieval graph
- 暂时不追求把所有 routing 都压给 planner
- 暂时不追求在一个节点里塞满所有本地检索策略

更好的做法是：

- planner 决定“是否需要本地检索、要不要继续检索、需不需要补搜”
- 本地 retrieval router 决定“先走哪个知识层级、如何收缩范围”

### 1.2 在线搜索 RAG：作为补充型 retrieval

在线搜索不应默认替代本地 RAG，而应当成为两类情况下的补充路径：

- 本地 RAG 不足以回答问题
- 用户明确要求联网搜索或获取最新信息

这条原则非常重要，因为它决定了后续系统的成本和可信边界。

#### 目标形态

在线搜索 RAG 的职责不是“凡事都搜”，而是：

- 为本地证据缺口提供补充
- 为时间敏感问题提供新鲜外部信息
- 为复杂任务提供外部事实背景
- 在需要时与本地检索结果统一进入同一套 evidence / citation / verification 流程

目标流程可以概括为：

```text
question
  -> local RAG first
  -> if insufficient or explicitly requested:
       online search
       -> extract
       -> temporary chunking
       -> rerank
       -> merge into evidence
```

#### 近期实施重点

- 进一步明确 local-first / search-fallback 的策略
- 让用户显式要求“搜索一下”时，直接走 search 优先路径
- 让 simple search task 尽量减少无意义 planner 回环
- 继续提高搜索结果筛选质量、实体约束和 chunk 去噪
- 保持 citation 和 verifier 对网页证据的统一处理

#### 尽量采用成熟 pipeline，而不是过度启发式

在线搜索这条线尤其要避免变成“规则越写越多”的系统。

更好的方向是把 search 部分逐步做成相对成熟的 pipeline：

```text
query planning
  -> search
  -> result normalization
  -> dedup
  -> extraction
  -> chunking
  -> rerank
  -> evidence selection
```

这里的关键点是：

- `search` 负责找到候选来源
- `extract` 负责拿到更完整文本
- `chunking` 负责变成可筛选证据
- `rerank` 负责把最相关内容推到前面
- `evidence selection` 负责把少量高价值网页证据交给下游回答链路

这意味着未来在线搜索的主要优化方向应当是：

- 更好的 query planning
- 更稳定的结果标准化和去重
- 更可靠的 extract / chunk / rerank
- 更统一的 evidence schema

而不是不断堆新的手工排除规则。

#### 对简单任务的路径优化

无论是本地 RAG 还是在线搜索，近期都应考虑 simple path：

- 简单本地问答不一定需要完整 supervisor loop
- 简单搜索问题也不一定需要多轮 planner / checker 循环

这里的重点不是“写死一个简单问题清单”，而是逐步识别：

- 哪些问题可以在低风险下走更短路径
- 哪些问题必须保留完整可信链路

也就是说，未来的图路径应该是能力导向的，而不是“所有问题一视同仁地走最重链路”。

### 1.3 并行 fan-out / fan-in：复杂任务的近期图能力

当前系统的默认链路仍然是串行的，但 LangGraph 已经具备支撑更复杂图结构的能力。

近期非常值得明确纳入计划的一点是：

- 对复杂检索和搜索任务，逐步引入并行 fan-out / fan-in

这不是为了“看起来高级”，而是因为真实复杂任务天然适合并行：

- 本地多个知识域可以并行粗召回
- 本地召回与在线搜索可以并行跑，再在后面汇合
- 多个 search query / sub-query 可以并行抓取候选结果
- 多个来源的 extract / chunk / rerank 可以并行处理

一个更合理的复杂 retrieval/search 图形态可以是：

```text
planner
  -> fan-out
     -> local retrieval branch
     -> online search branch
     -> optional tool/info branch
  -> fan-in
     -> evidence merge
     -> rerank / validate
     -> answer
```

#### 近期并行化的重点场景

- 本地多知识域并行候选召回
- 在线 search query 与 sub-query 的并行搜索
- 本地 RAG 与在线 search 的并行证据收集
- 多来源网页抽取与 chunk rerank 的并行执行

#### 并行化的约束

并行不是目标本身，重点是：

- 用于缩短复杂问题的 wall-clock 时间
- 用于提高证据覆盖度
- 最终仍要在 fan-in 阶段统一 evidence、去重、重排和验证

所以并行 fan-out / fan-in 的价值不只是快，还包括更适合形成成熟的 retrieval/search pipeline。

### 1.4 可信回答仍然是近期硬约束

近期不论是本地 RAG 还是在线搜索，都不应为了快而放松可信要求。

当前可信回答链路已经具备基础形态：

```text
retrieve/search
  -> normalize evidence
  -> answer generation
  -> citation mapping
  -> verification
  -> checker
```

近期的重点不是再扩很多节点，而是继续收紧质量：

- 引用宁缺毋滥
- 低相关 chunk 不强贴
- 证据不足时允许回答更保守
- 搜索结果与本地检索结果都进入统一 evidence 语义

### 1.5 近期里“工具”与 RAG 的边界

近期虽然会开始考虑“建知识库”这种工具能力，但它不属于通用复杂工具 runtime 的完整形态。

在近期阶段，更适合把它看成：

- 一个与 RAG 强耦合的 ingestion / indexing 子流程
- 一个为了让层次化本地知识库可持续扩展而必须补齐的工程能力

也就是说，近期的建库能力主要服务于：

- 本地 RAG 结构化
- 本地知识库扩容
- 统一 metadata 与向量索引规范

## 2. 长远：系统设计目标

长远来看，这个项目的目标不应停留在“会检索、会回答”的 agentic RAG 系统，而是逐步演化为一个更完整的 agent assistant runtime。

它更接近：

- Codex 类 coding agent
- Claude Code 类本地协作型 agent
- OpenClaw 这类具备多工具、多步分析与执行能力的复杂 agent 助手

### 2.1 长远目标不是堆 workflow，而是建立 agent runtime

长远设计的关键不是不断往 graph 里硬编码更多专用 workflow，而是建立一套稳定的 agent runtime：

- 能规划任务
- 能调用信息工具
- 能调用本地环境工具
- 能读写和修改文件
- 能编写和执行代码
- 能把工具结果重新纳入推理
- 能围绕复杂问题多轮迭代

因此，后续的重点应该是：

- 固定节点角色和系统契约
- 让具体 workflow 尽量由 agent 根据任务自行规划
- 只把少量高副作用、长耗时、强工程约束的流程显式工程化

### 2.2 长远能力分层

从系统设计上，后续更适合把项目分成四层理解。

#### A. Planner / Supervisor 层

负责：

- 任务拆分
- 能力选择
- 风险平衡
- 预算控制
- 是否继续迭代

它决定：

- 现在是先查本地、再查外部，还是直接执行工具
- 是不是已经够回答
- 是不是要进入更复杂的工作流

#### B. Retrieval 层

负责：

- 本地层次化 RAG
- 在线搜索 RAG
- evidence 统一结构
- citation / verification 的检索输入

这层长期来看会成为 agent 的“知识与证据基础设施”。

#### C. Tool Runtime 层

负责：

- 工具注册
- 参数校验
- 权限控制
- 执行审计
- 风险分级
- 结构化结果返回

这里的工具不只是搜索，而会逐步覆盖：

- 文件系统工具
- 本地文本检索工具
- 代码读写工具
- 命令执行工具
- 外部信息获取工具
- 知识库构建工具

#### D. Task Workflow 层

负责：

- 将若干工具组合成可复用的工程化子流程
- 处理高副作用、长时、复杂任务

例如未来可能存在：

- 建知识库子流程
- 代码分析与修复子流程
- 大规模文档整理子流程

但这里需要注意：

- 这些是“工程化子流程”
- 不是说所有复杂问题都要提前写死成 workflow

### 2.3 为什么建知识库可以是显式子流程

用户已经明确指出，建知识库这类任务可以更显式地写成子流程。这一点是合理的。

原因在于它天然具备以下特点：

- 高副作用
- 长时运行
- 涉及目录遍历、抽取、清洗、分层整理、索引写入
- 输出结果需要长期复用，而不是只服务当前一轮回答

因此，长远来看，建知识库更适合是：

- 一个独立 workflow family
- 但运行时仍受统一 tool runtime、state 记录和审计约束

这与“一般问题由 agent 自行规划 workflow”并不矛盾。

### 2.4 工具系统的长期定位

工具系统未来不应只被理解为“能 function calling 一下”。

它更像是 agent 的执行层，未来会服务于三类需求：

#### 信息型需求

- 搜索网页
- 抽取页面
- 读取 wiki / MediaWiki
- 查询外部资料

#### 本地环境需求

- 读文件
- 写文件
- 修改文件
- 搜文本
- 处理目录

#### 执行型需求

- 跑命令
- 跑测试
- 执行脚本
- 生成并验证代码结果

长期约束建议保持不变：

- 工具结果尽量结构化
- 高风险工具有更严格边界
- 所有调用都有审计与 trace
- 工具结果可以被纳入统一 evidence / state

### 2.6 长期设计原则：少启发式，多结构化 pipeline

长远来看，整个项目都应尽量减少“靠人工补丁式启发式把系统糊住”的倾向。

更值得坚持的方向是：

- 在 retrieval 中优先做层次化索引、metadata 设计、多路召回、重排序
- 在 search 中优先做标准化、抽取、chunk 化、重排与证据选择 pipeline
- 在 graph 中优先利用 LangGraph 的并行 fan-out / fan-in 和清晰节点边界
- 在 tool runtime 中优先做统一 schema、权限设计和执行审计

启发式仍然会存在，但更适合作为：

- fallback
- 风险保护
- 低成本兜底

而不是长期主干实现。

### 2.5 workflow 描述原则

为了避免系统后续越写越死，文档层面建议坚持下面的表达原则：

- 描述稳定节点角色，而不是枚举所有未来任务流程
- 描述能力边界，而不是硬编码所有问题怎么走
- 描述数据契约，而不是把 prompt 中的一次性策略写成系统原则
- 对复杂任务强调“由 agent 规划”
- 对建知识库这类高副作用任务，承认其可以存在显式子流程

这也是为什么当前文档会把 graph 写成：

- 规划能力
- 检索能力
- 搜索能力
- 执行能力
- 可信回答能力

而不是写成几十条固定 workflow 分支。

## 3. 两层目标之间的关系

近期与长远不是两套路线，而是同一条路线的不同深度。

可以这样理解：

- 近期的层次化本地 RAG，是未来 agent 长期记忆与文档工作台的基础
- 近期的在线搜索补充路径，是未来 agent 外部信息入口的基础
- 近期的建库能力，是未来复杂工具工作流的第一个工程化样板
- 当前的 supervisor graph，是未来复杂 agent runtime 的第一版控制骨架

所以近期工作不只是“补功能”，它其实是在为更大的系统打地基。

## 4. 当前不新增目标，但要明确方向

这一阶段暂时不新增新的主目标，仍然沿用这三条主线：

1. 本地 RAG 结构化与层次化
2. 在线搜索作为补充型 retrieval，并为简单任务优化路径
3. 工具系统逐步演化为更完整的 agent runtime，其中建知识库是优先工程化的子流程之一

当前需要做的是把这三条线说清楚、做扎实，而不是把 roadmap 不断扩张。

## 5. 一句话总结

近期，项目要把 RAG 做成“有结构、能路由、支持多路召回与重排序、还能通过并行 fan-out/fan-in 处理复杂任务”的信息基础设施。

长远，项目要在这套基础设施之上，逐步演化为一个能规划、能检索、能调用工具、能处理代码与本地环境、还能对复杂问题做多步分析与执行的 agent assistant runtime。
