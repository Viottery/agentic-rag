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
- semantic reranker 的第一版接入
- 基础的 grounding / verification 思路

但从成熟 agent 产品视角看，当前系统仍然更像：

- 一个“节点编排 + 检索实验 + in-process 执行包装”的可调试框架

而不是：

- 一个“有 fast path、execution agent、shell runtime、skill registry、service/API 边界和风控层”的 agent runtime

换句话说，当前最大的差距已经不再是“有没有 agent 味道”，
而是“有没有产品级 runtime 边界与执行治理”。

## 2. 当前做得对的地方

在分析差距之前，需要先明确当前已经做对的部分。

### 2.1 本地 RAG 已经有真实基础

当前本地 RAG 并不是单纯的向量 top-k：

- 层级 metadata 已经进入 payload
- hybrid retrieval 已经形成
- semantic reranker 已开始接入
- scope routing 已有基础接口
- evidence 输出链路已经存在

这意味着本地 RAG 很适合被进一步下沉为独立 service，
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

### 3.1 缺少稳定 fast path，简单问题成本过高

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
- 哪些问题才值得进入 planner loop

### 3.2 Planner 与执行边界仍然不够清晰

这是架构层的核心问题。

当前系统虽然已经有 planner，
但整体仍然带有很强的“planner 贴近底层执行细节”的痕迹。

这样的问题在于：

- planner 很容易被底层实现细节拖累
- prompt 中会不断堆积调用知识
- 子任务执行端不容易保持原子化
- 全局决策和局部执行混在一起

成熟的 agent runtime 更应该是：

- Planner 只做全局判断
- execution agent 只做单一职责执行

也就是说：

- Planner 决定“做什么”和“交给谁做”
- execution agent 决定“如何完成当前原子任务”

### 3.3 skill 还没有收敛成真正的 registry

这是当前产品化边界上最关键的缺口之一。

当前系统中的“skill”更接近：

- in-process 执行包装

这对产品化不利。

原因是：

- skill 与实现耦合过深
- planner 和 runtime 难以只消费稳定 metadata
- prompt 难以统一引用 skill 能力
- 跨平台调用说明难以组织

成熟产品更需要的是：

- skill 作为组织化调用知识层

也就是：

- `manifest`
- `prompt package`
- `invocation guide`
- `service binding`

而不是：

- skill 自己提供服务

### 3.4 缺少 shell-first 的执行基座

如果 execution agent 不能稳定使用 shell，
系统会出现一个常见问题：

- 明明需要和本地环境、CLI、脚本、文件系统、外部接口交互
- 但所有事情都被挤回 prompt 或 graph 节点中处理

这会造成：

- 执行能力弱
- 观测性差
- 复用已有工具困难
- Linux / Windows 兼容性差

成熟 agent 产品更需要的是：

- shell 作为 execution agent 的主要执行 substrate

但这并不意味着“裸奔 shell”，
而是：

- shell + policy engine + structured logging

### 3.5 Service / API 边界还不够独立

当前本地 RAG、搜索等能力已经有真实实现，
但它们仍然偏向：

- 直接嵌在应用内
- 通过节点或 Python 包装暴露

这对后续产品化会带来几个问题：

- 调用边界不稳定
- skill 与能力实现不易解耦
- 跨 runtime 或跨部署形态复用困难
- shell 调用入口不统一

成熟产品更需要：

- service/API 提供真实能力
- skill 只描述如何调用和如何使用

### 3.6 跨平台执行策略还没有被正式定义

如果未来要同时支持 Linux 和 Windows，
但现在没有明确规范：

- skill 如何给出不同平台的调用模板
- execution agent 如何做平台感知
- 如何避免 bash / powershell 的转义差异

那么后续扩展会很容易失控。

成熟产品不会试图把所有命令压成“一条万能字符串”，
而会建立：

- 平台感知模板
- 统一调用协议
- 文件式输入输出交换

### 3.7 Shell 风控还停留在构想阶段

如果 shell 会成为主要执行通道，
那风控就必须是产品功能，而不是一句“请小心执行命令”。

当前主要缺口包括：

- 命令风险分级
- 文件系统和网络作用域控制
- 环境变量继承与 secret 屏蔽
- 破坏性命令拦截
- 资源限制
- 命令审计

没有这些层，shell 越强，风险越高。

### 3.8 验证层仍未升级为真正的 grounding validator

### 3.9 Planner 缺少对空转链路的抑制

这是当前运行时成熟度里一个很现实的问题。

在“先检索，再计算/执行”的问题上，当前系统可能出现：

- 对同一实体、同一 scope 的近似重复 RAG 子任务
- 在 action/tool 结果已经明确是 degraded 或 mock 时，继续重复派发类似 action
- 最终靠 budget limit 或后置 validator/checker 才被动收束

这类问题带来的影响包括：

- 响应时间显著变长
- 无效 LLM 调用增加
- 用户会感觉系统在“想很多，但没有产生新信息”
- trace 虽然可观测，但暴露出 planner 缺乏止损策略

从产品视角看，这不是简单的“提示词不够好”，
而是 runtime policy 缺口。

成熟 runtime 至少需要：

- 重复子任务判重
- scope 级检索去重
- degraded/mock action 熔断
- 当能力缺口已经明确时，快速收束到 best-effort answer

### 3.10 缺少 conversation-aware 的上下文系统

当前系统仍然更接近：

- 单请求执行
- 单次状态返回
- 缺少真正的 conversation 边界

这会带来几个直接问题：

- 不同话题线程无法稳定隔离
- 短期上下文只能依赖当前请求状态
- 长期记忆没有稳定入口
- planner / execution / validator 的高层输出难以沉淀
- 很难在产品层提供真正连续的多轮协作体验

成熟 agent 产品需要的不只是聊天记录，
而是：

- `conversation`
- `turn`
- `execution trace`
- `turn summary`
- `memory note`

这五类对象共同构成上下文基础设施。

### 3.11 缺少多会话并发与单会话顺序保证

如果没有 conversation queue，
后续异步化会很容易出现上下文竞争问题。

典型风险包括：

- 同一 conversation 内多个请求同时修改 rolling summary
- 长期记忆写入顺序混乱
- planner 看到的不是稳定的最近状态
- 用户在同一线程里触发前后 turn 交错

因此成熟 runtime 需要明确：

- 多个 conversation 可以并发
- 同一 conversation 的 turn 必须串行
- 不依赖顺序的后处理才适合异步 fan-out

从产品视角看，当前“验证”还偏轻。

当前重点更接近：

- 有没有 citation
- 覆盖率够不够
- 有没有明显 unsupported paragraph

这当然有价值，但还不足以支撑成熟 agent 产品。

成熟产品需要 validator 检查的是：

- 执行结果是否真的完成了子任务
- 最终答案是否真的被这些结果支撑
- 哪些结论只是推断
- 哪些内容超出了证据边界
- 是否应该回到 planner 继续补任务

不过在当前阶段，
validator 的升级优先级仍低于：

- execution agent 边界
- shell runtime
- skill registry
- service 化

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
- “还是应该先收敛 planner / execution agent / shell / skill registry / service 这些 runtime 边界”

本文的结论非常明确：

- 下一阶段不应继续主要围绕旧 graph 复杂度扩张
- 而应优先转向 `fast path + execution agent + shell runtime + skill registry + service/API`

## 5. 优先级建议

### 5.1 第一优先级：引入稳定 fast gate

这是平均时延、成本和体验最直接的优化杠杆。

重点不是追求极其复杂的分类器，
而是先明确最小决策集：

- `direct_answer`
- `planner_loop`

只要这一步成立，系统的平均成本就会显著下降。

### 5.2 第二优先级：明确 planner 与 execution agent 边界

这是 runtime 成熟度最关键的一步。

目标是让：

- planner 不再贴近执行细节
- execution agent 承接 shell 和 skill 检索

这一步收敛之后，后面的 skill registry 和 service 化才会更稳定。

### 5.3 第三优先级：把 skill 重构为 registry entry

建议 skill 尽快统一成稳定 schema，
至少包含：

- 能力摘要
- 适用条件
- prompt 资源
- 调用模板
- service binding
- 平台适配信息

这样 skill 才能真正被检索、索引和注入 prompt。

### 5.4 第四优先级：将本地 RAG 下沉为 service/API

当前最值得先 service 化的能力就是本地 RAG。

原因：

- 它已经是项目最真实、最稳定的能力
- 内部步骤已经够丰富，值得独立封装
- 对 execution agent 的价值最大

建议尽快形成：

- `local_rag_service`

让它统一负责：

- routing
- hybrid retrieval
- rerank
- evidence packaging

### 5.5 第五优先级：建立 shell policy engine

如果 execution agent 以 shell 为主通道，
那风控必须同步建设。

否则很容易出现：

- 执行能力增强了
- 但平台风险也同步放大

因此：

- shell 能力和 shell policy 应该成对推进

### 5.6 第六优先级：继续稳固本地 RAG 质量

虽然架构主线已经变化，
但 retrieval quality 仍然非常重要。

当前应继续推进：

- 语料清洗
- 层次化路由稳定性
- hybrid retrieval 候选控制
- semantic reranker 调优
- evidence packaging 质量

原因很简单：

- 如果底层 service 本身质量不稳，再好的 planner 和 execution agent 也会被拖累

## 6. 当前不建议优先投入的方向

### 6.1 不建议继续把主要精力放在旧 planner prompt 微调上

当前最缺的已经不是：

- planner 再多几个技巧

而是：

- planner 的职责边界收敛
- execution agent 的执行边界建立
- skill registry 形成

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
但当前不应该成为 shell-first runtime 和 service 化的阻塞项。

它更适合继续保留在 TODO，
等到主架构收敛之后再推进。

## 7. 建议的阶段化路线

### Phase 1：文档与架构认知统一

目标：

- 先统一系统设计语言

内容：

- 明确 fast path
- 明确 planner 和 execution agent 边界
- 明确 skill registry
- 明确 service/API 边界
- 明确 shell policy

### Phase 2：最小 runtime 重构

目标：

- 让新架构先跑起来

内容：

- 引入 fast gate
- 切出 execution agent
- 定义 skill registry schema
- 将本地 RAG 拆到独立 service

### Phase 3：跨平台调用与风控

目标：

- 让执行层既强又可控

内容：

- 设计统一 CLI 调用协议
- 增加 Linux / Windows 平台适配模板
- 建立 shell policy engine
- 增强执行日志和审计

### Phase 4：验证层升级

目标：

- 让系统能判断“执行结果是否真的支撑答案”

内容：

- 将 verifier / checker 重构为 grounding validator
- 增强结果-答案关联性检查
- 增强 planner feedback

### Phase 5：扩展 skill 生态与 runtime 能力

目标：

- 从本地 RAG runtime 扩展到更完整的 agent runtime

内容：

- `web_search_service`
- `tool_execution_service`
- 更成熟的任务治理
- 更稳定的 API 与 trace 输出

## 8. 一句话结论

当前项目最缺的已经不是“再多一些 agent 节点”，
而是：

- 更短的简单问题路径
- 更清晰的 planner / execution 边界
- 更稳定的 skill registry
- 更明确的 service/API 边界
- 更强的 shell runtime 与风控能力

因此，下一阶段最关键的事情不是继续堆旧 workflow，
而是把系统真正重构成一个：

- `fast path + planner + execution agent + shell runtime + skill registry + service/API`

的产品级 agent runtime。
