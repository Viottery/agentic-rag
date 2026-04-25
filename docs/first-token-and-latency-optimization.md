# First Token And Latency Optimization

本文档用于沉淀当前项目在“首 token 体验”和“整体响应速度”上的优化方向。

这份文档不是在讨论“要不要继续用 graph”，
而是在讨论：

- 如何让必要的 graph 不再把首个可见输出压到整轮末尾
- 如何让简单问题走更短、更稳定的 hot path
- 如何把慢步骤前置、缓存、并行或后置
- 如何把 `Claude Code` 一类成熟 agent runtime 的提速方法迁移到本项目

## 1. 当前判断

当前系统的主要问题不是“步骤太多所以一定要删图”，
而是：

- 第一个对用户可见的输出出现得太晚
- 简单本地知识库问题仍然要经过偏重的阻塞链路
- 一些质量保障步骤仍然压在首屏关键路径上
- 热路径里仍然存在额外的进程与序列化开销

这几个因素叠加后，
用户感知到的就不是“系统在认真工作”，
而是“系统在长时间无响应”。

## 2. 当前瓶颈诊断

### 2.1 目前并没有真正的首 token 路径

当前 [app/api/routes/chat.py](../app/api/routes/chat.py) 只支持两种模式：

- `mode=wait`
- `mode=background`

这意味着：

- `mode=wait` 必须等整轮执行完才返回
- `mode=background` 只能立刻返回 job ack，而不是回答内容

因此当前 `/chat` 严格来说并没有真正的 `TTFT (time to first token)`，
只有：

- `time to background ack`
- `time to full response`

如果没有流式通道，
首 token 优化在 API 语义上就无从成立。

### 2.2 简单本地 KB 问答的热路径仍然偏重

当前简单问答虽然已经能走 `single_skill + local_rag_program`，
但从主图和本地 RAG 子流程来看，
依然可能经历以下阻塞阶段：

```text
/chat(wait)
  -> queue
  -> graph
  -> fast_gate
  -> task dispatch
  -> local_kb_retrieve_service
  -> local_rag_program
  -> query_refiner
  -> rag_router
  -> rag_agent
  -> answer_generator
  -> checker / validator
  -> return full result
```

其中真正对用户可见的内容出现在非常靠后的位置。

这意味着即便 graph 结构本身合理，
也会因为“回答生成与返回太晚”而造成明显迟滞。

### 2.3 本地 RAG 热路径还有额外的 subprocess hop

当前 [app/agent/services/local_rag_process_client.py](../app/agent/services/local_rag_process_client.py)
会在服务内热路径上执行：

- 临时 `request.json`
- 直接子 Python 进程
- 再由客户端走本地 RAG 服务 endpoint 调用
- 临时 `response.json` 回传

这条链路适合做解耦和兼容，
但不适合作为延迟敏感的默认热路径。

对简单 KB 问题来说，
这部分额外 hop 会显著拖慢可见响应。

### 2.4 质量保障步骤仍然压在首屏关键路径

当前 [app/agent/nodes.py](../app/agent/nodes.py) 中，
`answer_generator` 之后仍然有 `checker`、`validator` 等步骤。

这些步骤的价值很高，
但它们不一定都应该阻塞“第一个对用户可见的答案”。

如果把所有质量治理都放在首屏前面，
系统就会天然偏向“慢而完整”，
而不是“先可见、后增强”。

### 2.5 缺少分阶段延迟观测

当前更容易看到的是整轮耗时，
但要真正优化，
必须把延迟拆开。

至少需要独立记录：

- `queue_wait_ms`
- `routing_ms`
- `local_rag_hop_ms`
- `retrieve_ms`
- `answer_generation_ms`
- `post_check_ms`
- `time_to_first_event_ms`
- `time_to_first_answer_chunk_ms`
- `time_to_complete_ms`

没有这些细粒度指标，
优化就会退化为感觉驱动。

## 3. 从 Claude Code 可以直接迁移的方法

参考 `Claude Code` 这类成熟 agent runtime，
最值得迁移的不是某一个节点，
而是以下几条性能方法论。

### 3.1 强化 fast path，而不是默认全量装配

成熟 runtime 的共识不是“所有请求都先进入完整系统”，
而是：

- 先分流
- 再装配
- 只对高复杂度问题支付高成本

对本项目而言，
这意味着：

- 简单问答不该默认进入重型链路
- 明显的本地 KB 问题不该先走多次 LLM 判断
- action / shell / planner loop 只在必要时触发

### 3.2 会话级缓存优先于逐轮重算

成熟系统不会在每一轮都重新计算所有上下文部件。

应优先把下列内容做成 conversation-scoped cache：

- knowledge base 结构摘要
- 稳定的 system/context prompt 前缀
- 最近轮次压缩结果
- `memory_notes` 检索结果
- tool / skill 元信息

原则是：

- 会话内尽量复用
- 只对真正变化的部分做增量更新

### 3.3 把 prompt 稳定性当成性能约束

如果模型供应商支持 prompt cache，
那么 prompt 的稳定顺序和稳定形状就会直接影响速度与成本。

需要尽量保持稳定的部分包括：

- system prompt 主体
- tool / skill 描述顺序
- conversation context block 的结构
- 质量策略提示的开关方式

换句话说，
不要让 prompt 因为一些小状态变化频繁抖动。

### 3.4 把慢步骤藏到 streaming 期间

成熟 agent runtime 很少把所有工作都放到“回答开始之前”。

更常见的做法是：

- 一边 streaming，一边准备后续上下文
- 一边输出草稿，一边做 memory / citation / summary 处理
- 让用户先看到内容，再让系统补完结构化治理

这背后的核心思想是：

- 用模型生成时间覆盖系统工作时间

### 3.5 并发只给安全并发的阶段

成熟系统不会盲目并发所有步骤，
而是只并发“只读、无副作用、不会污染上下文”的工作。

对本项目适合并发或后置的工作包括：

- citation mapping
- trace summary writeback
- memory note extraction
- turn summary writeback
- 一些只读型后处理

不适合乱并发的则包括：

- 会修改主状态的关键决策步骤
- 依赖前一步结果的连续规划步骤

### 3.6 先做 cheap collapse，再做 heavy compact

上下文过长时，
不应该立刻进入重型总结。

更稳妥的顺序应是：

1. 先收紧 recent window
2. 再依赖 `rolling_summary`
3. 再补 `recent_turn_summaries`
4. 只有确实还超限时才做更重的 compact

这样不仅更快，
也更有利于保留上下文细节。

## 4. 面向本项目的具体优化方向

### 4.1 先建立真正的 `/chat/stream`

这是首要前提。

没有流式接口，
首 token 优化就只能停留在内部计算速度上，
无法转化成用户感知。

建议新增：

- `/chat/stream`

建议输出事件：

- `accepted`
- `queued`
- `routing_started`
- `retrieval_started`
- `draft_answer_started`
- `answer_chunk`
- `post_check_started`
- `completed`
- `failed`

第一阶段甚至不必一上来就做 token 级 streaming。
只要先让用户看到结构化进度事件，
体感就会明显改善。

### 4.2 引入 draft-first，严格校验后置

对于低风险路径，
建议把回答拆成两层：

- `draft answer`
- `strict post-check`

也就是说：

- 先把 `answer_generator` 的结果流给用户
- 再异步做 `checker` / `validator` / citation 整理

保守模式下可以保留：

- `strict=true` 时继续阻塞到完整校验结束

但默认模式更应偏向：

- 先给可用回答
- 再补强约束与校验

### 4.3 为简单本地 KB 问题建立更短 hot path

当前本地 RAG 路径中，
`query_refiner` 和 `rag_router` 仍然容易成为固定成本。

更适合的策略是：

- 对明显的本地实体问答，先做直接检索
- 命中不足、scope 不清或证据弱时，再升级到 query rewrite / route

也就是说：

```text
simple local kb ask
  -> direct retrieve
  -> answer draft
  -> fallback to refine/router only when needed
```

而不是：

```text
simple local kb ask
  -> refiner
  -> router
  -> retrieve
  -> answer
```

### 4.4 去掉默认热路径中的子进程 hop

对应用内 `/chat` 路径，
建议把默认调用改成：

- app process
  -> in-process local rag client
  -> local RAG service endpoint

而不是：

- app process
  -> subprocess client
  -> child python client
  -> local RAG service endpoint

subprocess 包装可以保留作为：

- 调试路径
- CLI 兼容路径
- 故障隔离路径

但不应继续作为默认热路径。

### 4.5 把非关键写回移出首屏路径

以下工作原则上都不需要阻塞首个可见回答：

- conversation trace 落盘
- turn summary 生成
- memory note 提取
- recall index 更新
- 只读型 citation 对齐

这些步骤可以：

- 在 streaming 期间并行做
- 在回答返回后异步收尾

### 4.6 为不同请求类型设定不同延迟预算

不要让所有请求共用一条相同的延迟哲学。

建议至少分三档：

- `fast_answer`
  目标是最低首 token 延迟，尽量单次生成
- `single_skill_local_kb`
  目标是先给草稿回答，再补质量治理
- `planner_loop`
  允许更长延迟，但必须用进度事件持续反馈

这比“统一慢路径”更接近成熟产品的运行方式。

### 4.7 把 warmup 提前到 conversation 或进程生命周期

建议把以下内容做预热：

- 会话首轮前的 KB 结构摘要
- 常用 prompt 模板装配
- 当前 conversation 的短期上下文 bundle
- 可能用到的 skill / tool registry

如果这些工作等到真正需要回答时才开始，
就会把本可提前支付的成本全部堆到首屏前。

## 5. 建议引入的延迟指标

为了让后续优化可以验证，
建议统一记录以下指标：

- `tt_ack_ms`
- `tt_first_event_ms`
- `tt_first_answer_chunk_ms`
- `tt_complete_ms`
- `queue_wait_ms`
- `context_prepare_ms`
- `fast_gate_ms`
- `local_rag_client_ms`
- `retrieve_ms`
- `answer_generator_ms`
- `checker_ms`
- `validator_ms`

同时需要对以下维度做分桶：

- `request_class`
- `path_selected`
- `conversation_turn_index`
- `tool_count`
- `retrieval_scope`

只有把 latency 与 path 映射起来，
后续才知道该优化哪一段。

## 6. 分阶段落地建议

### 6.1 第一阶段

第一阶段只追求最明显的体感收益。

建议优先做：

1. 新增 `/chat/stream`
2. 增加 `time_to_first_event_ms` 与 `time_to_complete_ms`
3. 将 `answer_generator` 结果提前可见
4. 将 `checker` / `validator` 从默认首屏路径后移

这一阶段的目标不是“让系统绝对更快”，
而是先把“长时间无反馈”改成“快速进入可见状态”。

### 6.2 第二阶段

第二阶段开始优化热路径结构。

建议优先做：

1. 为简单 local KB ask 建立 direct retrieve fast path
2. 去掉默认热路径中的 local rag subprocess hop
3. 建立 conversation-scoped context cache
4. 稳定 prompt prefix 与 skill/tool 顺序

这一阶段的目标是：

- 同时改善 TTFT 与整轮耗时

### 6.3 第三阶段

第三阶段再进一步压缩系统性开销。

建议推进：

1. 在 streaming 期间做 summary / memory / citation 预处理
2. 建立 cheap collapse -> heavy compact 的上下文控制策略
3. 对只读型后处理做安全并发
4. 为不同路径建立独立 latency budget 与回退策略

这一阶段的目标是：

- 让系统不仅“首屏快”，也“长会话下仍然稳定快”

## 7. 最终设计判断

对当前项目而言，
首 token 优化不应被理解为：

- 大幅删除 graph
- 取消验证
- 牺牲结构化治理

更合理的方向是：

- 保留必要结构
- 把可见输出前移
- 把慢步骤前置、缓存、并发或后置
- 让简单请求和复杂请求走不同延迟策略

一句话总结：

> 当前最该做的不是“继续减节点”，而是把系统从“整轮完成后才可见”改造成“先快速可见，再渐进补全质量”的 agent runtime。
