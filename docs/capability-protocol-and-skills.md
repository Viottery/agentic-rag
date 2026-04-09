# Capability Protocol And Skill Strategy

本文档用于定义项目下一阶段在 `tool`、`MCP` 与 `skill` 上的统一设计策略。

目标不是简单“增加更多工具”，
而是把系统正式收敛为：

- 模型输出负责声明意图
- runtime 负责解析、校验、授权与执行
- `MCP` 作为一种 transport-backed capability source
- `skill` 作为 prompt / workflow / invocation knowledge layer
- `service/API` 继续作为真实能力实现

这份文档可以看作是对现有 `skill registry + service/API + shell runtime`
路线的进一步收束。

## 1. 当前设计判断

下一阶段不应继续沿用以下混合边界：

- planner 直接知道底层工具执行细节
- skill 既负责描述调用方法，又直接藏着实现代码
- MCP 被当作一套旁路系统，而不是统一能力池的一部分
- tool 执行结果只以“内部 observation”存在，而不是显式 runtime artifact

更合理的边界应是：

- 模型只产出结构化的调用意图
- runtime 通过统一协议把调用意图映射到本地工具、service、shell 或 MCP
- 所有执行结果都显式回写为 `tool_result` / `observation`
- skill 不再直接等于执行器，而是“如何组织调用”的知识层

## 2. Tool Invocation Protocol

### 2.1 模型不直接执行环境

模型不应直接拥有：

- shell 执行权
- 文件系统调用权
- MCP transport 细节
- service client 细节

模型更适合输出：

- `tool_name`
- `input`
- `reason`
- 可选的 `expected_output`

也就是说，
模型应该只声明：

- “我现在想调用哪个能力”
- “我要传什么参数”

而不是直接决定：

- 如何连 transport
- 如何做权限判断
- 如何记录日志
- 如何将结果序列化回对话

### 2.2 Runtime 必须拥有统一 Tool 协议

无论最终能力来自：

- 本地 shell
- in-process Python / service client
- HTTP / socket service
- MCP server

都应该被统一包装成 runtime 可执行的 capability entry。

建议引入统一对象：

```text
CapabilityRegistryEntry
  - capability_id
  - name
  - kind
  - input_schema
  - output_schema
  - read_only
  - destructive
  - concurrency_safe
  - policy_class
  - transport
  - adapter
```

其中：

- `kind`
  可以是 `local_tool`、`service`、`shell`、`mcp`
- `adapter`
  负责把统一协议映射到真实执行实现

### 2.3 标准执行链路

建议统一执行链路为：

```text
model tool intent
  -> capability lookup
  -> input validation
  -> permission / policy
  -> adapter execution
  -> normalized observation
  -> tool_result artifact
  -> next model step
```

这里最重要的是：

- `tool_result` 必须成为显式 artifact
- 结果要以结构化对象回到 runtime
- planner / answer synthesizer / validator 消费的是显式结果，而不是隐式副作用

### 2.4 对当前项目的直接含义

这意味着：

- `planner` 不应继续直接编 shell
- `single_skill` 不应长期停留在 Python executor switch 上
- `action_agent` 生成的计划不应等于真实执行细节
- `skill_runtime.py` 应逐步从 executor switchboard 过渡为 descriptor / registry 层

也就是说，
下一阶段你项目里的执行模型应更接近：

```text
planner
  -> task spec
  -> execution agent
  -> tool intent
  -> capability registry
  -> adapter execution
  -> tool_result
  -> planner / answer
```

## 3. MCP Strategy

### 3.1 MCP 不是旁路系统

`MCP` 不应被当作“特殊扩展能力”，
而应被视作 capability source 的一种。

它和本地工具、service 的差别主要在于：

- transport 不同
- server 连接生命周期不同
- 权限 / auth / reconnect 逻辑不同

但对模型而言，
它不应该知道这些底层差异。

模型看到的仍然应该是：

- 一个统一名字
- 一个统一 schema
- 一个统一调用面

### 3.2 MCP 在 runtime 中的合理位置

建议把 MCP 放在：

```text
CapabilityRegistry
  -> Local tools
  -> Service-backed tools
  -> MCP-backed tools
```

也就是说：

- planner 不直接知道 MCP transport
- execution agent 也不直接感知 JSON-RPC 细节
- runtime adapter 负责 `Tool.call -> MCP client.callTool`

### 3.3 MCP 只暴露标准能力面

对每个 MCP capability，
runtime 应统一暴露：

- `name`
- `description`
- `input_schema`
- `read_only`
- `destructive`
- `open_world`
- `transport_kind`

transport 可以是：

- `stdio`
- `http`
- `sse`
- `ws`
- `sdk`
- 未来的 `in_process`

但这些信息主要供 runtime 和 policy 使用，
不是给 planner 当底层知识库。

### 3.4 当前项目的设计建议

对 `agentic-rag` 而言，
MCP 的引入方式应是：

- 先定义统一 capability 协议
- 再把 MCP server 视作 registry provider
- 最后才把 MCP tool 注入 execution agent 的能力池

不建议做成：

- planner 直接知道 `mcp__server__tool`
- 某个 graph node 直接硬编码某类 MCP transport
- skill 自己偷偷持有 MCP client 实现

## 4. Skill Strategy

### 4.1 skill 不等于 capability implementation

这是最重要的边界之一。

`Claude Code` 的做法很值得借：

- skill 更像 markdown + frontmatter 的 prompt command
- 它是 workflow / invocation knowledge package
- 它本身不直接等于底层能力实现

因此在本项目里，
skill 更适合被定义为：

- 任务入口
- 调用知识
- prompt packaging
- tool permission widening
- 执行模式切换

而不是：

- 本地 RAG 的 Python 封装
- 某个 service client 的实现体
- 某个 shell 命令的唯一承载体

### 4.2 skill 的推荐来源格式

建议 skill 使用：

- `markdown body`
- `frontmatter metadata`

其中 frontmatter 适合承载：

- `name`
- `description`
- `when_to_use`
- `allowed_tools`
- `model`
- `effort`
- `context`
- `agent`
- `hooks`
- `paths`
- `examples`

也就是说，
skill 应首先是一个“可读、可审查、可索引”的描述对象。

### 4.3 skill 的两种主要执行形态

参考 `Claude Code`，
skill 更适合有两种主执行形态。

#### inline skill

inline skill 的本质是：

- 把 skill 的 prompt package 注入当前对话 / 当前 execution context
- 为当前任务附加调用知识
- 可选地扩大允许工具范围
- 可选地修改模型 / effort 配置

它更像：

- “把一套任务指导装进当前上下文”

而不是：

- “立刻执行一段隐藏代码”

#### fork skill

fork skill 的本质是：

- 把 skill 交给专门的子 agent / execution context 执行

它更适合：

- 较长的工作流
- 需要独立预算与独立上下文的任务
- 需要把技能作为专门 worker 身份运行的场景

这意味着 skill 可以影响：

- 当前上下文
- 当前允许工具
- 当前执行 profile
- 是否需要 fork 子任务

但 skill 仍然不应直接成为 service implementation。

### 4.4 skill 可以注册 hooks，但不拥有真实能力

`Claude Code` 里 skill frontmatter 还能挂 hooks。

这个思路可以迁移，
但要保持边界：

- hooks 改变的是 session / task 行为
- tool / service / MCP 才是能力执行体

所以 skill 可以：

- 注册 session-scoped hooks
- 注入 prompt
- 提供 arguments / examples
- 指定 allowed tools

但 skill 不应该：

- 直接替代 tool registry
- 直接替代 service binding

### 4.5 skill 的动态发现策略

skill 不一定都要在启动时全量暴露。

适合引入的策略包括：

- 启动时加载静态 skill
- 根据当前项目路径动态激活 conditional skill
- 根据当前任务上下文做 skill discovery

这和你的项目现在的方向是兼容的：

- 保持技能库可检索
- 只在当前任务需要时注入相关 skill
- 避免一开始把所有 skill prompt 全塞给模型

### 4.6 Claude Code 的参考模式

结合 `Claude Code` 当前实现，
可以把它的 skill 机制总结为：

- skill 首先是 `Command` / prompt object
- skill 来源可以是本地 `SKILL.md`、bundled skill、plugin skill、MCP skill
- skill 统一进入命令注册表，而不是散落在各个执行节点里
- 模型通过 `SkillTool` 调用 skill，而不是直接执行 skill 内部逻辑

它的实际执行大致有两种：

- `inline`
  将 skill markdown 展开为当前对话中的 prompt package，并附带 allowed tools / model / effort 等上下文修改
- `fork`
  将 skill 交给子 agent，在隔离上下文和独立预算中执行

因此最值得迁移的不是某个文件格式，
而是这层抽象：

- `skill = how to work`
- `capability = what can be executed`
- `runtime = who actually executes and records the result`

这也是本项目后续设计必须守住的边界。

## 5. 对 Agentic-RAG 的迁移建议

### 5.1 Runtime 分层建议

建议后续收敛为：

```text
Planner Layer
  -> 只做 task decomposition / route

Execution Layer
  -> 产出 tool intent
  -> 选择 use-skill / call-capability / answer

Capability Layer
  -> local shell adapter
  -> service adapters
  -> MCP adapters

Skill Layer
  -> markdown/frontmatter descriptors
  -> prompt packages
  -> invocation guidance
  -> allowed-tools / hooks / execution-context metadata
```

### 5.2 现有代码的迁移方向

以下对象应逐步调整定位：

- `app/agent/skill_runtime.py`
  从 executor switch 逐步迁到 skill descriptor lookup
- `app/agent/nodes.py`
  中的 action / skill 执行节点应更多生成 `tool_intent` / `execution_plan`
- `app/agent/services/*`
  继续保留为真实能力实现层
- `app/runtime/shell_runtime.py`
  保留为 capability adapter，而不是 planner 可见实现细节

### 5.3 建议新增的结构化对象

建议后续在项目内逐步引入：

- `ToolIntent`
- `ToolExecutionResult`
- `CapabilityRegistryEntry`
- `SkillDescriptor`
- `SkillInvocationContext`

可以先从最小版开始：

```text
ToolIntent
  - name
  - input
  - reason

ToolExecutionResult
  - name
  - success
  - observation
  - evidence
  - degraded
  - metadata

SkillDescriptor
  - skill_id
  - name
  - description
  - when_to_use
  - prompt_files
  - allowed_tools
  - service_binding
  - execution_mode
```

### 5.4 近期最值得做的三件事

1. 先把 runtime capability 协议定义出来  
   让 shell / service / future MCP 先进入同一个执行协议

2. 把 skill 从“执行器”收缩成“描述对象”  
   `single_skill` 路径逐步改成 `resolve skill -> inject invocation guide -> execute capability`

3. 让执行结果显式化  
   统一把执行结果整理成 `tool_result / observation / evidence`，避免节点内部副作用散落

## 6. 最终判断

对本项目而言，
真正值得迁移的不是某一个具体文件实现，
而是这套边界观：

- 模型输出的是结构化调用意图，不是直接执行环境
- runtime 才拥有执行权、权限控制和 transport 细节
- MCP 只是 capability source 的一种
- skill 是 workflow / prompt / invocation knowledge layer
- service/API 继续作为真实能力实现

一句话总结：

> 下一阶段应把项目从“graph 调 executor”推进到“model intent -> capability protocol -> runtime execution -> explicit tool_result”的 agent runtime 形态，同时把 skill 正式收缩为描述层而不是实现层。
