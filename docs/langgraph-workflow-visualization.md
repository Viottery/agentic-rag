# Agent Runtime Architecture

本文档描述项目下一阶段的目标架构。

这份文档不再把系统理解为：

- planner 直接调用技能
- skill 自身提供服务
- graph 节点承担过多底层执行逻辑

新的目标架构是：

- `fast gate` 负责简单问题短链路
- `planner` 只做全局拆分与分配
- `execution agent` 负责执行原子任务
- `shell` 是 execution agent 与本地环境、外部能力交互的主要通道
- `skill registry` 提供可检索的调用描述、提示词和平台适配信息
- `service/API` 提供真实能力，例如本地 RAG、搜索与工具服务
- `grounding validator` 负责结果与答案的关联性和可解释性约束
- `shell policy engine` 负责风控

## 1. 总入口

```text
Client
  -> POST /chat
  -> FastAPI route
  -> build initial state
  -> fast gate
  -> direct answer or planner loop
  -> return final response
```

对应代码入口仍然是：

- `app/api/routes/chat.py`
- `app/agent/graph.py`

但 graph 的长期职责将从“固定节点编排”逐步转向“runtime orchestration”。

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

可选保留极轻量的 sanity check，但它不应演化成新的重链路。

### 2.2 Normal Path

复杂问题走统一的 planner loop：

```text
user input
  -> fast gate
  -> planner
  -> subtask decomposition
  -> execution agent
  -> shell interaction
  -> skill lookup
  -> service / local environment / external resources
  -> structured task result
  -> planner
  -> answer synthesizer
  -> grounding validator
  -> END or back to planner
```

这里有几个关键边界：

- planner 决定“现在做什么”
- execution agent 决定“如何完成当前原子任务”
- shell 是主要交互通道
- skill 提供调用知识，不提供服务
- service 提供真实能力
- validator 决定“结果是否真的支撑答案”

## 3. 角色边界

### 3.1 Planner

Planner 的职责应收敛为：

- 识别当前信息缺口
- 拆分单一职责子任务
- 决定将子任务交给哪个 execution agent
- 基于返回结果决定继续、补任务、还是进入回答

Planner 可以知道：

- skill 列表
- skill 的功能描述
- skill 的适用场景

但 Planner 不应直接：

- 调 skill
- 拼 shell 命令
- 决定底层调用参数细节
- 直接与 service API 交互

一句话说：

- Planner 只决定任务分配，不直接执行。

### 3.2 Execution Agent

Execution agent 是实际执行子任务的主体。

职责包括：

- 接收原子任务
- 根据任务需要检索 skill
- 通过 shell 与本地环境、服务或外部能力交互
- 返回结构化 task result

Execution agent 允许有局部策略，
但不应在子任务内部再演化成新的全局 planner。

### 3.3 Skill Registry

skill 不应等于服务实现。

skill 更适合作为：

- 调用方法描述
- 提示词包装
- 输入输出约定
- 平台适配信息
- 使用建议

skill 应可被检索、索引和注入 prompt，
但 skill 本身不提供能力。

可以把 skill 理解为：

- `manifest + prompt package + invocation guide`

### 3.4 Service / API

真正的能力应由独立服务提供。

例如：

- `local_rag_service`
- `web_search_service`
- `tool_execution_service`

这些服务可以通过 FastAPI 暴露接口，
负责真实执行：

- 路由
- 检索
- rerank
- 搜索
- 工具调用
- 文件与系统交互

### 3.5 Grounding Validator

validator 不再只是“引用覆盖率检查器”。

它更适合作为：

- `grounding validator`

职责包括：

- 检查执行结果是否满足子任务目标
- 检查最终答案是否被结果支撑
- 标记强支持 / 弱支持 / 无支持结论
- 给 planner 返回补任务建议
- 抑制 hallucination

## 4. Skill Registry 组织方式

skill 应组织成一个便于检索和索引的 registry，
而不是散落在节点实现里。

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

这里最重要的两个字段是：

- `service_binding`
  指向真实 service/API
- `platform_invocation`
  说明在 Linux / Windows 下如何通过 shell 调用

## 5. Shell 作为主要交互通道

execution agent 应保留较强的命令行能力。

shell 不应被视为例外能力，
而应被视为：

- execution agent 与外界交互的主要 substrate

execution agent 可以通过 shell：

- 访问项目工作区
- 调用本地 CLI
- 调用 skill 提供的推荐命令模板
- 访问本地 service API
- 调用受控外部服务

这意味着 shell 不应被 skill 替代。

更准确的关系是：

- shell 是基础执行通道
- skill 是 shell 使用时的组织化知识层

## 6. 跨平台策略

系统需要同时支持 Linux 和 Windows，
同时尽量保留 LLM 的命令行技术优势。

### 6.1 不强行统一 shell

建议保留：

- Linux: `bash`
- Windows: `powershell`

不要为了统一而抹掉各自的生态优势。

### 6.2 统一调用协议，而不是统一命令字符串

skill 不应只存一条 bash 命令。

更合理的做法是：

- skill 提供平台感知的调用模板
- execution agent 根据当前 OS 选择相应模板

例如：

- Linux 调用模板
- Windows 调用模板

但它们都应指向同一 skill 和同一 service binding。

### 6.3 尽量通过 CLI + 文件交换

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

这样既保留 shell 主通道，
又能显著减少跨平台命令构造的不稳定性。

## 7. Shell 风控

如果 shell 是主要交互通道，
风控必须是 runtime 级能力，而不是 prompt 级提醒。

建议至少包括以下层次。

### 7.1 命令分级

将命令按风险分层，例如：

- `L0` 只读本地
- `L1` 受控写入
- `L2` 网络读取
- `L3` 高风险写操作
- `L4` 危险系统操作

默认 execution agent 只拿到低到中风险权限。

### 7.2 作用域限制

限制：

- 文件系统访问范围
- 网络访问范围
- 可继承环境变量
- 可用可执行程序

### 7.3 审计

每条 shell 命令都应记录：

- task id
- agent id
- command
- cwd
- start/end time
- exit code
- stdout/stderr 摘要

### 7.4 破坏性命令拦截

需要 runtime 层直接阻止高风险命令模式，
而不是依赖 LLM 自觉避免。

### 7.5 资源限制

每条 shell 命令都应有：

- timeout
- max output
- max subprocesses
- resource ceilings

## 8. State 设计方向

下一阶段 state 应逐步围绕这些对象重构：

- `request`
- `fast_path_decision`
- `planner_state`
- `subtasks`
- `execution_results`
- `skill_results`
- `evidence`
- `answer`
- `validation`
- `trace`

重点变化是：

- 降低固定节点痕迹
- 强化 task / skill / execution / validation 四类结构化对象
- 让 planner 消费的是结果对象，而不是底层节点状态

## 9. 近期迁移路径

建议按以下顺序迁移：

1. 保留 fast gate
2. 明确 planner 与 execution agent 的职责分离
3. 将 skill 从“执行逻辑”重构为“registry entry”
4. 将本地 RAG 重构为独立 service/API
5. 设计统一 skill 调用 CLI
6. 增加 Linux / Windows 平台适配模板
7. 建立 shell policy engine
8. 最后再升级 grounding validator

## 10. 当前明确不优先推进的事项

- 父子索引 / 多层索引对象建设仍保留在 TODO
- 不优先继续扩张旧 graph 上的固定节点链
- 不优先把复杂性继续堆到 planner prompt 上

当前优先级更高的是：

- execution agent 边界
- shell runtime
- skill registry
- service 化
- shell 风控
