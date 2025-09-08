# ContextEngineer 特性
## Context Engineer 概念
Context Engineering 是⼀⻔设计、构建并优化动态⾃动化系统的学科，旨在为⼤型语⾔模型在正确的时间、以正确的格式，提供正确的信息和⼯具，从⽽可靠、可扩展地完成复杂任务。
prompt 告诉模型如何思考，⽽ Context 则赋予模型完成⼯作所需的知识和⼯具。
“Context”的范畴：
“Context”的定义已远超⽤户单次的即时提示，它涵盖了 LLM 在做出响应前所能看到的所有信息⽣态系统：
● 系统级指令和⻆⾊设定。
● 对话历史（短期记忆）。
● 持久化的⽤户偏好和事实（⻓期记忆）。
● 动态检索的外部数据（例如来⾃RAG）。
● 可⽤的⼯具（API、函数）及其定义。
● 期望的输出格式（例如，JSON Schema）。

## Context-Engineer 特性全景图
### 1️⃣ Context 基础管理

提供统一的上下文容器，解决“信息如何进入 LLM”的问题。

**特性**：
- Context 架构定义（SystemPrompt / UserPrompt / History / Tools / Memory / RAG）

- Token 预算管理（按比例分配 Token）

- Context 注入引擎（统一封装上下文拼装逻辑）

**价值**：

统一上下文抽象，降低开发复杂度

保证上下文输入的可控性和一致性

### 2️⃣ 动态上下文优化

解决“Lost in the Middle”与上下文溢出问题。

**特性**：

- 动态检索 (Select)：按需取数据、工具、历史

- 压缩与摘要 (Compress)：文档/历史自动裁剪

- 重排序与优先级：按任务相关性调度信息

**价值**：

降低无效 Token 消耗

提升关键信息利用率


## 技术方案

### Token 预算管理（Token Budgeting）技术方案 

#### 1) 目标与约束
按「目标→策略→算法→实现→观测」五层实现
**目标**：

在固定上下文窗口内，最大化“信号”密度（真正相关的信息），最小化“噪声”。


主动规避长上下文里的 Lost in the Middle（信息置于开头/结尾效果最佳，居中最差）。


在多轮/长时任务中，通过 Select / Compress / Isolate 控制注入量与位置。

**系统约束**：

模型上下文上限 L_ctx，推理留白（生成答案的 输出预算）必须保留。

工具/大对象结果需 隔离 在主上下文外，仅注入必要片段。

#### 2) 策略分层（静态配额 + 动态调度）

##### 2.1 语义分区（建议分桶）

将输入上下文划分为 8 个分桶（每桶有 min/max/权重）：

System & 安全约束

任务指令（User Task / 规划提示）

工具与 Schema（精简版说明 / 函数签名）

历史对话（可摘要）

长期记忆（事实/偏好摘要）

RAG 证据（检索+重排+压缩）

Few-shot 示例（如需）

Scratchpad（推理痕迹/中间变量）

对 历史/证据 桶统一应用 Select → Compress：Top-N 重排、过滤式压缩、内容提取式压缩与摘要。

##### 2.2 位置策略（抗 Lost-in-the-Middle）
将 关键信息 前置或后置（首因/近因），避免落在中段；对长证据做“首尾摘录 + 摘要”。

#### 3) 预算分配算法（可直接实现）
设：

L_ctx：模型上下文极限

R_out：输出预算（预估答案长度 + 安全余量）

Ω：系统开销（消息头/分隔/模板）

B = L_ctx - R_out - Ω：可用输入预算

每个桶 i 定义：min_i / max_i / w_i（最小、最大、权重），以及 ROI 评分函数 u_i(·)（信息对当前任务的边际效用）。

##### 3.1 初始配额（带上下界的按权重分水）
```
B' = B - Σ min_i
b_i = min( max_i,  min_i + (w_i / Σ w_j) * B' )
```

##### 3.2 动态调优（基于 ROI 的“水填充”）
计算各桶候选片段的 相关性/新颖性/去重得分，作为 u_i 的近似。

将剩余配额按 Δu_i/Δtokens 的边际收益排序，迭代加码直至耗尽。

对历史与证据桶，优先纳入：高相关、高时效、未出现的关键事实；低分项进入 摘要/提取 流程。

##### 3.3 超限回退顺序（Drop Order）
1. Few-shot → 2) 低分证据 → 3) 冗长历史 → 4) 工具描述（保持最小签名）
永不丢弃：System、安全约束、User 任务核心指令。

##### 5) 关键工程组件

- TokenizerService：统一分词与长度估计（支持多模型/多编码器）。
- BudgetManager：实现 §3 的分配/回退；对每桶执行配额控制。
- Compressor：过滤/提取/摘要多级管线。
- ContextAssembler：按照 首→中→尾 的布局规则拼装消息，落实位置策略。
- Policy Engine：按“任务类型/风险级别/成本目标”切换预算策略模板。

##### 配置模版参考
```
model:
  name: gpt-x
  l_ctx: 200000
  r_out:
    target: 1200         # 目标生成长度
    headroom: 300        # 安全余量

buckets:
  system:
    min: 300; max: 800; weight: 2.0; sticky: true
  task:
    min: 300; max: 1500; weight: 2.5; sticky: true
  tools:
    min: 120; max: 400;  weight: 0.8; compress: "signature_only"
  history:
    min: 0;   max: 3000; weight: 1.2; compress: "task_summary"
  memory:
    min: 0;   max: 800;  weight: 0.8; select: true
  rag:
    min: 0;   max: 5000; weight: 2.8; select: true; rerank: "listwise"
  fewshot:
    min: 0;   max: 1200; weight: 0.5; droppable: true
  scratchpad:
    min: 0;   max: 800;  weight: 0.6; placement: "tail"

policies:
  default:
    drop_order: [fewshot, rag_lowscore, history_lowscore, tools_verbose]
    place:
      head: [system, task, tools_min]
      middle: [rag_highscore, history_summary]
      tail: [scratchpad, citations]
  research_heavy:
    overrides:
      rag.weight: 3.5
      r_out.target: 2000
      history.compress: "aggressive_extract"

```