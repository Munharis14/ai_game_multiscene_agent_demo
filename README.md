# 游戏研发多场景 AI Agent Demo

这是一个基于 `Python + Streamlit + LangChain + Chroma` 的多场景 Agent/RAG Demo，用于验证 AI 在游戏研发知识检索、策划规则检查和运营反馈分析场景中的落地可行性。

## Demo 目标

游戏项目中常见的问题包括文档分散、活动规则检查依赖人工经验、玩家反馈整理耗时较长。该 Demo 将模拟游戏策划文档、活动规则、道具规则、客服 FAQ、版本公告和玩家反馈样本接入本地原型，用户可以通过自然语言提问，Agent 会根据问题选择合适工具并展示执行过程。

## 当前功能

当前 Demo 已经实现以下能力：

| 能力 | 说明 |
| --- | --- |
| LLM 自动选择 Tool | 使用配置的大模型判断用户意图，并在三个工具中选择最合适的一个；模型不可用时自动回退到关键词规则 |
| 研发文档问答 | 检索游戏 Markdown 文档，基于活动规则、客服 FAQ、版本公告等资料回答问题 |
| 活动规则完整性检查 | 检查活动规则是否包含活动名称、活动时间、参与条件、活动入口、奖励规则和异常处理等字段 |
| 玩家反馈深度分析 | 先对玩家反馈做规则统计，再调用大模型生成高频问题、情绪判断、处理优先级、分发部门和运营回复建议 |
| 短期 Memory 多轮追问 | 使用 Streamlit Session State 保留最近 5 轮问答，让 Agent 能理解“这个规则”“刚才那个问题”“继续分析”等追问 |
| 执行过程展示 | 页面展示 Agent 的工具选择、工具执行和结果生成过程，便于演示和复盘 |
| 引用来源展示 | 文档问答场景展示命中的文档片段和来源，降低模型幻觉风险 |

## 技术路线

```text
Markdown 文档
  -> 文档切分
  -> Embedding 向量化
  -> Chroma 向量库
  -> 读取当前会话短期 Memory
  -> LLM 判断任务类型
  -> 调用文档检索 / 活动规则检查 / 玩家反馈摘要工具
  -> 工具执行
  -> LLM 生成回答或深度分析
  -> 返回答案、执行过程和引用来源
```

## 项目结构

```text
ai_game_multiscene_agent_demo/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ docs/
│  ├─ 新手任务流程.md
│  ├─ 活动规则说明.md
│  ├─ 道具与奖励规则.md
│  ├─ 客服FAQ.md
│  └─ 版本更新公告.md
├─ data/
│  └─ player_feedback.md
├─ prompts/
│  └─ system_prompt.md
└─ report/
   ├─ Demo说明.md
   └─ Demo测试记录.md
```

## 运行方式

### 1. 创建虚拟环境

```powershell
cd D:\Users\Desktop\ai_game_multiscene_agent_demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 配置模型

复制环境变量文件：

```powershell
copy .env.example .env
```

然后编辑 `.env`：

```text
OPENAI_API_KEY=你的 API Key
OPENAI_BASE_URL=可选，OpenAI 兼容服务地址
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

如果使用阿里云百炼 / Qwen，可以这样配置：

```text
OPENAI_API_KEY=你的 DashScope API Key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v4
```

如果暂时没有 API Key，也可以直接运行。系统会使用本地轻量检索兜底模式，返回相关文档片段摘要。

### 4. 启动应用

```powershell
python -m streamlit run app.py --server.port=8700
```

浏览器会打开本地页面。如果没有自动打开，可以访问：

```text
http://127.0.0.1:8700
```

说明：部分 Windows 环境会保留 `8501` 附近端口，使用 `8700` 更稳。

## 示例问题

- 新手任务第一步是什么？
- 夏日星潮活动的参与条件是什么？
- 检查夏日星潮活动规则是否完整
- 帮我总结一下玩家反馈里的主要问题
- 充值不到账客服应该怎么处理？

## 演示重点

1. 展示 `docs/` 中的游戏研发文档。
2. 展示 `data/` 中的玩家反馈样本。
3. 输入不同类型的问题。
4. Agent 自动选择对应 Tool。
5. 连续追问，例如先问“夏日星潮活动的参与条件是什么？”，再问“这个规则有什么风险？”。
6. 页面展示执行过程、工具输出和引用来源。

## Agent 设计

当前 Demo 将三个游戏研发场景封装为 LangChain Tool：

```text
search_game_docs_tool
check_activity_rule_tool
summarize_player_feedback_tool
```

三个 Tool 的职责如下：

| Tool | 场景 | 作用 |
| --- | --- | --- |
| `search_game_docs_tool` | 研发知识检索、客服问答 | 检索游戏 Markdown 文档，并返回相关片段和来源 |
| `check_activity_rule_tool` | 策划规则检查 | 检查活动规则是否包含活动名称、时间、参与条件、入口、奖励、异常处理等字段 |
| `summarize_player_feedback_tool` | 运营反馈分析 | 汇总玩家反馈样本，并调用大模型输出高频问题、情绪判断、优先级、分发部门和回复建议 |

Agent 的最小执行链路是：

```text
接收问题
  -> 读取最近几轮会话上下文
  -> LLM 判断任务场景
  -> 选择 Tool
  -> 执行工具
  -> 根据场景生成回答或深度分析
  -> 展示执行过程
```

如果大模型不可用，系统会自动回退：

```text
Tool 选择：关键词规则兜底
文档回答：检索摘要兜底
反馈分析：规则统计摘要兜底
```

## Memory 设计

当前 Demo 使用 `st.session_state` 实现轻量短期 Memory。系统会保留当前浏览器会话最近 5 轮问答，并在 Tool 选择、文档检索和回答生成时作为上下文传入模型。

该 Memory 只服务于演示多轮上下文能力，不会写入数据库，也不会跨浏览器刷新或重启后长期保存。因此它适合支持“继续分析”“这个活动”“刚才那个问题”等短追问，但不属于用户画像、长期记忆或生产级数据沉淀。

## 注意事项

- 不要提交 `.env`。
- 不要上传真实公司文档、真实玩家数据、订单信息或内部代码。
- 当前版本使用内存 Chroma 索引，避免 Windows 本地 SQLite 文件被占用。
- 短期 Memory 仅保存在当前 Streamlit 会话中，演示前可以通过侧边栏“清空上下文”重置。
