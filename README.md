# 游戏研发多场景 AI Agent Demo

这是一个基于 `Python + Streamlit + LangChain + Chroma` 的多场景 Agent/RAG Demo，用于验证 AI 在游戏研发知识检索、策划规则检查和运营反馈分析场景中的落地可行性。

## Demo 目标

游戏项目中常见的问题包括文档分散、活动规则检查依赖人工经验、玩家反馈整理耗时较长。该 Demo 将模拟游戏策划文档、活动规则、道具规则、客服 FAQ、版本公告和玩家反馈样本接入本地原型，用户可以通过自然语言提问，Agent 会根据问题选择合适工具并展示执行过程。

## 技术路线

```text
Markdown 文档
  -> 文档切分
  -> Embedding 向量化
  -> Chroma 向量库
  -> Agent 判断任务类型
  -> 调用文档检索 / 活动规则检查 / 玩家反馈摘要工具
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
streamlit run app.py --server.port=8700
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
5. 页面展示执行过程、工具输出和引用来源。

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
| `summarize_player_feedback_tool` | 运营反馈分析 | 汇总玩家反馈样本，输出高频问题、情绪判断和处理建议 |

Agent 的最小执行链路是：

```text
接收问题 -> 判断任务场景 -> 选择 Tool -> 执行工具 -> 整理输出 -> 展示执行过程
```

## 注意事项

- 不要提交 `.env`。
- 不要上传真实公司文档、真实玩家数据、订单信息或内部代码。
- 当前版本使用内存 Chroma 索引，避免 Windows 本地 SQLite 文件被占用。
