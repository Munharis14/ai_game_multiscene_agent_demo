# 游戏研发文档问答 Agent Demo

这是一个基于 `Python + Streamlit + LangChain + Chroma` 的最小 Agent/RAG Demo，用于验证 AI 在游戏研发文档问答场景中的落地可行性。

## Demo 目标

游戏项目中常见的问题是文档分散、查询成本高、新人理解项目慢。该 Demo 将模拟游戏策划文档、活动规则、道具规则、客服 FAQ 和版本公告接入本地知识库，用户可以通过自然语言提问，Agent 会选择文档检索工具，检索相关文档片段并生成回答，同时展示执行过程和引用来源。

## 技术路线

```text
Markdown 文档
  -> 文档切分
  -> Embedding 向量化
  -> Chroma 向量库
  -> Agent 调用文档检索工具
  -> 相似片段检索
  -> LLM 基于片段回答
  -> 展示答案、执行过程和引用来源
```

## 项目结构

```text
ai-game-rag-demo/
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
├─ prompts/
│  └─ system_prompt.md
└─ report/
   ├─ Demo说明.md
   └─ Demo测试记录.md
```

## 运行方式

### 1. 创建虚拟环境

```powershell
cd D:\Users\Desktop\ai-game-rag-demo
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
- 充值不到账客服应该怎么处理？
- 背包满了奖励会丢失吗？
- 本次版本新增了哪些内容？

## 演示重点

1. 展示 `docs/` 中的游戏研发文档。
2. 输入问题。
3. 系统检索相关文档片段。
4. Agent 展示执行过程，包括选择工具、检索片段、组装 Prompt 和生成回答。
5. 系统生成回答。
6. 展示引用来源，说明回答不是凭空编造。

## Agent 设计

当前 Demo 将知识库检索封装为一个 LangChain Tool：

```text
search_game_docs_tool
```

Agent 的最小执行链路是：

```text
接收问题 -> 判断需要查询知识库 -> 调用文档检索工具 -> 组装 Prompt -> 调用模型/兜底摘要 -> 返回答案和引用
```

这个设计可以继续扩展更多工具，例如：

- `search_customer_faq_tool`：客服 FAQ 检索。
- `check_config_table_tool`：配置表规则检查。
- `summarize_feedback_tool`：玩家反馈摘要。
- `analyze_test_log_tool`：测试日志分析。

## 注意事项

- 不要提交 `.env`。
- 不要上传真实公司文档、真实玩家数据、订单信息或内部代码。
- 当前版本使用内存 Chroma 索引，避免 Windows 本地 SQLite 文件被占用。
