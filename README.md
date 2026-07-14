# 游戏研发文档问答助手 Demo

这是一个基于 `Python + Streamlit + LangChain + Chroma` 的最小 RAG Demo，用于验证 AI 在游戏研发文档问答场景中的落地可行性。

## Demo 目标

游戏项目中常见的问题是文档分散、查询成本高、新人理解项目慢。该 Demo 将模拟游戏策划文档、活动规则、道具规则、客服 FAQ 和版本公告接入本地知识库，用户可以通过自然语言提问，系统会检索相关文档片段并生成回答，同时展示引用来源。

## 技术路线

```text
Markdown 文档
  -> 文档切分
  -> Embedding 向量化
  -> Chroma 向量库
  -> 相似片段检索
  -> LLM 基于片段回答
  -> 展示答案和引用来源
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
   └─ Demo说明.md
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
4. 系统生成回答。
5. 展示引用来源，说明回答不是凭空编造。

## 注意事项

- 不要提交 `.env`。
- 不要上传真实公司文档、真实玩家数据、订单信息或内部代码。
- `chroma_db/` 是本地向量库缓存，不需要提交到 GitHub。
