import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "docs"
DATA_DIR = PROJECT_DIR / "data"
FEEDBACK_PATH = DATA_DIR / "player_feedback.md"
PROMPT_PATH = PROJECT_DIR / "prompts" / "system_prompt.md"
RETRIEVAL_K = 4
RETRIEVAL_CANDIDATES = 10


@dataclass
class AgentRun:
    answer: str
    docs: List[Document]
    steps: List[str]
    tool_name: str


class HashEmbeddings(Embeddings):
    """Small local fallback embedding for demos without an embedding API key."""

    def __init__(self, size: int = 384):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.size
        for token in self._tokens(text):
            vector[self._hash_token(token) % self.size] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

    @staticmethod
    def _hash_token(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        chinese_run = []
        buffer = []
        for char in text.lower():
            if "\u4e00" <= char <= "\u9fff":
                if buffer:
                    yield "".join(buffer)
                    buffer = []
                chinese_run.append(char)
            elif char.isalnum():
                if chinese_run:
                    yield from HashEmbeddings._chinese_tokens(chinese_run)
                    chinese_run = []
                buffer.append(char)
            elif buffer:
                yield "".join(buffer)
                buffer = []
                if chinese_run:
                    yield from HashEmbeddings._chinese_tokens(chinese_run)
                    chinese_run = []
            elif chinese_run:
                yield from HashEmbeddings._chinese_tokens(chinese_run)
                chinese_run = []
        if buffer:
            yield "".join(buffer)
        if chinese_run:
            yield from HashEmbeddings._chinese_tokens(chinese_run)

    @staticmethod
    def _chinese_tokens(chars: List[str]) -> Iterable[str]:
        for char in chars:
            yield char
        for index in range(len(chars) - 1):
            yield "".join(chars[index : index + 2])
        for index in range(len(chars) - 2):
            yield "".join(chars[index : index + 3])


def load_markdown_documents() -> List[Document]:
    documents: List[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=content,
                metadata={"source": path.name},
            )
        )
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "，", " "],
    )
    return splitter.split_documents(documents)


def get_embeddings() -> Embeddings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    if api_key:
        return OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )

    return HashEmbeddings()


@st.cache_resource(show_spinner=False)
def build_vector_store() -> Chroma:
    documents = load_markdown_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    try:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="game_rag_demo_openai",
        )
    except Exception as exc:
        st.warning(f"在线 Embedding 不可用，已切换到本地检索模式。原因：{exc.__class__.__name__}")
        return Chroma.from_documents(
            documents=chunks,
            embedding=HashEmbeddings(),
            collection_name="game_rag_demo_local",
        )


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def call_llm_text(prompt: str, temperature: float = 0.2) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini").strip()
    llm = ChatOpenAI(
        model=chat_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
    response = llm.invoke(prompt)
    return str(response.content).strip()


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```markdown"):
        stripped = stripped[len("```markdown") :].strip()
    elif stripped.startswith("```"):
        stripped = stripped[len("```") :].strip()
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")].strip()
    return stripped


def format_context(docs: List[Document]) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[片段 {index} | 来源：{source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieve_game_docs(question: str) -> List[Document]:
    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_CANDIDATES})
    docs = retriever.invoke(question)
    return rerank_documents(question, docs)[:RETRIEVAL_K]


def rerank_documents(question: str, docs: List[Document]) -> List[Document]:
    query_terms = {term for term in HashEmbeddings._tokens(question) if term.strip()}

    def score(doc: Document) -> int:
        source = doc.metadata.get("source", "")
        content = f"{source}\n{doc.page_content}".lower()
        return sum(len(term) for term in query_terms if term in content)

    return sorted(docs, key=score, reverse=True)


@tool
def search_game_docs_tool(query: str) -> str:
    """Search game project documents and return relevant snippets with sources."""
    docs = retrieve_game_docs(query)
    return format_context(docs)


@tool
def check_activity_rule_tool(query: str) -> str:
    """Check whether an activity rule document contains required planning fields."""
    activity_doc = find_activity_document(query)
    required_sections = [
        ("活动名称", "说明活动叫什么，避免运营、客服和玩家口径不一致。"),
        ("活动时间", "说明活动开始和结束时间，避免时区和重置时间争议。"),
        ("参与条件", "说明等级、任务进度、服务器等限制。"),
        ("活动入口", "说明玩家从哪里进入活动。"),
        ("活动玩法", "说明核心玩法和每日/每周限制。"),
        ("活动奖励", "说明奖励类型、兑换方式和发放方式。"),
        ("奖励发放规则", "说明实时发放、邮件发放、临时仓库等规则。"),
        ("异常处理", "说明客服遇到异常时需要核对哪些信息。"),
    ]
    content = activity_doc.page_content
    rows = []
    missing = []
    for section, reason in required_sections:
        exists = f"## {section}" in content or section in content
        status = "通过" if exists else "缺失"
        rows.append(f"| {section} | {status} | {reason} |")
        if not exists:
            missing.append(section)

    if missing:
        conclusion = f"当前活动规则还有 {len(missing)} 个关键字段需要补充：{', '.join(missing)}。"
    else:
        conclusion = "当前活动规则包含核心字段，可以作为活动上线前的基础检查样例。"

    return "\n".join(
        [
            f"## 活动规则完整性检查",
            "",
            f"检查文档：`{activity_doc.metadata.get('source', 'unknown')}`",
            "",
            "| 检查项 | 状态 | 检查意义 |",
            "| --- | --- | --- |",
            *rows,
            "",
            f"结论：{conclusion}",
            "",
            "建议：正式落地时可以把该工具接入策划文档评审流程，在活动上线前自动检查规则字段是否齐全。",
        ]
    )


@tool
def summarize_player_feedback_tool(query: str) -> str:
    """Summarize sample player feedback into issues, sentiment, and action suggestions."""
    feedback_items = load_player_feedback()
    stats = analyze_feedback_items(feedback_items)
    rule_based_summary = format_feedback_summary(feedback_items, stats)
    try:
        return enhance_feedback_summary_with_llm(query, feedback_items, rule_based_summary)
    except Exception as exc:
        return (
            f"{rule_based_summary}\n\n"
            f"补充说明：LLM 深度分析暂不可用，已返回规则统计摘要。错误类型：{exc.__class__.__name__}"
        )


def analyze_feedback_items(feedback_items: List[str]) -> dict:
    categories = {
        "活动与奖励": ["活动", "奖励", "积分", "兑换", "补偿", "排行榜"],
        "充值与订单": ["充值", "不到账", "订单", "支付", "补单"],
        "性能与稳定性": ["卡顿", "闪退", "加载", "延迟", "发热"],
        "新手体验": ["新手", "引导", "任务", "迷路", "教程"],
        "账号与安全": ["封禁", "申诉", "账号", "登录"],
    }
    negative_words = ["不到账", "卡顿", "闪退", "太难", "没收到", "封禁", "迷路", "延迟", "不清楚"]
    positive_words = ["喜欢", "清楚", "好看", "顺畅", "不错", "满意"]

    category_counts = {name: 0 for name in categories}
    negative_count = 0
    positive_count = 0
    for item in feedback_items:
        for name, keywords in categories.items():
            if any(keyword in item for keyword in keywords):
                category_counts[name] += 1
        if any(word in item for word in negative_words):
            negative_count += 1
        if any(word in item for word in positive_words):
            positive_count += 1

    sorted_categories = sorted(category_counts.items(), key=lambda pair: pair[1], reverse=True)
    top_rows = [f"| {name} | {count} |" for name, count in sorted_categories if count]
    sample_rows = [f"- {item}" for item in feedback_items[:5]]

    if negative_count > positive_count:
        sentiment = "负向反馈较多，需要优先关注活动奖励、充值订单和性能稳定性。"
    elif positive_count > negative_count:
        sentiment = "正向反馈较多，但仍需要持续观察高频问题。"
    else:
        sentiment = "正负反馈接近，建议结合真实工单量和玩家分层继续判断。"

    return {
        "category_counts": sorted_categories,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "sentiment": sentiment,
    }


def format_feedback_summary(feedback_items: List[str], stats: dict) -> str:
    top_rows = [f"| {name} | {count} |" for name, count in stats["category_counts"] if count]
    sample_rows = [f"- {item}" for item in feedback_items[:5]]

    return "\n".join(
        [
            "## 玩家反馈摘要",
            "",
            f"样本数量：{len(feedback_items)} 条",
            "",
            "| 问题类别 | 命中次数 |",
            "| --- | --- |",
            *top_rows,
            "",
            f"情绪判断：{stats['sentiment']}",
            "",
            "代表性反馈：",
            *sample_rows,
            "",
            "处理建议：",
            "- 优先排查充值不到账和活动奖励未到账问题，避免影响玩家信任。",
            "- 将卡顿、闪退、加载慢反馈同步给测试和客户端团队复现。",
            "- 对新手引导和活动入口说明进行文案优化，降低客服咨询量。",
        ]
    )


def enhance_feedback_summary_with_llm(
    query: str,
    feedback_items: List[str],
    rule_based_summary: str,
) -> str:
    feedback_text = "\n".join(f"- {item}" for item in feedback_items)
    prompt = f"""你是游戏运营分析助手。请基于玩家反馈样本和已有统计摘要，生成一份适合研发/运营复盘的分析结论。

用户问题：
{query}

玩家反馈样本：
{feedback_text}

已有统计摘要：
{rule_based_summary}

请用 Markdown 输出，必须包含：
1. 高频问题归类
2. 玩家情绪判断
3. 处理优先级
4. 建议分发部门
5. 运营回复建议

要求：
- 不要编造样本中不存在的事实。
- 结论要具体，适合项目组评审。
- 输出中文。"""
    llm_summary = call_llm_text(prompt, temperature=0.2)
    llm_summary = strip_markdown_fence(llm_summary)
    return "\n\n".join(
        [
            llm_summary,
            "---",
            "## 规则统计依据",
            rule_based_summary,
        ]
    )


def find_activity_document(query: str) -> Document:
    documents = load_markdown_documents()
    preferred_terms = ["活动规则", "夏日星潮", "活动"]
    for term in preferred_terms:
        for document in documents:
            source = document.metadata.get("source", "")
            if term in source or term in document.page_content:
                return document
    return documents[0]


def load_player_feedback() -> List[str]:
    content = FEEDBACK_PATH.read_text(encoding="utf-8")
    items = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:])
    return items


def answer_with_llm(question: str, docs: List[Document]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return fallback_answer(question, docs)

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini").strip()
    llm = ChatOpenAI(
        model=chat_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
    )
    prompt = f"""{load_system_prompt()}

用户问题：
{question}

检索到的文档片段：
{format_context(docs)}

请根据文档片段回答。"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        return (
            f"模型调用失败，已返回检索摘要。错误类型：{exc.__class__.__name__}\n\n"
            + fallback_answer(question, docs)
        )


def fallback_answer(question: str, docs: List[Document]) -> str:
    if not docs:
        return "当前资料中未找到明确依据。"

    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    snippets = []
    for doc in docs[:2]:
        text = doc.page_content.replace("\n", " ").strip()
        snippets.append(text[:260] + ("..." if len(text) > 260 else ""))

    if os.getenv("OPENAI_API_KEY", "").strip():
        model_status = "模型 API 暂不可用，因此先返回检索结果摘要。"
        next_step = "模型额度或服务恢复后，系统会调用大语言模型基于这些片段生成正式回答。"
    else:
        model_status = "当前未配置模型 API Key，因此先返回检索结果摘要。"
        next_step = "配置 OPENAI_API_KEY 后，系统会调用大语言模型基于这些片段生成正式回答。"

    return (
        f"{model_status}\n\n"
        f"问题：{question}\n\n"
        "可能相关资料：\n"
        + "\n\n".join(f"- {snippet}" for snippet in snippets)
        + "\n\n"
        f"来源：{', '.join(sources)}\n\n"
        f"{next_step}"
    )


def select_agent_tool_by_keywords(question: str) -> str:
    if any(keyword in question for keyword in ["反馈", "舆情", "评论", "玩家声音", "吐槽", "满意"]):
        return summarize_player_feedback_tool.name
    if any(keyword in question for keyword in ["检查", "完整", "缺失", "活动规则", "规则文档", "字段"]):
        return check_activity_rule_tool.name
    return search_game_docs_tool.name


def select_agent_tool(question: str) -> tuple[str, str]:
    fallback_tool = select_agent_tool_by_keywords(question)
    available_tools = [
        search_game_docs_tool.name,
        check_activity_rule_tool.name,
        summarize_player_feedback_tool.name,
    ]
    prompt = f"""你是游戏研发多场景 AI Agent 的工具路由器。请根据用户问题，从下面三个工具中选择最合适的一个。

可选工具：
1. {search_game_docs_tool.name}：用于查询游戏研发文档、活动规则、客服 FAQ、版本公告、道具规则等资料，并回答具体问题。
2. {check_activity_rule_tool.name}：用于检查活动规则文档是否完整，适合“检查字段、是否缺失、规则是否完整、上线前审核”类问题。
3. {summarize_player_feedback_tool.name}：用于汇总玩家反馈、评论、舆情、吐槽和运营问题，输出高频问题和处理建议。

用户问题：
{question}

请只输出工具名，不要输出解释。"""
    try:
        selected = call_llm_text(prompt, temperature=0).strip().replace("`", "")
        first_token = selected.split()[0] if selected.split() else ""
        if first_token in available_tools:
            return first_token, "LLM"
        for tool_name in available_tools:
            if tool_name in selected:
                return tool_name, "LLM"
    except Exception:
        pass
    return fallback_tool, "关键词兜底"


def run_multi_scene_agent(question: str) -> AgentRun:
    tool_name, route_method = select_agent_tool(question)
    steps = ["接收用户问题，并判断问题属于哪个游戏研发场景。"]
    if route_method == "LLM":
        steps.append("调用大语言模型进行 Tool 选择，模型根据用户意图选择最合适的工具。")
    else:
        steps.append("大语言模型路由不可用或返回异常，使用关键词规则兜底选择工具。")

    if tool_name == check_activity_rule_tool.name:
        steps.append(f"选择工具 `{check_activity_rule_tool.name}` 检查活动规则文档是否包含关键字段。")
        answer = check_activity_rule_tool.invoke({"query": question})
        steps.append("工具读取活动规则文档，并按活动名称、时间、参与条件、奖励、异常处理等字段逐项检查。")
        steps.append("输出检查表和补充建议，供策划评审或上线前自查使用。")
        return AgentRun(answer=answer, docs=[], steps=steps, tool_name=tool_name)

    if tool_name == summarize_player_feedback_tool.name:
        steps.append(f"选择工具 `{summarize_player_feedback_tool.name}` 汇总玩家反馈样本。")
        answer = summarize_player_feedback_tool.invoke({"query": question})
        steps.append("工具读取玩家反馈样本，先按活动奖励、充值订单、性能稳定性等类别统计高频问题。")
        steps.append("调用大语言模型生成深度运营分析，包括优先级、分发部门和回复建议；模型不可用时返回规则统计摘要。")
        return AgentRun(answer=answer, docs=[], steps=steps, tool_name=tool_name)

    steps.append(f"选择工具 `{search_game_docs_tool.name}` 检索活动规则、客服 FAQ、道具规则和版本公告等资料。")
    docs = retrieve_game_docs(question)
    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    steps.append(f"工具返回 {len(docs)} 个相关片段，来源包括：{', '.join(sources)}。")
    steps.append("将用户问题和检索片段组装成 Prompt，要求模型基于资料回答并保留依据。")
    answer = answer_with_llm(question, docs)
    if os.getenv("OPENAI_API_KEY", "").strip():
        steps.append("调用大语言模型生成回答；如果模型不可用，则自动返回检索摘要。")
    else:
        steps.append("当前未配置可用模型额度，使用本地检索摘要作为兜底回答。")
    steps.append("向用户展示答案和引用来源，便于人工复核。")
    return AgentRun(answer=answer, docs=docs, steps=steps, tool_name=tool_name)


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="游戏研发多场景 AI Agent",
        page_icon="",
        layout="wide",
    )

    st.title("游戏研发多场景 AI Agent")
    st.caption("LangChain + Tool + Chroma + Streamlit 的多场景 Agent/RAG Demo")

    with st.sidebar:
        st.header("Demo 资料")
        docs = load_markdown_documents()
        st.write(f"已加载 {len(docs)} 份 Markdown 文档")
        for doc in docs:
            st.markdown(f"- `{doc.metadata['source']}`")

        st.header("运行状态")
        if os.getenv("OPENAI_API_KEY", "").strip():
            st.success("已配置模型 API Key")
        else:
            st.warning("未配置 API Key，当前使用检索摘要兜底模式")

        st.header("Agent 工具")
        for tool_name in [
            search_game_docs_tool.name,
            check_activity_rule_tool.name,
            summarize_player_feedback_tool.name,
        ]:
            st.code(tool_name)

        if st.button("重建索引"):
            build_vector_store.clear()
            st.rerun()

    question = st.text_input(
        "请输入问题",
        placeholder="例如：检查夏日星潮活动规则是否完整",
    )

    examples = [
        "新手任务第一步是什么？",
        "夏日星潮活动的参与条件是什么？",
        "检查夏日星潮活动规则是否完整",
        "帮我总结一下玩家反馈里的主要问题",
        "充值不到账客服应该怎么处理？",
    ]

    st.write("示例问题：")
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example):
            question = example

    if question:
        with st.spinner("Agent 正在选择工具并处理任务..."):
            agent_run = run_multi_scene_agent(question)

        st.info(f"本次调用工具：`{agent_run.tool_name}`")

        st.subheader("回答")
        st.markdown(agent_run.answer)

        st.subheader("Agent 执行过程")
        for index, step in enumerate(agent_run.steps, start=1):
            st.markdown(f"{index}. {step}")

        if agent_run.docs:
            st.subheader("引用来源")
            for index, doc in enumerate(agent_run.docs, start=1):
                source = doc.metadata.get("source", "unknown")
                with st.expander(f"片段 {index}：{source}"):
                    st.write(doc.page_content)


if __name__ == "__main__":
    main()
