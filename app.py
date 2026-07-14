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
PROMPT_PATH = PROJECT_DIR / "prompts" / "system_prompt.md"
RETRIEVAL_K = 4
RETRIEVAL_CANDIDATES = 10


@dataclass
class AgentRun:
    answer: str
    docs: List[Document]
    steps: List[str]


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


def run_document_agent(question: str) -> AgentRun:
    steps = [
        "接收用户问题，并判断需要查询游戏研发知识库。",
        f"选择工具 `{search_game_docs_tool.name}` 检索活动规则、客服 FAQ、道具规则和版本公告等资料。",
    ]
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
    return AgentRun(answer=answer, docs=docs, steps=steps)


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="游戏研发文档问答助手",
        page_icon="",
        layout="wide",
    )

    st.title("游戏研发文档问答 Agent")
    st.caption("LangChain + Tool + Chroma + Streamlit 的最小 Agent/RAG Demo")

    with st.sidebar:
        st.header("Demo 文档")
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
        st.code(search_game_docs_tool.name)

        if st.button("重建索引"):
            build_vector_store.clear()
            st.rerun()

    question = st.text_input(
        "请输入问题",
        placeholder="例如：夏日星潮活动的参与条件是什么？",
    )

    examples = [
        "新手任务第一步是什么？",
        "夏日星潮活动的参与条件是什么？",
        "充值不到账客服应该怎么处理？",
        "背包满了奖励会丢失吗？",
        "本次版本新增了哪些内容？",
    ]

    st.write("示例问题：")
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example):
            question = example

    if question:
        with st.spinner("Agent 正在选择工具、检索文档并生成回答..."):
            agent_run = run_document_agent(question)

        st.subheader("回答")
        st.markdown(agent_run.answer)

        st.subheader("Agent 执行过程")
        for index, step in enumerate(agent_run.steps, start=1):
            st.markdown(f"{index}. {step}")

        st.subheader("引用来源")
        for index, doc in enumerate(agent_run.docs, start=1):
            source = doc.metadata.get("source", "unknown")
            with st.expander(f"片段 {index}：{source}"):
                st.write(doc.page_content)


if __name__ == "__main__":
    main()
