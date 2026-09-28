import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).with_name(".env"))


def get_setting(name, default=None):
    """Read Streamlit Cloud Secrets first, then local environment variables."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, default)


API_KEY = get_setting("DEEPSEEK_API_KEY")
BASE_URL = get_setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = get_setting("DEEPSEEK_MODEL", "deepseek-chat")
MAX_INPUT_LENGTH = 2000
MAX_REQUESTS_PER_SESSION = 10


def extract_section(text, name):
    pattern = rf"\[{name}\]\s*(.*?)\s*\[/{name}\]"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else "未能识别该部分，请查看完整结果。"


def parse_result(raw_result):
    return {
        "faithful": extract_section(raw_result, "FAITHFUL"),
        "natural": extract_section(raw_result, "NATURAL"),
        "literary": extract_section(raw_result, "LITERARY"),
        "analysis": extract_section(raw_result, "ANALYSIS"),
        "comparison": extract_section(raw_result, "COMPARISON"),
        "risks": extract_section(raw_result, "RISKS"),
    }


def generate_translation(text):
    if not API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请检查本地 .env 或云端 Secrets。")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0,
        max_retries=2,
    )

    prompt = f"""
你是一名严谨的中英文学翻译辅助专家。

请处理下面的中文文学文本。必须严格使用以下六组标记输出，不要修改标记名称：

[FAITHFUL]
忠实型英文译文
[/FAITHFUL]

[NATURAL]
自然型英文译文
[/NATURAL]

[LITERARY]
文学型英文译文
[/LITERARY]

[ANALYSIS]
分析比喻、意象、成语、文化负载词、人物语气和情感色彩
[/ANALYSIS]

[COMPARISON]
比较三种译法的主要差异，并说明各自适用场景
[/COMPARISON]

[RISKS]
指出漏译、误译、增译、意象变化或风格变化风险
[/RISKS]

要求：
- 不增加原文没有的事实
- 保留人物关系、叙述视角、情绪和文学意象
- 同一段文本中的重复词尽量保持译法一致
- 只输出英文译文和中文分析，不要输出其他标题

中文原文：
{text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是严谨的中英文学翻译辅助专家。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=2500,
    )

    result = response.choices[0].message.content
    if not result:
        raise RuntimeError("模型返回了空结果。")
    return result


def build_download_text(source, translations):
    return f"""# 中英文学翻译对照稿

## 中文原文

{source}

## 忠实型译文

{translations['faithful']}

## 自然型译文

{translations['natural']}

## 文学型译文

{translations['literary']}

## 文学特征分析

{translations['analysis']}

## 译法比较

{translations['comparison']}

## 风险提示

{translations['risks']}
"""


st.set_page_config(page_title="中英文学翻译辅助工具", page_icon="📖", layout="wide")

if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "translation" not in st.session_state:
    st.session_state.translation = None
if "source" not in st.session_state:
    st.session_state.source = ""
if "translation_version" not in st.session_state:
    st.session_state.translation_version = 0

st.title("中英文学翻译辅助工具")
st.caption("比较不同译法，保留译者的判断与修改空间")

with st.sidebar:
    st.subheader("当前配置")
    st.caption(f"模型：{MODEL}")
    st.caption(f"本次会话已请求：{st.session_state.request_count}/{MAX_REQUESTS_PER_SESSION}")
    st.info("测试版：每次翻译都会消耗 API 额度，请避免重复点击。")

source_text = st.text_area(
    "请输入中文文学片段",
    value=st.session_state.source,
    height=220,
    max_chars=MAX_INPUT_LENGTH,
    placeholder="例如：月色像一层薄霜，落在寂静的庭院里。",
)
st.caption(f"{len(source_text)}/{MAX_INPUT_LENGTH} 字符")

button_col, clear_col = st.columns([1, 1])
with button_col:
    generate_clicked = st.button("生成三种译法", type="primary", use_container_width=True)
with clear_col:
    if st.button("清空当前任务", use_container_width=True):
        st.session_state.translation = None
        st.session_state.source = ""
        st.rerun()

if generate_clicked:
    if not source_text.strip():
        st.warning("请先输入中文原文。")
    elif len(source_text) > MAX_INPUT_LENGTH:
        st.error(f"单次输入不能超过 {MAX_INPUT_LENGTH} 个字符。")
    elif st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.error("本次会话已达到测试次数上限。")
    else:
        with st.spinner("正在生成译文和文学分析……"):
            try:
                raw_result = generate_translation(source_text)
                st.session_state.translation = parse_result(raw_result)
                st.session_state.source = source_text
                st.session_state.request_count += 1
                st.session_state.translation_version += 1
            except Exception as error:
                st.error(f"调用模型失败：{error}")

if st.session_state.translation:
    result = st.session_state.translation
    st.subheader("译文比较与人工修改")
    tab_faithful, tab_natural, tab_literary = st.tabs(["忠实型", "自然型", "文学型"])

    with tab_faithful:
        result["faithful"] = st.text_area(
            "忠实型译文（可编辑）",
            result["faithful"],
            height=260,
            key=f"faithful_edit_{st.session_state.translation_version}",
        )
    with tab_natural:
        result["natural"] = st.text_area(
            "自然型译文（可编辑）",
            result["natural"],
            height=260,
            key=f"natural_edit_{st.session_state.translation_version}",
        )
    with tab_literary:
        result["literary"] = st.text_area(
            "文学型译文（可编辑）",
            result["literary"],
            height=260,
            key=f"literary_edit_{st.session_state.translation_version}",
        )

    st.subheader("文学特征与风险分析")
    analysis_col, comparison_col = st.columns(2)
    with analysis_col:
        st.markdown(result["analysis"])
    with comparison_col:
        st.markdown(result["comparison"])
    st.warning(result["risks"])

    st.download_button(
        "下载中英对照稿",
        data=build_download_text(st.session_state.source, result),
        file_name="literary_translation_result.md",
        mime="text/markdown",
        use_container_width=True,
    )
