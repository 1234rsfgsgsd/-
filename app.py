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
STYLE_OPTIONS = {
    "现代简洁": "使用清晰、克制、现代的英文，避免不必要的古雅词汇。",
    "古典雅致": "使用典雅、含蓄、有文学质感的英文，但不要改变原文信息。",
    "诗意抒情": "突出节奏、意象和情感色彩，保留适度的诗性表达。",
    "口语叙事": "使用自然、亲切、易读的英文，适合人物对话和叙事文本。",
    "忠实保守": "尽量贴近原文句意和结构，少做解释性增译和自由改写。",
}


def parse_glossary(text):
    """Parse one term per line in the form Chinese=English."""
    glossary = []
    for line in text.splitlines():
        if "=" not in line:
            continue
        source, target = [part.strip() for part in line.split("=", 1)]
        if source and target:
            glossary.append((source, target))
    return glossary


def detect_direction(text):
    chinese = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    if chinese >= 2 and chinese >= latin * 0.1:
        return "中文到英文"
    if latin >= 3 and latin > chinese:
        return "英文到中文"
    return "无法判断"


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
        "recommendation": extract_section(raw_result, "RECOMMENDATION"),
        "comparison": extract_section(raw_result, "COMPARISON"),
        "risks": extract_section(raw_result, "RISKS"),
        "cultural": extract_section(raw_result, "CULTURAL"),
        "quality": extract_section(raw_result, "QUALITY"),
        "alignment": extract_section(raw_result, "ALIGNMENT"),
    }


def generate_translation(text, glossary, style, direction):
    if not API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请检查本地 .env 或云端 Secrets。")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0,
        max_retries=2,
    )

    glossary_text = "\n".join(f"- {source} = {target}" for source, target in glossary)
    if not glossary_text:
        glossary_text = "（用户没有提供术语表，请自行识别并说明关键术语。）"

    prompt = f"""
你是一名严谨的中英文学翻译辅助专家。

请处理下面的文学文本。输入语言为：{direction}。必须严格使用以下标记输出，不要修改标记名称。
如果输入是中文，请翻译成英文；如果输入是英文，请翻译成中文。译文部分必须使用目标语言，不能原样重复输入。

[FAITHFUL]
直译型英文译文
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

[RECOMMENDATION]
用简短中文说明这段文本适合的翻译或使用场景，并给出选用建议。例如：适合文学出版、课堂赏析、普通读者阅读、旅游宣传或学术研究；说明更推荐直译型、自然型还是文学型译文，以及理由。不要虚构原文没有的背景。
[/RECOMMENDATION]

[COMPARISON]
比较三种译法的主要差异，并说明各自适用场景
[/COMPARISON]

[RISKS]
指出漏译、误译、增译、意象变化或风格变化风险
[/RISKS]

[CULTURAL]
列出文化负载词、成语、典故和重要文学意象。每行使用：原词 | 类型 | 文化含义 | 推荐英文处理 | 处理理由
[/CULTURAL]

[QUALITY]
检查三种译文中的漏译、误译、增译、人称或时态变化、术语不一致、意象丢失和风格偏移。每条问题使用：问题类型 | 具体位置或词语 | 问题说明 | 修改建议。如果没有明显问题，请写“未发现明显问题”。
[/QUALITY]

[ALIGNMENT]
将原文按句切分，并为每句提供对应的文学型英文译文和一句简短修改建议。每行使用：原文句子 || 英文句子 || 建议
[/ALIGNMENT]

要求：
- 不增加原文没有的事实
- 保留人物关系、叙述视角、情绪和文学意象
- 同一段文本中的重复词尽量保持译法一致
- 优先遵守用户提供的术语表
- 文本整体风格要求：{style}
- 只输出英文译文和中文分析，不要输出其他标题

用户术语表：
{glossary_text}

待翻译原文：
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


def build_download_text(source, translations, glossary):
    return f"""# 中英文学翻译对照稿

## 中文原文

{source}

## 识别的翻译方向

{translations['direction']}

## 直译型译文

{translations['faithful']}

## 自然型译文

{translations['natural']}

## 文学型译文

{translations['literary']}

## 文学特征分析

{translations['analysis']}

## 适用场景与选用建议

{translations['recommendation']}

## 译法比较

{translations['comparison']}

## 风险提示

{translations['risks']}

## 文化负载词与文学意象

{translations['cultural']}

## 译文质量检查

{translations['quality']}

## 逐句对照与修改建议

{translations['alignment']}

## 用户术语表

{glossary or '未设置术语表。'}
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
if "glossary_text" not in st.session_state:
    st.session_state.glossary_text = ""
if "style" not in st.session_state:
    st.session_state.style = "现代简洁"

st.title("中英文学翻译辅助工具")
st.caption("比较不同译法，保留译者的判断与修改空间")

with st.sidebar:
    st.subheader("当前配置")
    st.caption(f"模型：{MODEL}")
    st.caption(f"本次会话已请求：{st.session_state.request_count}/{MAX_REQUESTS_PER_SESSION}")
    st.info("测试版：每次翻译都会消耗 API 额度，请避免重复点击。")
    st.subheader("中英术语表")
    st.caption("每行一个，格式：中文词=English term")
    glossary_text = st.text_area(
        "术语表",
        value=st.session_state.glossary_text,
        height=150,
        placeholder="庭院=courtyard\n月色=moonlight\n黛玉=Daiyu",
        key="glossary_input",
    )
    st.session_state.glossary_text = glossary_text

style_name = st.selectbox(
    "文学风格",
    options=list(STYLE_OPTIONS),
    index=list(STYLE_OPTIONS).index(st.session_state.style),
    help="选择后，模型会将该风格要求用于三种译文和逐句对照。",
)
st.session_state.style = style_name
style_instruction = STYLE_OPTIONS[style_name]

source_text = st.text_area(
    "请输入中文或英文文学片段",
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
        st.warning("请先输入中文或英文原文。")
    elif len(source_text) > MAX_INPUT_LENGTH:
        st.error(f"单次输入不能超过 {MAX_INPUT_LENGTH} 个字符。")
    elif st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.error("本次会话已达到测试次数上限。")
    else:
        with st.spinner("正在生成译文和文学分析……"):
            try:
                direction = detect_direction(source_text)
                if direction == "无法判断":
                    raise ValueError("无法判断输入语言，请输入中文或英文文本。")
                raw_result = generate_translation(
                    source_text,
                    parse_glossary(glossary_text),
                    style_instruction,
                    direction,
                )
                st.session_state.translation = parse_result(raw_result)
                st.session_state.translation["direction"] = direction
                st.session_state.source = source_text
                st.session_state.request_count += 1
                st.session_state.translation_version += 1
                st.success(f"已识别方向：{direction}")
            except Exception as error:
                st.error(f"调用模型失败：{error}")

if st.session_state.translation:
    result = st.session_state.translation
    st.subheader("译文比较与人工修改")
    tab_faithful, tab_natural, tab_literary = st.tabs(["直译型", "自然型", "文学型"])

    with tab_faithful:
        result["faithful"] = st.text_area(
            "直译型译文（可编辑）",
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

    st.subheader("文学特征、适用场景与风险分析")
    analysis_col, comparison_col = st.columns(2)
    with analysis_col:
        st.markdown(result["analysis"])
        st.info(result["recommendation"])
    with comparison_col:
        st.markdown(result["comparison"])
    st.warning(result["risks"])
    st.subheader("文化负载词与文学意象")
    st.markdown(result["cultural"])

    st.subheader("译文质量检查")
    st.markdown(result["quality"])

    st.subheader("逐句对照与修改建议")
    for row in result["alignment"].splitlines():
        parts = [part.strip() for part in row.split("||")]
        if len(parts) == 3:
            original, translated, suggestion = parts
            with st.container(border=True):
                st.markdown(f"**原文**：{original}")
                st.markdown(f"**文学型译文**：{translated}")
                st.caption(f"修改建议：{suggestion}")
        elif row.strip():
            st.markdown(row)

    st.download_button(
        "下载中英对照稿",
        data=build_download_text(
            st.session_state.source,
            result,
            st.session_state.glossary_text,
        ),
        file_name="literary_translation_result.md",
        mime="text/markdown",
        use_container_width=True,
    )
