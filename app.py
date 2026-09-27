import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def demo_result(text):
    return f"""## 忠实型译文

[演示模式] 请配置 OPENAI_API_KEY 后生成真实译文。

## 自然型译文

[演示模式]

## 文学型译文

[演示模式]

## 文学特征分析

原文长度：{len(text)} 个字符
"""


def generate_translation(text):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return demo_result(text)

    client_options = {"api_key": api_key}

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_options["base_url"] = base_url

    client = OpenAI(**client_options)

    prompt = f"""
你是一名中英文学翻译辅助专家。

请处理下面的中文文学文本，并翻译成英文输出：

## 忠实型译文
尽量保留原文信息和句子关系。

## 自然型译文
符合英语母语读者的表达习惯。

## 文学型译文
尽量保留原文的意象、节奏、情感和文学风格。

## 文学特征分析
识别其中的：
- 比喻
- 意象
- 成语或文化负载词
- 人物语气
- 情感色彩

## 译法比较
说明三种译法的主要差异。

## 风险提示
指出可能存在的漏译、误译、增译或风格变化。

要求：
- 不要添加原文没有的事实
- 保留叙述视角和人物关系
- 对文化负载词说明处理方式

中文原文：
{text}
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "你是严谨的中英文学翻译辅助专家。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content


st.set_page_config(
    page_title="中英文学翻译辅助工具",
    layout="wide",
)

st.title("中英文学翻译辅助工具")
st.caption("生成多种译法，并分析文学风格与文化负载词")

source_text = st.text_area(
    "请输入中文文学片段",
    height=220,
    placeholder="例如：月色像一层薄霜，落在寂静的庭院里。",
)

if st.button("生成三种译法", type="primary"):
    if not source_text.strip():
        st.warning("请先输入中文原文。")
    else:
        with st.spinner("正在生成译文和分析结果……"):
            try:
                result = generate_translation(source_text)
                st.markdown(result)

                st.download_button(
                    "下载分析结果",
                    data=result,
                    file_name="translation_result.md",
                    mime="text/markdown",
                )
            except Exception as error:
                st.error(f"调用模型失败：{error}")