import json
import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image, ImageDraw

from rag_chain import build_chain

load_dotenv()
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "")

st.set_page_config(
    page_title="Document Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PaddleOCR Document Chatbot")
st.caption("Ask questions about your scanned document pages.")

# Check that ChromaDB exists before loading the chain
if not os.path.exists("chroma_db"):
    st.error(
        "Vector database not found. "
        "Run `python ingest.py` first to embed your documents."
    )
    st.stop()

# Load the RAG chain once per session
if "chain" not in st.session_state:
    with st.spinner("Loading RAG pipeline..."):
        st.session_state.chain = build_chain()

# Two separate history stores:
#   messages      → display (role + content + sources + images)
#   chat_history  → LangChain message objects for context
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar controls
with st.sidebar:
    st.header("Options")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        "- Your document pages were OCR-processed and stored as Markdown files.\n"
        "- Figures were described by GPT-4o-vision and stored alongside the text.\n"
        "- Your question retrieves the most relevant passages, which GPT-4o uses to answer.\n"
        "- Source pages and extracted figures are shown below each answer."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_page_num(source: str):
    m = re.search(r"page_(\d+)", os.path.basename(source))
    return int(m.group(1)) if m else None


def render_image_with_bbox(image_path: str, layout_image_path: str, bbox: list):
    annotated = None
    if layout_image_path and os.path.exists(layout_image_path):
        layout_img = Image.open(layout_image_path).copy()
        draw = ImageDraw.Draw(layout_img)
        draw.rectangle(bbox, outline="red", width=5)
        annotated = layout_img
    return image_path, annotated


def show_image_result(img_info: dict):
    """Side-by-side: extracted figure (left) + layout page with red bbox (right)."""
    extracted_path, annotated_layout = render_image_with_bbox(
        img_info["image_path"],
        img_info["layout_image_path"],
        img_info["bbox"],
    )
    page_label = img_info["page_num"] if img_info["page_num"] != -1 else "unknown"

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(extracted_path):
            st.image(extracted_path, caption=f"Extracted figure ({img_info['source']})",
                     use_container_width=True)
        else:
            st.warning(f"Image file not found: {extracted_path}")
    with col2:
        if annotated_layout is not None:
            st.image(annotated_layout,
                     caption=f"Location in page {page_label} (red box)",
                     use_container_width=True)
        else:
            st.info("Layout image not available.")


def show_text_sources(text_sources: list):
    """Show source layout pages for text-based context in a collapsible panel."""
    if not text_sources:
        return
    labels = ", ".join(f"p.{s['page_num']}" for s in text_sources)
    with st.expander(f"Source pages: {labels}"):
        cols = st.columns(min(len(text_sources), 3))
        for i, src in enumerate(text_sources):
            with cols[i % 3]:
                if src["layout_path"] and os.path.exists(src["layout_path"]):
                    st.image(src["layout_path"], caption=f"Page {src['page_num']}",
                             use_container_width=True)
                else:
                    st.caption(f"Page {src['page_num']} — layout image not found")


def build_context_display(context_docs):
    """Split retrieved docs into image infos and deduplicated text sources."""
    image_infos = []
    seen_pages = set()
    text_sources = []

    for doc in context_docs:
        if doc.metadata.get("type") == "image_description":
            image_infos.append({
                "image_path": doc.metadata["image_path"],
                "layout_image_path": doc.metadata.get("layout_image_path", ""),
                "bbox": json.loads(doc.metadata["bbox"]),
                "page_num": doc.metadata.get("page_num", -1),
                "source": doc.metadata.get("source", ""),
            })
        else:
            source = doc.metadata.get("source", "")
            page_num = extract_page_num(source)
            if page_num is not None and page_num not in seen_pages:
                seen_pages.add(page_num)
                layout_path = ""
                if OUTPUT_DIR:
                    layout_path = os.path.join(
                        OUTPUT_DIR, "layout", f"page_{page_num}_layout_det_res.png"
                    )
                text_sources.append({"page_num": page_num, "layout_path": layout_path})

    text_sources.sort(key=lambda x: x["page_num"])
    return image_infos, text_sources


def render_context(image_infos, text_sources):
    for img_info in image_infos:
        show_image_result(img_info)
    show_text_sources(text_sources)


# ---------------------------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")
        if msg["role"] == "assistant":
            render_context(msg.get("images", []), msg.get("text_sources", []))


# ---------------------------------------------------------------------------
# Handle new user input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history,
            })

        answer = result["answer"]
        sources = sorted({
            doc.metadata.get("source", "unknown")
            for doc in result.get("context", [])
        })

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.write(f"- {src}")

        image_infos, text_sources = build_context_display(result.get("context", []))
        render_context(image_infos, text_sources)

    # Update both history stores
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer),
    ])
    msg_dict = {"role": "assistant", "content": answer, "sources": sources}
    if image_infos:
        msg_dict["images"] = image_infos
    if text_sources:
        msg_dict["text_sources"] = text_sources
    st.session_state.messages.append(msg_dict)
