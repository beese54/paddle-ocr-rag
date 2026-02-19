import os
import glob
import json
import base64
import re

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

MARKDOWN_DIR = "data/markdown"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "paddle_ocr_docs"

IMG_DIR = os.getenv("IMG_DIR", "")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "")


def parse_bbox_from_filename(filename: str) -> list:
    """Extract [x1, y1, x2, y2] from img_in_image_box_591_968_1161_1810.jpg."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    return [int(p) for p in parts[-4:]]


def find_page_for_image(bbox: list, output_dir: str):
    """Search page JSON files to find which page contains a block with this bbox."""
    json_dir = os.path.join(output_dir, "json")
    for json_file in sorted(glob.glob(os.path.join(json_dir, "page_*_res.json"))):
        m = re.search(r"page_(\d+)_res\.json", os.path.basename(json_file))
        if not m:
            continue
        page_num = int(m.group(1))
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for block in data.get("parsing_res_list", []):
            block_bbox = block.get("block_bbox")
            if block_bbox is not None and [int(v) for v in block_bbox] == bbox:
                return page_num
    return None


def describe_image(img_path: str, client: OpenAI) -> str:
    """Send an image to GPT-4o and return a detailed description."""
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(img_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are analyzing a figure extracted from a technical document.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail: what type of figure it is "
                            "(diagram, chart, photo, etc.), what it shows, any labels, "
                            "values, or components visible."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img_data}"},
                    },
                ],
            },
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


def ingest_images(client: OpenAI) -> list:
    """Describe all extracted images and return them as LangChain Documents."""
    img_docs = []
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

    for img_path in img_files:
        filename = os.path.basename(img_path)
        try:
            bbox = parse_bbox_from_filename(filename)
            page_num = find_page_for_image(bbox, OUTPUT_DIR)

            layout_path = ""
            if page_num is not None:
                layout_path = os.path.join(
                    OUTPUT_DIR, "layout", f"page_{page_num}_layout_det_res.png"
                )

            page_label = str(page_num) if page_num is not None else "unknown"
            print(f"  Describing {filename} (page {page_label})...")
            description = describe_image(img_path, client)

            doc = Document(
                page_content=f"Figure from page {page_label}: {filename}\n\n{description}",
                metadata={
                    "source": filename,
                    "type": "image_description",
                    "image_path": os.path.abspath(img_path),
                    "layout_image_path": layout_path,
                    "bbox": json.dumps(bbox),
                    "page_num": page_num if page_num is not None else -1,
                },
            )
            img_docs.append(doc)
        except Exception as e:
            print(f"  Warning: failed to process {filename}: {e}")

    return img_docs


def ingest():
    # --- Text ingestion ---
    md_files = sorted(glob.glob(os.path.join(MARKDOWN_DIR, "*.md")))

    if not md_files:
        print(f"No .md files found in '{MARKDOWN_DIR}/'.")
        print("Copy your markdown files there and re-run.")
        return

    print(f"Found {len(md_files)} markdown file(s). Loading...")

    docs = []
    for path in md_files:
        loader = TextLoader(path, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = os.path.basename(path)
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} text chunk(s).")

    # --- Image ingestion ---
    if IMG_DIR and OUTPUT_DIR:
        img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
        print(f"\nDescribing {len(img_files)} image(s) with GPT-4o-vision...")
        client = OpenAI()
        img_docs = ingest_images(client)
        chunks.extend(img_docs)
        print(f"Added {len(img_docs)} image description(s).")
    else:
        print("\nIMG_DIR / OUTPUT_DIR not set — skipping image ingestion.")

    # --- Store in ChromaDB ---
    print(f"\nEmbedding and storing {len(chunks)} chunk(s) in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Done. {len(chunks)} chunks stored in '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    ingest()
