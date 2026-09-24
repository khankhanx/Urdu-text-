import streamlit as st
from groq import Groq
from PIL import Image
import fitz
import io
import base64


# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Urdu OCR Editor",
    page_icon="📄",
    layout="wide"
)


# ─── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

textarea {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
    line-height: 2.8 !important;
    background-color: #fffdf5 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Groq API ──────────────────────────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY Streamlit Secrets mein nahi mili.")
    st.stop()


client = Groq(api_key=GROQ_API_KEY)


# ─── Vision Model ──────────────────────────────────────────
MODEL_NAME = "qwen/qwen3.8-27b"


# ─── OCR Prompt ────────────────────────────────────────────
OCR_PROMPT = """
You are an expert OCR system specializing in Urdu and English documents.

Extract ALL visible text from the supplied document image.

Rules:

1. Extract the text exactly as visible.
2. Do NOT translate Urdu into English.
3. Do NOT translate English into Urdu.
4. Preserve Urdu Unicode correctly.
5. Preserve English text exactly.
6. Preserve numbers, dates and punctuation.
7. Preserve line breaks where possible.
8. Preserve headings and paragraphs.
9. Preserve the document's original reading order.
10. Do not summarize.
11. Do not explain anything.
12. Do not add information that is not visible.
13. If a word is genuinely impossible to read, write [unclear].

Return ONLY the extracted text.
"""


# ─── PDF → Image ───────────────────────────────────────────
def pdf_to_image(uploaded_file):

    pdf_bytes = uploaded_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    if len(doc) == 0:
        raise ValueError("PDF mein koi page nahi mila.")

    page = doc[0]

    pix = page.get_pixmap(
        dpi=200,
        alpha=False
    )

    img_bytes = pix.tobytes("png")

    doc.close()

    return Image.open(
        io.BytesIO(img_bytes)
    )


# ─── Image → Base64 ────────────────────────────────────────
def image_to_base64(image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ─── OCR ──────────────────────────────────────────────────
def extract_text(image):

    image_base64 = image_to_base64(image)

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": OCR_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this document image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],

        temperature=0,
        max_tokens=8192
    )

    text = response.choices[0].message.content

    if not text:
        raise ValueError(
            "Model ne koi text return nahi kiya."
        )

    return text


# ─── UI ────────────────────────────────────────────────────
st.title("📄 Urdu / English OCR Editor")

st.caption(
    "Scan شدہ دستاویز اپلوڈ کریں — قابلِ ترمیم متن حاصل کریں"
)


uploaded = st.file_uploader(
    "Image یا PDF اپلوڈ کریں",
    type=[
        "png",
        "jpg",
        "jpeg",
        "pdf"
    ]
)


if uploaded:

    try:

        if uploaded.type == "application/pdf":
            image = pdf_to_image(uploaded)
        else:
            image = Image.open(uploaded)

    except Exception as e:

        st.error(
            f"File open نہیں ہو سکی: {str(e)}"
        )

        st.stop()


    col1, col2 = st.columns(2)


    # ─── Original Image ────────────────────────────────────
    with col1:

        st.subheader("📷 اپلوڈ شدہ دستاویز")

        st.image(
            image,
            use_container_width=True
        )


    # ─── OCR Editor ────────────────────────────────────────
    with col2:

        st.subheader("✏️ قابلِ ترمیم متن")

        font_size = st.slider(
            "Font Size",
            min_value=14,
            max_value=36,
            value=20,
            step=2
        )

        st.markdown(
            f"""
            <style>
            textarea {{
                font-size: {font_size}px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


        if st.button(
            "🔍 متن نکالیں (Extract Text)",
            use_container_width=True
        ):

            with st.spinner(
                "متن نکالا جا رہا ہے..."
            ):

                try:

                    extracted = extract_text(image)

                    st.session_state[
                        "extracted_text"
                    ] = extracted

                    st.success(
                        "متن کامیابی سے نکال لیا گیا ✅"
                    )

                except Exception as e:

                    st.error(
                        f"Groq API Error: {str(e)}"
                    )


        if "extracted_text" in st.session_state:

            edited_text = st.text_area(
                "متن یہاں ترمیم کریں",
                value=st.session_state[
                    "extracted_text"
                ],
                height=450,
                key="editor"
            )


            # ─── Download ────────────────────────────────
            b64 = base64.b64encode(
                edited_text.encode("utf-8")
            ).decode()


            st.markdown(
                f"""
                <a
                    href="data:text/plain;charset=utf-8;base64,{b64}"
                    download="urdu_text.txt"
                >
                    <button style="
                        background:#4CAF50;
                        color:white;
                        border:none;
                        padding:10px 20px;
                        border-radius:5px;
                        cursor:pointer;
                        font-size:16px;
                        width:100%;
                        margin-top:10px;
                    ">
                        💾 Text Download کریں
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )
