import streamlit as st
from groq import Groq
from PIL import Image
import fitz
import io
import base64


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Urdu / English OCR Editor",
    page_icon="📄",
    layout="wide"
)


# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

textarea {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
    line-height: 2.5 !important;
    background-color: #fffdf5 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# GROQ API
# ─────────────────────────────────────────
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY Streamlit Secrets mein nahi mili.")
    st.stop()

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
MODEL_NAME = "qwen/qwen3.8-27b"


# ─────────────────────────────────────────
# OCR PROMPT
# ─────────────────────────────────────────
OCR_PROMPT = """
You are a professional OCR system specialized in Urdu Nastaliq
and English scanned documents.

Extract ALL visible text from the image.

STRICT RULES:

1. Extract everything visible.
2. Do not summarize.
3. Do not translate.
4. Do not rewrite.
5. Do not correct spelling.
6. Preserve Urdu exactly as visible.
7. Preserve English exactly as visible.
8. Preserve numbers and dates.
9. Preserve punctuation.
10. Preserve paragraphs.
11. Preserve line breaks as much as possible.
12. Preserve headings.
13. Preserve the original reading order.
14. If a word is genuinely unreadable, write [unclear].
15. Do not add explanations.
16. Do not add comments.

Return ONLY the extracted text.
"""


# ─────────────────────────────────────────
# PDF → IMAGE
# ─────────────────────────────────────────
def pdf_to_image(uploaded_file):

    pdf_bytes = uploaded_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    if len(doc) == 0:
        doc.close()
        raise ValueError("PDF mein koi page nahi mila.")

    page = doc[0]

    pix = page.get_pixmap(
        dpi=250,
        alpha=False
    )

    image_bytes = pix.tobytes("png")

    doc.close()

    return Image.open(
        io.BytesIO(image_bytes)
    )


# ─────────────────────────────────────────
# IMAGE → BASE64
# ─────────────────────────────────────────
def image_to_base64(image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ─────────────────────────────────────────
# OCR FUNCTION
# ─────────────────────────────────────────
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
                        "text": "Extract all text from this document."
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,"
                                f"{image_base64}"
                            )
                        }
                    }

                ]
            }

        ],

        temperature=0,

        max_completion_tokens=8192,

        reasoning_effort="none"
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError(
            "Groq ne koi text return nahi kiya."
        )

    return result


# ─────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────
st.title("📄 Urdu / English OCR Editor")

st.caption(
    "Scan شدہ دستاویز اپلوڈ کریں — قابلِ ترمیم متن حاصل کریں"
)


# ─────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────
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

    # ─────────────────────────────────────
    # OPEN FILE
    # ─────────────────────────────────────
    try:

        if uploaded.type == "application/pdf":

            image = pdf_to_image(uploaded)

        else:

            image = Image.open(uploaded)

            if image.mode != "RGB":
                image = image.convert("RGB")

    except Exception as e:

        st.error(
            f"File open نہیں ہو سکی: {str(e)}"
        )

        st.stop()


    # ─────────────────────────────────────
    # TWO COLUMNS
    # ─────────────────────────────────────
    col1, col2 = st.columns(2)


    # ─────────────────────────────────────
    # ORIGINAL DOCUMENT
    # ─────────────────────────────────────
    with col1:

        st.subheader("📷 اپلوڈ شدہ دستاویز")

        st.image(
            image,
            use_container_width=True
        )


    # ─────────────────────────────────────
    # OCR EDITOR
    # ─────────────────────────────────────
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


        # ─────────────────────────────────
        # EXTRACT BUTTON
        # ─────────────────────────────────
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


        # ─────────────────────────────────
        # TEXT EDITOR
        # ─────────────────────────────────
        if "extracted_text" in st.session_state:

            edited_text = st.text_area(
                "متن یہاں ترمیم کریں",
                value=st.session_state[
                    "extracted_text"
                ],
                height=500,
                key="editor"
            )


            # ─────────────────────────────
            # DOWNLOAD
            # ─────────────────────────────
            text_bytes = edited_text.encode(
                "utf-8"
            )

            st.download_button(
                label="💾 Text Download کریں",
                data=text_bytes,
                file_name="urdu_text.txt",
                mime="text/plain",
                use_container_width=True
            )
