import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import fitz  # PyMuPDF
import io
import base64


# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Urdu OCR Editor",
    page_icon="📄",
    layout="wide"
)


# ─── CSS Styling ──────────────────────────────────────────
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


# ─── Gemini API Key ────────────────────────────────────────
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("GEMINI_API_KEY Streamlit Secrets mein nahi mili.")
    st.stop()


# ─── Gemini Client ─────────────────────────────────────────
client = genai.Client(api_key=API_KEY)


# ─── Model ─────────────────────────────────────────────────
MODEL_NAME = "gemini-3.6-flash"


# ─── OCR Prompt ─────────────────────────────────────────────
OCR_PROMPT = """
You are an expert OCR assistant specializing in Urdu and English text extraction.

You will receive an image of a scanned document that may contain:

- Urdu text written in Nastaliq style
- English text
- Mixed Urdu and English
- Numbers
- Headings
- Tables
- Official document formatting

Your task:

1. Extract ALL visible text from the image.
2. Preserve the original wording exactly.
3. Do NOT translate anything.
4. Keep Urdu text in proper Urdu Unicode.
5. Keep English text in English.
6. Preserve line breaks as much as possible.
7. Preserve headings and paragraphs.
8. Preserve numbers and dates accurately.
9. Do NOT add explanations.
10. Do NOT summarize.
11. Do NOT correct grammar or spelling.
12. If a word is genuinely unreadable, write [unclear].

Return ONLY the extracted text.
"""


# ─── PDF to Image ──────────────────────────────────────────
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


# ─── Image to Bytes ────────────────────────────────────────
def image_to_bytes(image: Image.Image) -> bytes:

    # RGB mein convert karein
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ─── Extract Text ──────────────────────────────────────────
def extract_text(image: Image.Image) -> str:

    img_bytes = image_to_bytes(image)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/png"
            ),
            types.Part.from_text(
                text=OCR_PROMPT
            ),
        ]
    )

    if not response.text:
        raise ValueError(
            "Gemini ne koi text return nahi kiya."
        )

    return response.text


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

    # ─── Convert File ───────────────────────────────────────
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


    # ─── Left Side ─────────────────────────────────────────
    with col1:

        st.subheader("📷 اپلوڈ شدہ دستاویز")

        st.image(
            image,
            use_container_width=True
        )


    # ─── Right Side ────────────────────────────────────────
    with col2:

        st.subheader("✏️ قابلِ ترمیم متن")


        # Font Size
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


        # Extract Button
        if st.button(
            "🔍 متن نکالیں (Extract Text)",
            use_container_width=True
        ):

            with st.spinner(
                "متن نکالا جا رہا ہے... براہ کرم انتظار کریں"
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

                    error_message = str(e)

                    # Authentication Error
                    if (
                        "401" in error_message
                        or "UNAUTHENTICATED" in error_message
                        or "ACCESS_TOKEN_TYPE_UNSUPPORTED"
                        in error_message
                    ):

                        st.error(
                            "Gemini API authentication error ہے."
                        )

                        st.warning(
                            """
                            API key authenticate نہیں ہو رہی۔

                            Streamlit Secrets میں GEMINI_API_KEY
                            کو دوبارہ check کریں اور نئی API key
                            کے ساتھ app کو redeploy کریں۔
                            """
                        )

                    else:

                        st.error(
                            f"Gemini error: {error_message}"
                        )


        # ─── Text Editor ───────────────────────────────────
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
