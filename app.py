import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    height: 48px;
    font-size: 17px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📄 Smart Document Scanner</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">CamScanner / HP Scanner style document enhancement</div>',
    unsafe_allow_html=True
)


# =========================================================
# FUNCTIONS
# =========================================================

def pil_to_cv(image):

    image = image.convert("RGB")

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


def cv_to_pil(image):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(image)


def resize_for_processing(image, max_width=1800):

    height, width = image.shape[:2]

    if width <= max_width:
        return image

    ratio = max_width / width

    new_size = (
        int(width * ratio),
        int(height * ratio)
    )

    return cv2.resize(
        image,
        new_size,
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# AUTO DOCUMENT SCAN
# =========================================================

def scan_document(image):

    image = resize_for_processing(image)

    original = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Noise reduction
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Edge detection
    edges = cv2.Canny(
        blur,
        50,
        150
    )

    # Strengthen edges
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    document = None

    image_area = image.shape[0] * image.shape[1]

    for contour in contours[:30]:

        area = cv2.contourArea(contour)

        if area < image_area * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:

            document = approx.reshape(4, 2)

            break

    if document is None:

        return original, False

    # Order corners
    pts = document.astype(np.float32)

    s = pts.sum(axis=1)

    diff = np.diff(
        pts,
        axis=1
    )

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    ordered = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)

    # Calculate dimensions
    width_a = np.linalg.norm(
        bottom_right - bottom_left
    )

    width_b = np.linalg.norm(
        top_right - top_left
    )

    max_width = int(
        max(width_a, width_b)
    )

    height_a = np.linalg.norm(
        top_right - bottom_right
    )

    height_b = np.linalg.norm(
        top_left - bottom_left
    )

    max_height = int(
        max(height_a, height_b)
    )

    max_width = max(
        max_width,
        500
    )

    max_height = max(
        max_height,
        500
    )

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        ordered,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return warped, True


# =========================================================
# SCANNER FILTERS
# =========================================================

def original_filter(image):

    return image


def auto_color(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    result = cv2.merge([
        l,
        a,
        b
    ])

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )


def document_color(image):

    result = auto_color(image)

    # Slight sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    result = cv2.filter2D(
        result,
        -1,
        kernel
    )

    return result


def grayscale_filter(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )


def black_white_filter(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    result = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_GRAY2BGR
    )


def magic_filter(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove uneven lighting
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        25
    )

    normalized = cv2.divide(
        gray,
        background,
        scale=255
    )

    normalized = cv2.normalize(
        normalized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Sharpen
    sharpen = cv2.addWeighted(
        normalized,
        1.4,
        cv2.GaussianBlur(
            normalized,
            (0, 0),
            2
        ),
        -0.4,
        0
    )

    return cv2.cvtColor(
        sharpen,
        cv2.COLOR_GRAY2BGR
    )


# =========================================================
# IMAGE DOWNLOAD
# =========================================================

def image_bytes(image):

    image = cv_to_pil(image)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95
    )

    return buffer.getvalue()


# =========================================================
# UPLOAD
# =========================================================

uploaded = st.file_uploader(
    "📷 Document ki photo upload karein",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded:

    original_pil = Image.open(uploaded)

    original_cv = pil_to_cv(
        original_pil
    )

    st.divider()

    # =====================================================
    # AUTO SCAN
    # =====================================================

    with st.spinner(
        "Document detect aur scan ho raha hai..."
    ):

        scanned, detected = scan_document(
            original_cv
        )

    if detected:

        st.success(
            "Document automatically detect ho gaya."
        )

    else:

        st.warning(
            "Document edges automatically detect nahi hue. "
            "Original image ko process kiya gaya hai."
        )


    # =====================================================
    # FILTER
    # =====================================================

    st.subheader("🎨 Scanner Mode")

    filter_name = st.selectbox(
        "Document style select karein",

        [
            "Auto Color",
            "Magic Scan",
            "Document Color",
            "Grayscale",
            "Black & White",
            "Original"
        ]
    )


    if filter_name == "Auto Color":

        final_image = auto_color(
            scanned
        )

    elif filter_name == "Magic Scan":

        final_image = magic_filter(
            scanned
        )

    elif filter_name == "Document Color":

        final_image = document_color(
            scanned
        )

    elif filter_name == "Grayscale":

        final_image = grayscale_filter(
            scanned
        )

    elif filter_name == "Black & White":

        final_image = black_white_filter(
            scanned
        )

    else:

        final_image = scanned


    # =====================================================
    # PREVIEW
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Original")

        st.image(
            original_pil,
            use_container_width=True
        )


    with col2:

        st.subheader("✨ Scanned Result")

        st.image(
            cv_to_pil(final_image),
            use_container_width=True
        )


    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.divider()

    st.subheader("💾 Download")

    st.download_button(
        "⬇️ Download Scanned Document",
        data=image_bytes(final_image),
        file_name="scanned_document.jpg",
        mime="image/jpeg",
        use_container_width=True
    )
