import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Magic Document Scanner",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #777;
    margin-bottom: 25px;
}

div.stButton > button {
    width: 100%;
    min-height: 46px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("📄 Magic Document Scanner")

st.markdown(
    '<div class="subtitle">'
    'CamScanner-style automatic crop, clean white paper, '
    'sharp text and printer-friendly scanning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE CONVERSION
# ============================================================

def pil_to_cv(image):

    image = image.convert("RGB")

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


def cv_to_pil(image):

    if len(image.shape) == 2:
        return Image.fromarray(image)

    return Image.fromarray(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )


# ============================================================
# RESIZE
# ============================================================

def resize_for_detection(image, max_width=1600):

    h, w = image.shape[:2]

    if w <= max_width:
        return image

    ratio = max_width / w

    return cv2.resize(
        image,
        (
            int(w * ratio),
            int(h * ratio)
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# ORDER CORNERS
# ============================================================

def order_points(points):

    points = np.array(
        points,
        dtype=np.float32
    )

    s = points.sum(axis=1)

    diff = np.diff(
        points,
        axis=1
    )

    top_left = points[np.argmin(s)]

    bottom_right = points[np.argmax(s)]

    top_right = points[np.argmin(diff)]

    bottom_left = points[np.argmax(diff)]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ],
        dtype=np.float32
    )


# ============================================================
# AUTO DOCUMENT DETECTION
# ============================================================

def detect_document(image):

    small = resize_for_detection(
        image
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        40,
        150
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = (
        small.shape[0] *
        small.shape[1]
    )

    for contour in contours[:50]:

        area = cv2.contourArea(
            contour
        )

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

            points = approx.reshape(
                4,
                2
            )

            # Convert coordinates back
            # to original image size.

            sx = image.shape[1] / small.shape[1]
            sy = image.shape[0] / small.shape[0]

            points = points.astype(
                np.float32
            )

            points[:, 0] *= sx
            points[:, 1] *= sy

            return order_points(
                points
            )

    return None


# ============================================================
# PERSPECTIVE TRANSFORM
# ============================================================

def perspective_crop(
    image,
    points,
    margin=8
):

    rect = order_points(
        points
    )

    tl, tr, br, bl = rect

    width1 = np.linalg.norm(
        br - bl
    )

    width2 = np.linalg.norm(
        tr - tl
    )

    max_width = int(
        max(width1, width2)
    )

    height1 = np.linalg.norm(
        tr - br
    )

    height2 = np.linalg.norm(
        tl - bl
    )

    max_height = int(
        max(height1, height2)
    )

    max_width = max(
        max_width,
        500
    )

    max_height = max(
        max_height,
        500
    )

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    result = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height
        ),
        flags=cv2.INTER_CUBIC
    )

    # Small white border removal
    if margin > 0:

        h, w = result.shape[:2]

        if (
            h > margin * 2 and
            w > margin * 2
        ):
            result = result[
                margin:h-margin,
                margin:w-margin
            ]

    return result


# ============================================================
# SHADOW / BACKGROUND REMOVAL
# ============================================================

def remove_shadows(gray):

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        25
    )

    background = np.maximum(
        background,
        1
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

    return normalized


# ============================================================
# MAGIC SCAN
# ============================================================

def magic_scan(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Shadow correction
    gray = remove_shadows(
        gray
    )

    # Local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    # Gentle denoise
    gray = cv2.bilateralFilter(
        gray,
        7,
        35,
        35
    )

    # Slight sharpening
    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.2
    )

    sharp = cv2.addWeighted(
        gray,
        1.35,
        blur,
        -0.35,
        0
    )

    # Adaptive white background
    bw = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )

    # Remove tiny black noise
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        kernel
    )

    return bw


# ============================================================
# CLEAN COLOR SCAN
# ============================================================

def magic_color(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(
        l
    )

    result = cv2.merge(
        [l, a, b]
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )

    # Gentle sharpening
    blur = cv2.GaussianBlur(
        result,
        (0, 0),
        1
    )

    result = cv2.addWeighted(
        result,
        1.25,
        blur,
        -0.25,
        0
    )

    return result


# ============================================================
# GRAYSCALE
# ============================================================

def grayscale_scan(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = remove_shadows(
        gray
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    return gray


# ============================================================
# PRINTER MODE
# ============================================================

def printer_scan(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = remove_shadows(
        gray
    )

    # Very clean paper
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        10
    )

    # Remove tiny dust
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        kernel
    )

    return bw


# ============================================================
# SAVE IMAGE
# ============================================================

def encode_jpg(image):

    if len(image.shape) == 2:

        pil = Image.fromarray(
            image
        )

    else:

        pil = cv_to_pil(
            image
        )

    buffer = io.BytesIO()

    pil.save(
        buffer,
        format="JPEG",
        quality=97,
        subsampling=0
    )

    return buffer.getvalue()


# ============================================================
# UPLOAD
# ============================================================

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

    original = Image.open(
        uploaded
    ).convert("RGB")

    original_cv = pil_to_cv(
        original
    )

    # ========================================================
    # AUTO DETECT
    # ========================================================

    with st.spinner(
        "Document detect ho raha hai..."
    ):

        detected_points = detect_document(
            original_cv
        )


    if detected_points is not None:

        st.success(
            "✓ Document automatically detect ho gaya"
        )

        scanned_base = perspective_crop(
            original_cv,
            detected_points
        )

    else:

        st.warning(
            "Automatic crop nahi mila. "
            "Full image use ki ja rahi hai."
        )

        scanned_base = original_cv.copy()


    # ========================================================
    # CROP ADJUSTMENT
    # ========================================================

    st.divider()

    st.subheader(
        "✂️ Crop / Perspective"
    )

    st.caption(
        "Auto crop ke baad agar page ka corner galat ho "
        "to neeche adjustment controls use karein."
    )


    # Manual percentage crop controls
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        top_crop = st.slider(
            "Top",
            0,
            20,
            0
        )

    with c2:
        bottom_crop = st.slider(
            "Bottom",
            0,
            20,
            0
        )

    with c3:
        left_crop = st.slider(
            "Left",
            0,
            20,
            0
        )

    with c4:
        right_crop = st.slider(
            "Right",
            0,
            20,
            0
        )


    h, w = scanned_base.shape[:2]

    y1 = int(
        h * top_crop / 100
    )

    y2 = int(
        h * (1 - bottom_crop / 100)
    )

    x1 = int(
        w * left_crop / 100
    )

    x2 = int(
        w * (1 - right_crop / 100)
    )

    if x2 > x1 and y2 > y1:

        cropped = scanned_base[
            y1:y2,
            x1:x2
        ]

    else:

        cropped = scanned_base


    # ========================================================
    # FILTER
    # ========================================================

    st.divider()

    st.subheader(
        "✨ Magic Scanner"
    )

    mode = st.radio(
        "Scan Mode",

        [
            "Magic White",
            "Magic Color",
            "Printer Clean",
            "Grayscale",
            "Original"
        ],

        horizontal=True
    )


    if mode == "Magic White":

        result = magic_scan(
            cropped
        )

    elif mode == "Magic Color":

        result = magic_color(
            cropped
        )

    elif mode == "Printer Clean":

        result = printer_scan(
            cropped
        )

    elif mode == "Grayscale":

        result = grayscale_scan(
            cropped
        )

    else:

        result = cropped


    # ========================================================
    # PREVIEW
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "📷 Original"
        )

        st.image(
            original,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "✨ Final Scan"
        )

        st.image(
            cv_to_pil(result),
            use_container_width=True
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.divider()

    st.subheader(
        "💾 Download"
    )

    jpg_data = encode_jpg(
        result
    )

    st.download_button(
        "⬇️ Download High Quality Scan",
        data=jpg_data,
        file_name="magic_scan.jpg",
        mime="image/jpeg",
        use_container_width=True
    )
