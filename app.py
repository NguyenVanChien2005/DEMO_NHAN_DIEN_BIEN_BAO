"""
===================================================================================
ỨNG DỤNG WEB DEMO NHẬN DIỆN BIỂN BÁO GIAO THÔNG BẰNG YOLOV11 & STREAMLIT
===================================================================================
Tính năng nổi bật:
- Mô hình YOLOv11 nhận diện chuẩn xác 8 lớp biển báo giao thông Việt Nam.
- ĐỒNG BỘ & GHÉP ĐÔI VIDEO (Side-by-Side Single Video): Ghép video gốc và video
  nhận diện thành 1 video duy nhất chia đôi màn hình, đồng bộ 100% từng mili-giây.
- Hỗ trợ Slow-motion (0.20x - 1.00x) làm chậm video đầu ra giúp hội đồng dễ quan sát.
- ĐỐI CHIẾU TRỰC QUAN (Visual Verification Inspector):
    + Tự động crop ảnh cận cảnh từng biển báo theo tọa độ Bounding Box [x1, y1, x2, y2].
    + Hiển thị cột Thumbnail trực tiếp trong bảng dữ liệu (st.column_config.ImageColumn).
    + Bộ công cụ phóng to đối chiếu (Inspector) giúp hội đồng kiểm chứng độ chính xác
      giữa dự đoán AI và biển báo thực tế ngoài hiện trường.
    + Tối ưu bộ nhớ đệm (nén ảnh ~2KB/mẫu) chống nặng trình duyệt khi video dài.
- Khắc phục lỗi video không chạy trên web bằng chuẩn mã hóa H.264 (yuv420p).
- 4 Tabs chuyên biệt:
    Tab 1: Nhận diện Ảnh & Video (Ghép đôi Side-by-Side, Slow-motion, Bảng đối chiếu).
    Tab 2: Thống kê & Biểu đồ đánh giá trực quan (KPIs, Bar, Pie, Histogram, CSV).
    Tab 3: Nhận diện trực tiếp qua Webcam (Snapshot & Live Stream FPS thời gian thực).
    Tab 4: Tra cứu chi tiết quy chuẩn 8 lớp biển báo giao thông đường bộ Việt Nam.
===================================================================================
"""

import os
import time
import base64
import tempfile
import subprocess
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

# Hỗ trợ mã hóa video H.264 chuẩn Web
try:
    import imageio
    import imageio_ffmpeg
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# ==============================================================================
# 1. CẤU HÌNH TRANG VÀ THIẾT KẾ GIAO DIỆN HIỆN ĐẠI (UI/UX)
# ==============================================================================
st.set_page_config(
    page_title="Nhận Diện Biển Báo Giao Thông | YOLOv11",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS giao diện chuyên nghiệp
st.markdown("""
<style>
    /* Gradient Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0284c7 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #bae6fd;
        font-weight: 400;
        margin-bottom: 0px;
    }
    
    /* Metric Cards */
    .stMetric {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }
    
    /* Inspector Container */
    .inspector-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    
    /* Card danh mục tra cứu biển báo */
    .sign-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .sign-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 6px 16px rgba(56, 189, 248, 0.15);
    }
    .sign-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .sign-type {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .type-cam { background: #fee2e2; color: #b91c1c; }
    .type-nguyhiem { background: #fef3c7; color: #b45309; }
    .type-chidan { background: #e0f2fe; color: #0369a1; }
    .type-khac { background: #f1f5f9; color: #475569; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. TỪ ĐIỂN VÀ THÔNG TIN CHI TIẾT 8 LỚP BIỂN BÁO
# ==============================================================================
SIGN_CATALOG: Dict[str, Dict[str, Any]] = {
    "cam_di_nguoc_chieu": {
        "id": 1,
        "name_vn": "Cấm đi ngược chiều",
        "icon": "⛔",
        "code": "P.102",
        "group": "Biển báo cấm",
        "type_class": "type-cam",
        "description": "Báo đường cấm tất cả các loại xe (cơ giới và thô sơ) đi vào theo chiều đặt biển, trừ các xe được ưu tiên theo quy định."
    },
    "cam_dung_va_do": {
        "id": 2,
        "name_vn": "Cấm dừng và đỗ xe",
        "icon": "🚫",
        "code": "P.130",
        "group": "Biển báo cấm",
        "type_class": "type-cam",
        "description": "Báo nơi cấm dừng xe và đỗ xe. Áp dụng cho mọi phương tiện giao thông trên đoạn đường từ vị trí cắm biển."
    },
    "cam_quay_dau": {
        "id": 3,
        "name_vn": "Cấm quay đầu xe",
        "icon": "↩️",
        "code": "P.124a",
        "group": "Biển báo cấm",
        "type_class": "type-cam",
        "description": "Báo cấm các loại xe quay đầu xe theo kiểu chữ U, trừ các xe được ưu tiên khẩn cấp."
    },
    "cam_re": {
        "id": 4,
        "name_vn": "Cấm rẽ",
        "icon": "🔀",
        "code": "P.123",
        "group": "Biển báo cấm",
        "type_class": "type-cam",
        "description": "Báo cấm các phương tiện rẽ trái hoặc rẽ phải tại nơi đường giao nhau."
    },
    "giao_nhau_uu_tien": {
        "id": 5,
        "name_vn": "Giao nhau với đường không ưu tiên",
        "icon": "⚠️",
        "code": "W.207",
        "group": "Biển cảnh báo",
        "type_class": "type-nguyhiem",
        "description": "Báo trước sắp đến nơi giao nhau với đường không ưu tiên. Xe trên tuyến đường có biển này được quyền ưu tiên đi trước qua nút giao."
    },
    "gioi_han_toc_do": {
        "id": 6,
        "name_vn": "Giới hạn tốc độ tối đa",
        "icon": "🛑",
        "code": "P.127",
        "group": "Biển báo cấm",
        "type_class": "type-cam",
        "description": "Báo tốc độ tối đa cho phép các phương tiện cơ giới chạy trên đoạn đường tính từ vị trí đặt biển."
    },
    "nguoi_di_bo": {
        "id": 7,
        "name_vn": "Đường người đi bộ cắt ngang",
        "icon": "🚶",
        "code": "W.224",
        "group": "Biển cảnh báo",
        "type_class": "type-nguyhiem",
        "description": "Báo trước sắp tới phần đường dành cho người đi bộ sang đường. Người điều khiển phương tiện phải giảm tốc độ và nhường đường."
    },
    "Other": {
        "id": 0,
        "name_vn": "Biển báo khác (Other)",
        "icon": "ℹ️",
        "code": "Phụ/Khác",
        "group": "Khác",
        "type_class": "type-khac",
        "description": "Bao gồm các loại biển báo phụ, biển chỉ dẫn hoặc biển báo đặc thù khác ngoài các lớp mục tiêu chính."
    }
}

def format_class_name(raw_name: str) -> str:
    """Trả về tên tiếng Việt có dấu chuẩn xác kèm biểu tượng mô tả."""
    if raw_name in SIGN_CATALOG:
        return f"{SIGN_CATALOG[raw_name]['icon']} {SIGN_CATALOG[raw_name]['name_vn']}"
    return raw_name


# ==============================================================================
# 3. NẠP MÔ HÌNH YOLOV11 NGẦM ĐỊNH (CỐ ĐỊNH FILE WEIGHTS TRONG NỀN)
# ==============================================================================
DEFAULT_WEIGHTS = "traffic_sign_8class_best.pt"

# Tự động kiểm tra file weights mặc định
if not os.path.exists(DEFAULT_WEIGHTS):
    if os.path.exists("traffic_sign_8class_best.pt.pt"):
        DEFAULT_WEIGHTS = "traffic_sign_8class_best.pt.pt"

@st.cache_resource(show_spinner=False)
def get_cached_yolo_model(weights_file: str):
    """Nạp và lưu trữ mô hình YOLOv11 vào bộ nhớ cache của Streamlit."""
    try:
        model = YOLO(weights_file)
        return model, None
    except Exception as err:
        return None, str(err)

# Nạp mô hình
model, load_error = get_cached_yolo_model(DEFAULT_WEIGHTS)


# ==============================================================================
# 4. HÀM CẮT ẢNH & TẠO THUMBNAIL TỐI ƯU BỘ NHỚ CHO ĐỐI CHIẾU TRỰC QUAN
# ==============================================================================
def crop_and_encode_thumbnail(frame_bgr: np.ndarray, xyxy: List[float], max_size=(100, 100), pad=8) -> Tuple[str, np.ndarray]:
    """
    Cắt vùng ảnh Bounding Box với padding nhẹ, tối ưu bộ nhớ:
    - Trả về data URI Base64 nén JPEG (~2-3 KB) để nhúng trực tiếp vào st.dataframe.
    - Trả về mảng numpy RGB để hiển thị cận cảnh trong Inspector.
    """
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    
    # Thêm padding nhẹ để nhìn rõ toàn thể biển báo
    x1_pad = max(0, x1 - pad)
    y1_pad = max(0, y1 - pad)
    x2_pad = min(w_img, x2 + pad)
    y2_pad = min(h_img, y2 + pad)
    
    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        blank = np.zeros((60, 60, 3), dtype=np.uint8)
        return "", blank
        
    crop_bgr = frame_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    
    # Tạo thumbnail nhẹ cho bảng st.dataframe
    h_c, w_c = crop_bgr.shape[:2]
    scale = min(max_size[0] / max(w_c, 1), max_size[1] / max(h_c, 1), 1.0)
    new_w = max(int(w_c * scale), 1)
    new_h = max(int(h_c * scale), 1)
    resized_crop = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Nén JPEG chất lượng 80% (chỉ chiếm ~2 KB)
    ok, buf = cv2.imencode('.jpg', resized_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
        uri = f"data:image/jpeg;base64,{b64_str}"
    else:
        uri = ""
        
    return uri, crop_rgb


def create_summary_table(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Tổng hợp số lượng và độ tự tin theo từng lớp biển báo."""
    if not detections:
        return pd.DataFrame()
    
    df_raw = pd.DataFrame(detections)
    rows = []
    
    for cls_name, grp in df_raw.groupby("Tên Biển Báo"):
        rows.append({
            "Tên Biển Báo": cls_name,
            "Mã Lớp (Class)": grp["Mã Lớp"].iloc[0],
            "Số Lượng": len(grp),
            "Độ Tự Tin TB": f"{grp['Độ Tự Tin (Conf)'].mean() * 100:.1f}%",
            "Độ Tự Tin Cao Nhất": f"{grp['Độ Tự Tin (Conf)'].max() * 100:.1f}%"
        })
        
    df_out = pd.DataFrame(rows)
    return df_out.sort_values(by="Số Lượng", ascending=False).reset_index(drop=True)


# ==============================================================================
# 5. KHỞI TẠO SESSION STATE
# ==============================================================================
if "detection_records" not in st.session_state:
    st.session_state["detection_records"] = []

if "last_source_type" not in st.session_state:
    st.session_state["last_source_type"] = "Chưa có dữ liệu"


# ==============================================================================
# 6. SIDEBAR TINH GỌN (CHỈ GIỮ THAM SỐ CẦN THIẾT, KHÔNG RỐI MẮT)
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830310.png", width=60)
    st.title("Điều Khiển AI")
    
    # Trạng thái mô hình nhỏ gọn
    if load_error:
        st.error(f"Lỗi nạp mô hình: {load_error}")
        st.stop()
    else:
        st.caption("🟢 **Mô hình**: YOLOv11 Engine Ready")
    
    st.markdown("---")
    
    # Menu cấu hình thông số nhận diện
    st.markdown("#### ⚙️ Ngưỡng Nhận Diện")
    conf_threshold = st.slider(
        "Độ tự tin tối thiểu (Confidence):",
        min_value=0.05,
        max_value=1.00,
        value=0.40,
        step=0.05,
        help="Lọc bỏ những dự đoán có độ tự tin thấp hơn giá trị này."
    )
    
    iou_threshold = st.slider(
        "Ngưỡng khử trùng lặp (IoU / NMS):",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Loại bỏ các Bounding Box chồng lấn trên cùng đối tượng."
    )

    st.markdown("---")
    with st.expander("ℹ️ Thông Tin Đồ Án", expanded=False):
        st.markdown(
            "- **Đề tài**: Nhận diện biển báo giao thông Việt Nam\n"
            "- **Mô hình**: Ultralytics YOLOv11\n"
            "- **Weights**: `traffic_sign_8class_best.pt`\n"
            "- **Tính năng nâng cao**: Đối chiếu trực quan Crop BBox\n"
            "- **Video**: Ghép đôi Side-by-Side đồng bộ 100%\n"
            "- **Framework**: Streamlit, OpenCV, ImageIO"
        )
    st.caption("© 2026 Traffic AI Assistant")


# ==============================================================================
# 7. BANNER TIÊU ĐỀ CHÍNH HIỆN ĐẠI (HERO BANNER)
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🚦 HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG</div>
    <div class="hero-subtitle">Ứng dụng thị giác máy tính hỗ trợ lái xe an toàn và giám sát giao thông thông minh với <b>YOLOv11</b></div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 8. BỐ CỤC 4 TABS CHUYÊN BIỆT
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Nhận Diện Ảnh & Video",
    "📊 Thống Kê & Đánh Giá",
    "🎥 Nhận Diện Trực Tiếp (Webcam)",
    "📖 Tra Cứu 8 Lớp Biển Báo"
])


# ==============================================================================
# TAB 1: NHẬN DIỆN ẢNH VÀ VIDEO (+ ĐỐI CHIẾU TRỰC QUAN THUMBNAIL CẬN CẢNH)
# ==============================================================================
with tab1:
    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        media_mode = st.radio(
            "Chọn định dạng tập tin đầu vào:",
            options=["Hình Ảnh (.jpg, .png, .jpeg)", "Video (.mp4, .avi, .mov)"],
            horizontal=True
        )
    with col_opt2:
        compare_mode = False
        if "Hình Ảnh" in media_mode:
            compare_mode = st.checkbox(
                "🔬 Bật chế độ so sánh độ nhạy (2 mức Confidence)",
                value=False,
                help="Xem nhanh kết quả nhận diện của cùng 1 ảnh ở 2 mức độ tự tin khác nhau."
            )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 1.1: XỬ LÝ HÌNH ẢNH
    # --------------------------------------------------------------------------
    if "Hình Ảnh" in media_mode:
        uploaded_image = st.file_uploader(
            "Tải lên ảnh chụp biển báo giao thông:",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        
        if uploaded_image is not None:
            try:
                # Đọc ảnh an toàn bằng OpenCV
                raw_bytes = uploaded_image.getvalue()
                np_buffer = np.frombuffer(raw_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                
                if img_bgr is None:
                    st.error("❌ Không thể giải mã tập tin ảnh. Vui lòng thử lại với một ảnh khác!")
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Chế độ so sánh độ nhạy
                    if compare_mode:
                        st.info("🔬 **Chế Độ So Sánh Độ Nhạy (Confidence Comparison)**: Đang chạy nhận diện song song ở 2 mức ngưỡng.")
                        
                        cmp_col_s1, cmp_col_s2 = st.columns(2)
                        with cmp_col_s1:
                            conf_low = st.slider("Ngưỡng thấp (Nhạy cao - bắt nhiều hơn):", 0.10, 0.50, 0.25, 0.05)
                        with cmp_col_s2:
                            conf_high = st.slider("Ngưỡng cao (Lọc chặt - độ chính xác cao):", 0.50, 0.95, 0.70, 0.05)
                        
                        res_low = model.predict(source=img_bgr, conf=conf_low, iou=iou_threshold, verbose=False)[0]
                        res_high = model.predict(source=img_bgr, conf=conf_high, iou=iou_threshold, verbose=False)[0]
                        
                        img_low_rgb = cv2.cvtColor(res_low.plot(), cv2.COLOR_BGR2RGB)
                        img_high_rgb = cv2.cvtColor(res_high.plot(), cv2.COLOR_BGR2RGB)
                        
                        count_low = len(res_low.boxes) if res_low.boxes is not None else 0
                        count_high = len(res_high.boxes) if res_high.boxes is not None else 0
                        
                        cmp_c1, cmp_c2 = st.columns(2)
                        with cmp_c1:
                            st.markdown(f"##### 🎯 Ngưỡng Tự Tin: **{conf_low:.2f}** (Phát hiện: **{count_low}** biển)")
                            st.image(img_low_rgb, use_container_width=True)
                        with cmp_c2:
                            st.markdown(f"##### 🛡️ Ngưỡng Tự Tin: **{conf_high:.2f}** (Phát hiện: **{count_high}** biển)")
                            st.image(img_high_rgb, use_container_width=True)
                            
                        st.caption("💡 *Lưu ý kỹ thuật: Ngưỡng độ tự tin thấp hỗ trợ phát hiện các biển báo ở cự ly xa hoặc điều kiện ánh sáng yếu; trong khi ngưỡng cao giúp tối ưu độ chính xác và triệt tiêu cảnh báo giả (False Positives).*")
                    
                    # Chế độ tiêu chuẩn
                    else:
                        t_start = time.time()
                        results = model.predict(
                            source=img_bgr,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            verbose=False
                        )
                        inference_time = (time.time() - t_start) * 1000
                        result = results[0]
                        
                        annotated_bgr = result.plot()
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Thu thập danh sách phát hiện kèm ảnh crop đối chiếu
                        image_detections = []
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                cid = int(box.cls[0].item())
                                raw_name = model.names[cid] if model.names else str(cid)
                                conf = float(box.conf[0].item())
                                coords = [round(v, 1) for v in box.xyxy[0].tolist()]
                                w_box = int(coords[2] - coords[0])
                                h_box = int(coords[3] - coords[1])
                                
                                # Cắt ảnh cận cảnh và tạo thumbnail nhẹ
                                thumb_uri, crop_rgb = crop_and_encode_thumbnail(img_bgr, coords, max_size=(100, 100), pad=10)
                                
                                image_detections.append({
                                    "STT": len(image_detections) + 1,
                                    "Ảnh Biển Báo": thumb_uri,
                                    "Frame/Nguồn": f"Ảnh: {uploaded_image.name}",
                                    "Tên Biển Báo": format_class_name(raw_name),
                                    "Mã Lớp": raw_name,
                                    "Độ Tự Tin (Conf)": round(conf, 4),
                                    "Kích Thước BBox": f"{w_box}x{h_box} px",
                                    "Tọa Độ [x1, y1, x2, y2]": str(coords),
                                    "_crop_rgb": crop_rgb,
                                    "_full_frame_rgb": annotated_rgb
                                })
                        
                        st.session_state["detection_records"] = image_detections
                        st.session_state["last_source_type"] = f"Ảnh: {uploaded_image.name}"
                        
                        st.markdown("#### 🔍 Kết Quả Nhận Diện Song Song:")
                        col_in, col_out = st.columns(2)
                        with col_in:
                            st.info("📷 **Ảnh Đầu Vào Gốc**")
                            st.image(img_rgb, use_container_width=True)
                        with col_out:
                            st.success("🎯 **Ảnh Đã Nhận Diện (YOLOv11)**")
                            st.image(annotated_rgb, use_container_width=True)
                            
                            ok_enc, buf_enc = cv2.imencode(".jpg", annotated_bgr)
                            if ok_enc:
                                st.download_button(
                                    label="📥 Tải Ảnh Kết Quả Về Máy",
                                    data=buf_enc.tobytes(),
                                    file_name=f"detected_{uploaded_image.name}",
                                    mime="image/jpeg"
                                )
                        
                        # Thẻ đo lường nhanh
                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric("Thời Gian Suy Luận", f"{inference_time:.1f} ms")
                        mcol2.metric("Số Biển Báo Phát Hiện", f"{len(image_detections)}")
                        avg_conf = (np.mean([d["Độ Tự Tin (Conf)"] for d in image_detections]) * 100) if image_detections else 0
                        mcol3.metric("Độ Tin Cậy Trung Bình", f"{avg_conf:.1f}%")
                        
                        # BẢNG TÓM TẮT & BẢNG ĐỐI CHIẾU TRỰC QUAN CÓ THUMBNAIL
                        st.markdown("---")
                        st.markdown("#### 📋 1. Bảng Tóm Tắt Số Lượng Theo Từng Loại:")
                        if image_detections:
                            df_sum = create_summary_table(image_detections)
                            st.dataframe(df_sum, use_container_width=True)
                            
                            st.markdown("#### 🔎 2. Bảng Đối Chiếu Trực Quan Từng Biển Báo (Có Thumbnail Cắt Sát Viền):")
                            st.caption("💡 Hội đồng có thể quan sát trực tiếp hình ảnh thumbnail cắt từ bounding box tương ứng:")
                            
                            # Hiển thị bảng kèm cột ImageColumn
                            display_cols = ["Ảnh Biển Báo", "STT", "Tên Biển Báo", "Mã Lớp", "Độ Tự Tin (Conf)", "Kích Thước BBox", "Tọa Độ [x1, y1, x2, y2]"]
                            df_display = pd.DataFrame(image_detections)[display_cols]
                            
                            st.dataframe(
                                df_display,
                                column_config={
                                    "Ảnh Biển Báo": st.column_config.ImageColumn(
                                        "Ảnh Cắt Biển Báo",
                                        help="Vùng ảnh Bounding Box được cắt tự động",
                                        width="small"
                                    ),
                                    "Độ Tự Tin (Conf)": st.column_config.ProgressColumn(
                                        "Độ Tự Tin",
                                        format="%.2f",
                                        min_value=0.0,
                                        max_value=1.0
                                    )
                                },
                                use_container_width=True,
                                height=280
                            )
                            
                            # KHU VỰC PHÓNG TO ĐỐI CHIẾU CHI TIẾT (INSPECTOR)
                            with st.expander("🔬 Xem Phóng To Đối Chiếu Chi Tiết Từng Biển Báo (Inspector)", expanded=True):
                                sel_idx = st.selectbox(
                                    "Chọn biển báo để kiểm chứng độ chính xác:",
                                    options=range(len(image_detections)),
                                    format_func=lambda i: f"#{image_detections[i]['STT']} - {image_detections[i]['Tên Biển Báo']} ({image_detections[i]['Độ Tự Tin (Conf)']*100:.1f}%)"
                                )
                                sel_item = image_detections[sel_idx]
                                
                                insp_col1, insp_col2 = st.columns([1, 2])
                                with insp_col1:
                                    st.markdown("##### 🔍 Ảnh Cận Cảnh (Cropped BBox):")
                                    st.image(sel_item["_crop_rgb"], width=220, caption=f"Kích thước: {sel_item['Kích Thước BBox']}")
                                with insp_col2:
                                    st.markdown("##### 📌 Thông Số Kiểm Chứng:")
                                    st.markdown(f"- **Tên nhận diện**: `{sel_item['Tên Biển Báo']}`")
                                    st.markdown(f"- **Mã lớp YOLOv11**: `{sel_item['Mã Lớp']}`")
                                    st.markdown(f"- **Độ tin cậy xác suất**: **{sel_item['Độ Tự Tin (Conf)']*100:.2f}%**")
                                    st.markdown(f"- **Tọa độ trên ảnh**: `{sel_item['Tọa Độ [x1, y1, x2, y2]']}`")
                        else:
                            st.warning("⚠️ Không phát hiện biển báo nào đạt ngưỡng độ tự tin này. Bạn có thể giảm thanh trượt 'Confidence' ở thanh bên trái!")
                            
            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi khi xử lý ảnh: {e}")

    # --------------------------------------------------------------------------
    # 1.2: XỬ LÝ VIDEO CLIP (GHÉP ĐÔI CHIA ĐÔI MÀN HÌNH + CẮT ẢNH ĐỐI CHIẾU TỪNG FRAME)
    # --------------------------------------------------------------------------
    else:
        uploaded_video = st.file_uploader(
            "Tải lên video camera hành trình / đường phố:",
            type=["mp4", "avi", "mov", "mkv"]
        )
        
        if uploaded_video is not None:
            # Ghi video ra file tạm an toàn bằng getvalue()
            t_vid_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_vid_in.write(uploaded_video.getvalue())
            in_vid_path = t_vid_in.name
            t_vid_in.close()
            
            # Đọc thông số video gốc
            cap_info = cv2.VideoCapture(in_vid_path)
            if not cap_info.isOpened():
                st.error("❌ Không thể đọc file video này. Vui lòng kiểm tra lại định dạng file!")
            else:
                total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                orig_fps = cap_info.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_info.release()
                
                duration = total_frames / orig_fps if orig_fps > 0 else 0
                st.info(
                    f"🎬 **Thông Số Video Gốc**: Kích thước mỗi khung: **{width}x{height}** | "
                    f"FPS: **{orig_fps:.1f}** | Tổng frames: **{total_frames}** | "
                    f"Thời lượng gốc: **{duration:.1f} giây**"
                )
                
                # CẤU HÌNH XỬ LÝ VÀ SLOW-MOTION
                st.markdown("#### ⚙️ Cấu Hình Ghép Đôi & Làm Chậm Video (Slow-Motion):")
                v_cfg_col1, v_cfg_col2 = st.columns(2)
                
                with v_cfg_col1:
                    step_skip = st.select_slider(
                        "1. Bước nhảy frames (Frame Skip):",
                        options=[1, 2, 3, 5],
                        value=1,
                        format_func=lambda x: "Mọi frame (100% đầy đủ từng frame)" if x == 1 else f"Lấy mẫu mỗi {x} frames (Tăng tốc độ xử lý)",
                        help="Nếu video dài, bạn có thể nhảy frame để tiết kiệm thời gian chạy demo."
                    )
                
                with v_cfg_col2:
                    # THANH TRƯỢT HỆ SỐ LÀM CHẬM (SLOW-MOTION)
                    slow_factor = st.slider(
                        "2. Hệ số làm chậm video đầu ra (Slow-Motion):",
                        min_value=0.20,
                        max_value=1.00,
                        value=0.50,
                        step=0.05,
                        format="%.2fx",
                        help="Hệ số < 1.0 sẽ làm chậm video (ví dụ 0.50x giúp video chạy chậm gấp 2 lần, dễ dàng nhìn rõ biển báo)."
                    )
                
                # Tính toán FPS video kết quả
                calc_out_fps = max(round((orig_fps / step_skip) * slow_factor, 1), 3.0)
                combined_w = width * 2
                combined_h = height
                
                st.success(
                    f"✨ **Cơ chế ghép đôi đồng bộ & trích xuất đối chiếu**: Video kết quả sẽ ghép đôi tự động 2 khung hình cạnh nhau: "
                    f"**Gốc ({width}x{height}) + Nhận diện ({width}x{height}) ➔ Khung hình chia đôi ({combined_w}x{combined_h})** "
                    f"ở tốc độ **{calc_out_fps} FPS** (chậm **{1/slow_factor:.1f} lần** so với gốc). Đồng thời tự động cắt ảnh từng biển báo phát hiện để đối chiếu.",
                    icon="🎯"
                )
                
                # Nút bấm bắt đầu xử lý video
                if st.button("🚀 Bắt Đầu Xử Lý, Ghép Đôi & Cắt Ảnh Đối Chiếu", type="primary"):
                    prog_bar = st.progress(0)
                    prog_text = st.empty()
                    live_prev = st.empty()
                    
                    # Đường dẫn file video H.264 đầu ra
                    out_vid_path = os.path.join(tempfile.gettempdir(), f"combined_sidebyside_{int(time.time())}.mp4")
                    
                    cap = cv2.VideoCapture(in_vid_path)
                    f_count = 0
                    p_count = 0
                    video_records = []
                    t_vid_start = time.time()
                    
                    # Giới hạn tối đa 120 phát hiện lưu trữ để tránh tràn RAM bộ nhớ trình duyệt
                    MAX_SAVED_CROPS = 120
                    
                    # Khởi tạo Video Writer chuẩn Web H.264
                    use_imageio_writer = HAS_IMAGEIO
                    if use_imageio_writer:
                        video_writer = imageio.get_writer(
                            out_vid_path,
                            fps=calc_out_fps,
                            codec='libx264',
                            pixelformat='yuv420p',
                            macro_block_size=None
                        )
                    else:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(out_vid_path, fourcc, calc_out_fps, (combined_w, combined_h))
                    
                    try:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                break
                            
                            f_count += 1
                            if f_count % step_skip != 0:
                                continue
                            
                            # 1. Khung hình gốc (bên trái)
                            frame_left = frame.copy()
                            if frame_left.shape[1] != width or frame_left.shape[0] != height:
                                frame_left = cv2.resize(frame_left, (width, height))
                                
                            cv2.rectangle(frame_left, (0, 0), (280, 42), (15, 23, 42), -1)
                            cv2.putText(
                                frame_left,
                                "VIDEO GOC (INPUT)",
                                (14, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.72,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA
                            )
                            
                            # 2. Khung hình nhận diện YOLOv11 (bên phải)
                            res = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                            frame_right = res.plot()
                            if frame_right.shape[1] != width or frame_right.shape[0] != height:
                                frame_right = cv2.resize(frame_right, (width, height))
                                
                            cv2.rectangle(frame_right, (0, 0), (330, 42), (2, 132, 199), -1)
                            cv2.putText(
                                frame_right,
                                "YOLOV11 DETECTED",
                                (14, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.72,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA
                            )
                            
                            # 3. Ghép ngang 2 khung hình (Side-by-Side Horizontal Stack)
                            combined_frame_bgr = np.hstack([frame_left, frame_right])
                            
                            divider_x = width
                            cv2.line(combined_frame_bgr, (divider_x, 0), (divider_x, height), (255, 255, 255), 2)
                            
                            # Ghi vào file video kết quả
                            if use_imageio_writer:
                                combined_frame_rgb = cv2.cvtColor(combined_frame_bgr, cv2.COLOR_BGR2RGB)
                                video_writer.append_data(combined_frame_rgb)
                            else:
                                video_writer.write(combined_frame_bgr)
                                
                            p_count += 1
                            
                            # 4. Trích xuất và cắt ảnh Bounding Box từng biển báo (Tối ưu bộ nhớ)
                            if res.boxes is not None and len(res.boxes) > 0:
                                for b in res.boxes:
                                    cid = int(b.cls[0].item())
                                    raw_n = model.names[cid] if model.names else str(cid)
                                    c_conf = float(b.conf[0].item())
                                    coords = [round(v, 1) for v in b.xyxy[0].tolist()]
                                    w_b = int(coords[2] - coords[0])
                                    h_b = int(coords[3] - coords[1])
                                    
                                    # Cắt ảnh gọn nhẹ khi còn dưới ngưỡng giới hạn bộ nhớ
                                    if len(video_records) < MAX_SAVED_CROPS:
                                        t_uri, c_rgb = crop_and_encode_thumbnail(frame, coords, max_size=(100, 100), pad=8)
                                    else:
                                        t_uri, c_rgb = "", np.zeros((10, 10, 3), dtype=np.uint8)
                                        
                                    video_records.append({
                                        "STT": len(video_records) + 1,
                                        "Ảnh Biển Báo": t_uri,
                                        "Frame/Nguồn": f"Frame #{f_count}",
                                        "Tên Biển Báo": format_class_name(raw_n),
                                        "Mã Lớp": raw_n,
                                        "Độ Tự Tin (Conf)": round(c_conf, 4),
                                        "Kích Thước BBox": f"{w_b}x{h_b} px",
                                        "Tọa Độ [x1, y1, x2, y2]": str(coords),
                                        "_crop_rgb": c_rgb
                                    })
                            
                            # Cập nhật thanh tiến trình
                            progress = min(f_count / max(total_frames, 1), 1.0)
                            prog_bar.progress(progress)
                            elapsed = time.time() - t_vid_start
                            cur_fps = p_count / elapsed if elapsed > 0 else 0
                            prog_text.markdown(f"⏳ Đang xử lý: Frame **{f_count}/{total_frames}** ({progress*100:.1f}%) | Tốc độ xử lý AI: **{cur_fps:.1f} FPS**")
                            
                            if p_count % 6 == 0:
                                preview_thumb = cv2.resize(combined_frame_bgr, (720, int(720 * height / combined_w)))
                                live_prev.image(
                                    cv2.cvtColor(preview_thumb, cv2.COLOR_BGR2RGB),
                                    caption=f"Đang render video ghép đôi đồng bộ (Frame {f_count})",
                                    use_container_width=True
                                )
                                
                    finally:
                        cap.release()
                        if use_imageio_writer:
                            video_writer.close()
                        else:
                            video_writer.release()
                            
                    prog_bar.progress(1.0)
                    prog_text.success(f"🎉 Hoàn thành ghép đôi video! Đã tạo **{p_count} frames** đồng bộ ở tốc độ **{calc_out_fps} FPS**.")
                    live_prev.empty()
                    
                    with open(out_vid_path, "rb") as f_out:
                        final_video_bytes = f_out.read()
                    
                    st.session_state["detection_records"] = video_records
                    st.session_state["last_source_type"] = f"Video Ghép Đôi: {uploaded_video.name} ({p_count} frames)"
                    st.session_state["combined_video_bytes"] = final_video_bytes

                # HIỂN THỊ DUY NHẤT 1 VIDEO GHÉP ĐÔI CHIA ĐÔI MÀN HÌNH ĐỒNG BỘ 100%
                if "combined_video_bytes" in st.session_state:
                    st.markdown("---")
                    st.markdown("#### 📺 Video Ghép Đôi Chia Đôi Màn Hình (Đồng Bộ 100% Khung Hình):")
                    st.info("💡 **Tính năng đồng bộ khung hình:** Khi nhấn nút phát (Play), cả hai luồng hình ảnh (Video gốc & Video nhận diện) sẽ tự động chạy song song khớp từng mili-giây, hỗ trợ quan sát và đối chiếu kết quả nhận diện trực quan tức thì.")
                    
                    st.video(st.session_state["combined_video_bytes"])
                    
                    st.download_button(
                        label="📥 Tải Video Ghép Đôi Chia Đôi Màn Hình Về Máy (MP4)",
                        data=st.session_state["combined_video_bytes"],
                        file_name="traffic_signs_side_by_side_yolov11.mp4",
                        mime="video/mp4"
                    )
                    
                    # BẢNG TÓM TẮT & BẢNG ĐỐI CHIẾU TRỰC QUAN CÓ THUMBNAIL
                    st.markdown("---")
                    st.markdown("#### 📋 1. Bảng Tóm Tắt Số Lượng Theo Từng Loại Trong Video:")
                    
                    v_records = st.session_state.get("detection_records", [])
                    if v_records:
                        df_v_summary = create_summary_table(v_records)
                        st.dataframe(df_v_summary, use_container_width=True)
                        
                        st.markdown("#### 🔎 2. Bảng Đối Chiếu Trực Quan Biển Báo Cắt Từng Frame (Có Thumbnail):")
                        
                        display_cols = ["Ảnh Biển Báo", "STT", "Frame/Nguồn", "Tên Biển Báo", "Mã Lớp", "Độ Tự Tin (Conf)", "Kích Thước BBox", "Tọa Độ [x1, y1, x2, y2]"]
                        df_v_disp = pd.DataFrame(v_records)[display_cols]
                        
                        st.dataframe(
                            df_v_disp,
                            column_config={
                                "Ảnh Biển Báo": st.column_config.ImageColumn(
                                    "Ảnh Cắt Biển Báo",
                                    help="Vùng ảnh Bounding Box được cắt tự động",
                                    width="small"
                                ),
                                "Độ Tự Tin (Conf)": st.column_config.ProgressColumn(
                                    "Độ Tự Tin",
                                    format="%.2f",
                                    min_value=0.0,
                                    max_value=1.0
                                )
                            },
                            use_container_width=True,
                            height=320
                        )
                        
                        # KHU VỰC PHÓNG TO ĐỐI CHIẾU CHI TIẾT (INSPECTOR)
                        with st.expander("🔬 Xem Phóng To Đối Chiếu Chi Tiết Từng Biển Báo Trong Video (Inspector)", expanded=True):
                            valid_indices = [i for i, r in enumerate(v_records) if r["_crop_rgb"].shape[0] > 10]
                            if valid_indices:
                                sel_v_idx = st.selectbox(
                                    "Chọn biển báo để kiểm chứng độ chính xác:",
                                    options=valid_indices,
                                    format_func=lambda i: f"#{v_records[i]['STT']} ({v_records[i]['Frame/Nguồn']}) - {v_records[i]['Tên Biển Báo']} ({v_records[i]['Độ Tự Tin (Conf)']*100:.1f}%)"
                                )
                                sel_v_item = v_records[sel_v_idx]
                                
                                vinsp_col1, vinsp_col2 = st.columns([1, 2])
                                with vinsp_col1:
                                    st.markdown("##### 🔍 Ảnh Cắt Cận Cảnh (Cropped BBox):")
                                    st.image(sel_v_item["_crop_rgb"], width=220, caption=f"Kích thước: {sel_v_item['Kích Thước BBox']}")
                                with vinsp_col2:
                                    st.markdown("##### 📌 Thông Số Kiểm Chứng:")
                                    st.markdown(f"- **Xuất hiện tại**: `{sel_v_item['Frame/Nguồn']}`")
                                    st.markdown(f"- **Tên nhận diện**: `{sel_v_item['Tên Biển Báo']}`")
                                    st.markdown(f"- **Mã lớp YOLOv11**: `{sel_v_item['Mã Lớp']}`")
                                    st.markdown(f"- **Độ tin cậy xác suất**: **{sel_v_item['Độ Tự Tin (Conf)']*100:.2f}%**")
                                    st.markdown(f"- **Tọa độ Bounding Box**: `{sel_v_item['Tọa Độ [x1, y1, x2, y2]']}`")
                    else:
                        st.warning("⚠️ Không phát hiện thấy biển báo nào đạt ngưỡng độ tự tin trong đoạn video này.")


# ==============================================================================
# TAB 2: THỐNG KÊ VÀ BIỂU ĐỒ ĐÁNH GIÁ (ANALYTICS & CHARTS)
# ==============================================================================
with tab2:
    st.markdown("### 📊 Báo Cáo Thống Kê & Phân Tích Kết Quả Nhận Diện")
    
    records = st.session_state.get("detection_records", [])
    source_type = st.session_state.get("last_source_type", "Chưa có")
    
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.write(f"Nguồn dữ liệu hiện tại: **{source_type}** | Tổng số phát hiện: **{len(records)}**")
    with col_hdr2:
        if st.button("🎲 Tạo Dữ Liệu Mẫu (Demo Data)"):
            sample_classes = list(SIGN_CATALOG.keys())
            sample_records = []
            for i in range(50):
                cls_raw = np.random.choice(sample_classes)
                sample_records.append({
                    "STT": i + 1,
                    "Ảnh Biển Báo": "",
                    "Frame/Nguồn": f"Frame #{np.random.randint(1, 100)}",
                    "Tên Biển Báo": format_class_name(cls_raw),
                    "Mã Lớp": cls_raw,
                    "Độ Tự Tin (Conf)": round(float(np.random.uniform(0.45, 0.98)), 4),
                    "Kích Thước BBox": "80x85 px",
                    "Tọa Độ [x1, y1, x2, y2]": "[120, 140, 240, 260]",
                    "_crop_rgb": np.zeros((10, 10, 3), dtype=np.uint8)
                })
            st.session_state["detection_records"] = sample_records
            st.session_state["last_source_type"] = "Dữ Liệu Demo Mô Phỏng"
            st.rerun()

    if not records:
        st.info(
            "💡 **Chưa có dữ liệu thống kê!**\n\n"
            "- Hãy chuyển sang **Tab 1** để tải ảnh hoặc video lên và chạy nhận diện bằng YOLOv11.\n"
            "- Hoặc nhấn nút **'🎲 Tạo Dữ Liệu Mẫu (Demo Data)'** ở góc trên để xem ngay các biểu đồ."
        )
    else:
        df = pd.DataFrame(records)
        
        # 1. Các thẻ chỉ số KPI
        st.markdown("#### 1. Các Chỉ Số Đánh Giá Chính (KPIs)")
        k1, k2, k3, k4 = st.columns(4)
        
        total_detections = len(df)
        unique_classes = df["Tên Biển Báo"].nunique()
        avg_confidence = df["Độ Tự Tin (Conf)"].mean() * 100
        max_confidence = df["Độ Tự Tin (Conf)"].max() * 100
        
        k1.metric("Tổng Số Biển Báo", f"{total_detections:,}")
        k2.metric("Số Lớp Xuất Hiện", f"{unique_classes} loại")
        k3.metric("Độ Tin Cậy Trung Bình", f"{avg_confidence:.2f}%")
        k4.metric("Độ Tin Cậy Cao Nhất", f"{max_confidence:.2f}%")
        
        st.markdown("---")
        
        # 2. Biểu đồ cột và biểu đồ tròn
        st.markdown("#### 2. Tần Suất & Tỷ Lệ Xuất Hiện Của Từng Loại Biển Báo")
        c_col1, c_col2 = st.columns(2)
        
        counts = df["Tên Biển Báo"].value_counts().reset_index()
        counts.columns = ["Tên Biển Báo", "Số Lượng"]
        
        with c_col1:
            fig_bar = px.bar(
                counts,
                x="Tên Biển Báo",
                y="Số Lượng",
                text="Số Lượng",
                color="Số Lượng",
                color_continuous_scale="Blues",
                title="📊 Biểu Đồ Cột: Tần Suất Xuất Hiện",
                labels={"Tên Biển Báo": "Loại Biển", "Số Lượng": "Số Lần Phát Hiện"}
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(
                xaxis_tickangle=-30,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=80)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_col2:
            fig_pie = px.pie(
                counts,
                names="Tên Biển Báo",
                values="Số Lượng",
                title="🍩 Biểu Đồ Tròn: Tỷ Lệ Phân Bố Từng Loại",
                hole=0.42,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        
        # 3. Phân bố độ tin cậy
        st.markdown("#### 3. Phân Bố Độ Tự Tin (Confidence Score Distribution)")
        conf_col1, conf_col2 = st.columns(2)
        
        with conf_col1:
            fig_hist = px.histogram(
                df,
                x="Độ Tự Tin (Conf)",
                nbins=15,
                marginal="rug",
                title="📈 Phân Phối Điểm Độ Tin Cậy (Histogram)",
                color_discrete_sequence=["#0284c7"],
                labels={"Độ Tự Tin (Conf)": "Ngưỡng Độ Tự Tin"}
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=360,
                yaxis_title="Số Lượng Mẫu"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with conf_col2:
            fig_box = px.box(
                df,
                x="Tên Biển Báo",
                y="Độ Tự Tin (Conf)",
                color="Tên Biển Báo",
                title="📦 Hộp Phân Vị Độ Tự Tin Từng Loại (Boxplot)",
                labels={"Tên Biển Báo": "Loại Biển", "Độ Tự Tin (Conf)": "Độ Tin Cậy"}
            )
            fig_box.update_layout(
                xaxis_tickangle=-30,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(l=20, r=20, t=50, b=80)
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        
        # 4. Bảng dữ liệu chi tiết và xuất CSV
        st.markdown("#### 4. Bảng Dữ Liệu Chi Tiết Các Lần Nhận Diện (Xuất CSV Báo Cáo)")
        filter_opts = ["Tất cả"] + sorted(df["Tên Biển Báo"].unique().tolist())
        selected_filter = st.selectbox("Lọc theo loại biển báo:", filter_opts)
        
        filtered_df = df if selected_filter == "Tất cả" else df[df["Tên Biển Báo"] == selected_filter]
        
        # Loại bỏ các cột internal numpy array trước khi xuất bảng/CSV
        export_cols = [c for c in filtered_df.columns if not c.startswith("_") and c != "Ảnh Biển Báo"]
        st.dataframe(filtered_df[export_cols], use_container_width=True, height=280)
        
        csv_bytes = filtered_df[export_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Tải Báo Cáo Kết Quả Dạng File CSV",
            data=csv_bytes,
            file_name="bao_cao_nhan_dien_bien_bao_yolov11.csv",
            mime="text/csv"
        )


# ==============================================================================
# TAB 3: NHẬN DIỆN QUA WEBCAM TRỰC TIẾP
# ==============================================================================
with tab3:
    st.markdown("### 🎥 Nhận Diện Trực Tiếp Bằng Camera / Webcam")
    st.markdown("Hỗ trợ 2 phương thức: **Chụp ảnh tức thì (Snapshot)** hoặc **Mở luồng camera thời gian thực (Live Stream)**.")
    
    # Quản lý trạng thái kích hoạt camera để tránh trình duyệt tự động xin quyền khi vừa vào trang
    if "camera_active" not in st.session_state:
        st.session_state["camera_active"] = False

    if not st.session_state["camera_active"]:
        st.info(
            "🔒 **Bảo Vệ Quyền Riêng Tư & Tối Ưu Tải Trang:**\n\n"
            "Chức năng Camera đang ở chế độ **TẮT** để tránh trình duyệt tự động hiển thị hộp thoại xin quyền truy cập khi vừa mở website.\n\n"
            "👉 Hãy nhấn nút bên dưới khi bạn đã sẵn sàng sử dụng Camera trên thiết bị để nhận diện biển báo."
        )
        if st.button("🚀 Bật Camera Nhận Diện", type="primary"):
            st.session_state["camera_active"] = True
            st.rerun()
    else:
        cam_hdr_c1, cam_hdr_c2 = st.columns([3, 1])
        with cam_hdr_c1:
            st.success("🟢 **Camera đã được kích hoạt.** Bạn có thể chọn chụp ảnh nhanh hoặc truyền luồng trực tiếp.")
        with cam_hdr_c2:
            if st.button("❌ Tắt Camera", use_container_width=True):
                st.session_state["camera_active"] = False
                st.session_state["webcam_running"] = False
                st.rerun()

        cam_option = st.radio(
            "Chọn chế độ camera:",
            options=["📷 Chụp ảnh nhanh từ Camera (Snapshot)", "🔴 Truyền luồng thời gian thực (Live Stream)"],
            horizontal=True
        )
    
    # Snapshot
    if st.session_state["camera_active"] and "Snapshot" in cam_option:
        st.info("💡 Hướng camera về phía biển báo và nhấn nút **'Take Photo'** để chụp.")
        photo_capture = st.camera_input("Chụp ảnh nhận diện:")
        
        if photo_capture is not None:
            try:
                cam_raw = photo_capture.getvalue()
                cam_bgr = cv2.imdecode(np.frombuffer(cam_raw, np.uint8), cv2.IMREAD_COLOR)
                
                if cam_bgr is not None:
                    cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
                    
                    t_cam = time.time()
                    cam_results = model.predict(source=cam_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                    cam_fps = 1.0 / (time.time() - t_cam) if (time.time() - t_cam) > 0 else 0
                    
                    annotated_cam = cv2.cvtColor(cam_results.plot(), cv2.COLOR_BGR2RGB)
                    
                    cam_detections = []
                    if cam_results.boxes is not None and len(cam_results.boxes) > 0:
                        for b in cam_results.boxes:
                            cid = int(b.cls[0].item())
                            cname = model.names[cid] if model.names else str(cid)
                            conf = float(b.conf[0].item())
                            coords = [round(v, 1) for v in b.xyxy[0].tolist()]
                            w_b = int(coords[2] - coords[0])
                            h_b = int(coords[3] - coords[1])
                            
                            t_uri, c_rgb = crop_and_encode_thumbnail(cam_bgr, coords, max_size=(100, 100), pad=8)
                            
                            cam_detections.append({
                                "STT": len(cam_detections) + 1,
                                "Ảnh Biển Báo": t_uri,
                                "Frame/Nguồn": "Webcam Snapshot",
                                "Tên Biển Báo": format_class_name(cname),
                                "Mã Lớp": cname,
                                "Độ Tự Tin (Conf)": round(conf, 4),
                                "Kích Thước BBox": f"{w_b}x{h_b} px",
                                "Tọa Độ [x1, y1, x2, y2]": str(coords),
                                "_crop_rgb": c_rgb
                            })
                    
                    st.session_state["detection_records"] = cam_detections
                    st.session_state["last_source_type"] = "Webcam Snapshot"
                    
                    st.markdown("#### 🎯 Kết Quả Phân Tích:")
                    w1, w2 = st.columns(2)
                    with w1:
                        st.caption("Ảnh Gốc Vừa Chụp")
                        st.image(cam_rgb, use_container_width=True)
                    with w2:
                        st.caption("Ảnh Đã Nhận Diện (YOLOv11)")
                        st.image(annotated_cam, use_container_width=True)
                        
                    st.success(f"Tìm thấy **{len(cam_detections)}** biển báo! Tốc độ: **{cam_fps:.1f} FPS**")
                    
                    if cam_detections:
                        st.markdown("##### 📋 Tóm Tắt Phát Hiện:")
                        st.dataframe(create_summary_table(cam_detections), use_container_width=True)
                        
                        # Hiển thị bảng thumbnail đối chiếu
                        st.markdown("##### 🔎 Đối Chiếu Biển Báo Vừa Chụp (Thumbnail):")
                        disp_cols = ["Ảnh Biển Báo", "STT", "Tên Biển Báo", "Độ Tự Tin (Conf)", "Kích Thước BBox", "Tọa Độ [x1, y1, x2, y2]"]
                        st.dataframe(
                            pd.DataFrame(cam_detections)[disp_cols],
                            column_config={
                                "Ảnh Biển Báo": st.column_config.ImageColumn("Ảnh Cắt", width="small")
                            },
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý webcam: {e}")
                
    # Live Stream
    elif st.session_state["camera_active"]:
        st.warning("⚠️ **Lưu ý**: Chế độ này sẽ kết nối trực tiếp camera phần cứng qua OpenCV. Nhấn **'Bắt đầu Webcam'** để mở luồng.")
        
        c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 2])
        btn_start = c_btn1.button("▶️ Bắt Đầu Webcam", type="primary")
        btn_stop = c_btn2.button("⏹️ Dừng Webcam")
        cam_id = c_btn3.number_input("Cổng Camera ID:", min_value=0, max_value=5, value=0, step=1)
        
        frame_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        if btn_start:
            st.session_state["webcam_running"] = True
        if btn_stop:
            st.session_state["webcam_running"] = False
            
        if st.session_state.get("webcam_running", False):
            cap = cv2.VideoCapture(int(cam_id))
            if not cap.isOpened():
                st.error(f"❌ Không thể kết nối với Camera ID {cam_id}. Vui lòng kiểm tra lại thiết bị!")
                st.session_state["webcam_running"] = False
            else:
                try:
                    last_time = time.time()
                    while st.session_state.get("webcam_running", False):
                        ret, f_cam = cap.read()
                        if not ret or f_cam is None:
                            break
                        
                        f_res = model.predict(source=f_cam, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                        f_annotated = f_res.plot()
                        
                        now = time.time()
                        fps_calc = 1.0 / (now - last_time) if (now - last_time) > 0 else 0
                        last_time = now
                        
                        cv2.putText(
                            f_annotated,
                            f"YOLOv11 - FPS: {fps_calc:.1f}",
                            (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                        
                        frame_placeholder.image(cv2.cvtColor(f_annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        fps_placeholder.info(f"⚡ Trạng thái: Đang stream trực tiếp | Tốc độ: **{fps_calc:.1f} FPS**")
                except Exception as e:
                    st.error(f"❌ Lỗi luồng webcam: {e}")
                finally:
                    cap.release()
                    frame_placeholder.empty()
                    fps_placeholder.empty()
                    st.session_state["webcam_running"] = False
                    st.info("Đã tắt kết nối Webcam an toàn.")


# ==============================================================================
# TAB 4: TRA CỨU 8 LỚP BIỂN BÁO GIAO THÔNG (CLASSES REFERENCE CATALOG)
# ==============================================================================
with tab4:
    st.markdown("### 📖 Tra Cứu Quy Chuẩn 8 Lớp Biển Báo Giao Thông Đường Bộ")
    st.markdown(
        "Bảng quy chuẩn chi tiết giúp hội đồng và người xem tra cứu nhanh **ý nghĩa, mã hiệu quy chuẩn Việt Nam (QCVN 41:2019/BGTVT)** "
        "và phân nhóm của từng lớp biển báo được huấn luyện trong mô hình."
    )
    st.markdown("---")
    
    col_cards_left, col_cards_right = st.columns(2)
    sign_items = list(SIGN_CATALOG.items())
    
    for idx, (cls_key, info) in enumerate(sign_items):
        target_col = col_cards_left if idx % 2 == 0 else col_cards_right
        
        with target_col:
            st.markdown(f"""
            <div class="sign-card">
                <div class="sign-header">
                    <span>{info['icon']}</span>
                    <span>{info['name_vn']}</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <span class="sign-type {info['type_class']}">{info['group']}</span>
                    <span class="badge">Mã QC: {info['code']}</span>
                    <span class="badge">Dataset Class: <code>{cls_key}</code> (ID: {info['id']})</span>
                </div>
                <p style="color: #475569; font-size: 0.92rem; margin-bottom: 0px; line-height: 1.5;">
                    {info['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.info(
        "💡 **Quy chuẩn kỹ thuật & Phạm vi nhận diện:** Hệ thống được huấn luyện và tối ưu hóa nhận diện dựa trên các nhóm biển báo giao thông đường bộ "
        "trọng điểm tại Việt Nam theo **Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ (QCVN 41:2019/BGTVT)**. "
        "Mô hình YOLOv11 đạt độ tin cậy cao trên các nhóm biển báo có tần suất xuất hiện lớn trong đô thị, đặc biệt là nhóm biển báo cấm và biển cảnh báo nguy hiểm."
    )


# ==============================================================================
# 9. PHẦN CHÂN TRANG (FOOTER)
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.9rem; padding: 10px 0;">
        🚦 Hệ Thống Demo Nhận Diện Biển Báo Giao Thông | Kiến Trúc <b>YOLOv11</b> & Giao Diện <b>Streamlit</b><br>
        Tự động nạp file mô hình: <code>traffic_sign_8class_best.pt</code>
    </div>
    """,
    unsafe_allow_html=True
)
