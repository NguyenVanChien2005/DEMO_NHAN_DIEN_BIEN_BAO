# 🚦 HỆ THỐNG DEMO NHẬN DIỆN BIỂN BÁO GIAO THÔNG VỚI YOLOV11 & STREAMLIT

Hệ thống ứng dụng Web Demo hoàn chỉnh hỗ trợ nhận diện và phân loại 8 lớp biển báo giao thông Việt Nam theo thời gian thực bằng mô hình học sâu **Ultralytics YOLOv11** và giao diện người dùng **Streamlit** tối ưu cho báo cáo/thuyết trình đồ án.

---

## 📁 Cấu Trúc Thư Mục

```text
DEMO_NHAN_DIEN_BIEN_BAO/
│
├── app.py                         # File mã nguồn chính của ứng dụng Streamlit (YOLOv11)
├── traffic_sign_8class_best.pt    # File trọng số mô hình YOLOv11 chuẩn (8 lớp biển báo)
├── requirements.txt               # Danh sách thư viện cần thiết
└── README.md                      # Hướng dẫn sử dụng và thuyết trình đồ án
```

---

## 🚀 Khởi Chạy Ứng Dụng

Trong Terminal của VSCode tại thư mục `d:\DEMO_NHAN_DIEN_BIEN_BAO`, chạy lệnh:

```bash
streamlit run app.py
```
> Trình duyệt web sẽ tự động mở tại địa chỉ: `http://localhost:8501`

---

## 🌟 Điểm Cải Tiến Mới: Trích Xuất & Đối Chiếu Trực Quan

1. **Tự Động Cắt Vùng Biển Báo (Bounding Box Crop)**:
   - Trong quá trình quét ảnh hoặc từng frame video, hệ thống tự động trích xuất vùng biển báo dựa trên tọa độ `[x1, y1, x2, y2]` kèm một khoảng đệm an toàn (padding).
   - Tối ưu bộ nhớ bằng cách nén ảnh thumbnail dạng **JPEG Base64 Data URI (~2KB/mẫu)**, đảm bảo xử lý video dài mà không làm lag hay tràn RAM trình duyệt.

2. **Cột Thumbnail Trực Tiếp Trên Bảng Dữ Liệu**:
   - Sử dụng `st.column_config.ImageColumn` hiển thị ảnh thu nhỏ của biển báo ngay cạnh tên lớp, độ tự tin và tọa độ.
   - Thêm thanh tiến trình `ProgressColumn` hiển thị trực quan mức độ tin cậy của mô hình.

3. **Bộ Phóng To Đối Chiếu Chi Tiết (Inspector)**:
   - Tích hợp menu mở rộng (`st.expander`) cho phép hội đồng chọn bất kỳ biển báo nào đã phát hiện:
     - **Ảnh Cận Cảnh (Cropped Zoom-in)**: Phóng to biển báo để soi rõ hoa văn, số hiệu, ký hiệu.
     - **Thông Số Đo Lường**: Hiển thị kích thước pixel của Bounding Box, frame xuất hiện, tên quy chuẩn QCVN và xác suất độ tin cậy.

4. **Đồng Bộ & Ghép Đôi Video (Side-by-Side Synchronized Single Video)**:
   - Ghép ngang 2 khung hình: **Video gốc (trái) + Video nhận diện YOLOv11 (phải)** thành **1 file video duy nhất chia đôi màn hình** (`np.hstack`).
   - Bấm Play một lần là 2 nửa video chạy song song đồng bộ 100% từng mili-giây.
   - Tích hợp thanh trượt **Làm chậm video (Slow-motion từ 0.20x đến 1.00x)**.
