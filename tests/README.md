# Automated checks

Chạy bằng `python -m pytest -q`. Phần lớn test dùng một gói dữ liệu tổng hợp nhỏ (`abc_uwb_falltrack.synthetic`), không cần dữ liệu thật. Vài test so sánh với code chính thức trong gói dữ liệu (evaluator, bộ lọc nền) và tự bỏ qua nếu gói chưa được tải.

| File | Kiểm tra |
|---|---|
| `test_splitting.py` | nhóm theo người không nằm ở hai fold hoặc hai split; seed cố định cho cùng một split, seed khác vẫn cho split hợp lệ; purge cửa sổ vắt qua hai fold; cửa sổ vắt qua hai fold không có score OOF; cửa sổ luôn theo split của luồng |
| `test_windowing.py` | cửa sổ không vượt quá luồng; chồng lấn ở đầu mút; nhãn mơ hồ; chính sách cho cú ngã bị gắn cờ |
| `test_preprocessing.py` | trừ nền theo khúc giống trừ trên cả luồng và giống bản chính thức; đặc trưng xác định và xử lý NaN/Inf; scaler/imputer chỉ fit trên train |
| `test_metrics.py` | dữ liệu mất cân bằng lớp; fold chỉ có một lớp; quy tắc ghép sự kiện khớp evaluator chính thức |
| `test_postprocessing.py` | gộp sự kiện, độ dài tối thiểu, cooldown, sự kiện không vắt qua hai luồng; làm mượt và run dừng ở cửa sổ bị thiếu (NaN hoặc khoảng trống `w_index`) |
| `test_smoke_pipeline.py` | chạy end-to-end trên dữ liệu tổng hợp, artifact đầy đủ, không ghi đè run, `--help` và CLI train → evaluate → chấm điểm → chẩn đoán model fold |
