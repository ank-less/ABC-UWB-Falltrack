# Đặc tả UWB-FallTrack

Tài liệu này ghi lại các yêu cầu chính thức của challenge để cả nhóm dùng chung. Một mục chỉ được chuyển từ “cần xác nhận” sang “đã xác nhận” khi có nguồn đi kèm: hướng dẫn, trang challenge, dữ liệu mẫu hoặc trao đổi với ban tổ chức.

## Nguồn và mức độ tin cậy

Ba loại nguồn được ghi riêng như sau:

1. **Trang challenge của ban tổ chức** (đọc ngày 2026-10-22):
   - <https://autocare.ai/abc2027/> (trang hội nghị ABC 2027, mục Challenge);
   - <https://sites.google.com/view/abc2027-uwbfalltrack-challenge/> (các trang The Challenge, Registration, Dataset and Tutorial, FAQ, Organisers). Do LIARA Lab (UQAC) chủ trì, Sozo Lab (Kyutech) đồng tổ chức.
2. **Gói dữ liệu `abc2027_uwb_train_v1.zip`**, tải từ link Google Drive đăng trên trang "Dataset and Tutorial" ở trên. Vì gói được phát hành từ trang chính thức, README, `evaluate.py` và `sample_submission.csv` trong gói được xem là tài liệu của ban tổ chức (đọc ngày 2026-10-22). Gói nặng 8,17 GB khi nén và khoảng 12 GB khi giải nén.
3. **Quan sát trực tiếp trên file** (`scripts/inspect_data.py`, chạy ngày 2026-10-22): xác nhận lại schema bằng chính dữ liệu.

Thư mục [Google Drive của nhóm](https://drive.google.com/drive/folders/182TmF7QxRepxTcQosjwLR5O0KWgSvlgv) chỉ là bản chia sẻ nội bộ, không phải nguồn chính thức.

## Đã xác nhận

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Bài toán | Phát hiện và định vị theo thời gian từng cú ngã trong luồng radar UWB liên tục, chưa được cắt đoạn. Luồng trộn ADL với các cú ngã. | Trang challenge; README gói |
| Phiên bản dữ liệu | `abc2027_uwb_train_v1`: 13 luồng train (6,18 h), 5 luồng val (2,05 h), 478 cú ngã có nhãn (train 359, val 119). Test chưa phát hành. | README gói; kiểm tra file |
| Cảm biến | 3 radar Novelda XeThru X4M200 (serial 29171, 29444, 30722), căn hộ LIARA khoảng 40 m² | Trang Dataset; README gói |
| Định dạng file | `streams/<split>/stream_XX.h5`: `cir` có shape `(3, T, 368)`, kiểu float32 (số thực); `t` tính bằng giây; `radar_serials`; attribute `split`, `stream_index`, `seed`, `fade_s` | README gói; kiểm tra file |
| Tần số, đồng bộ | Ba radar được căn về chung một lưới 50 Hz: `t` bắt đầu từ 0, bước đúng 0,02 s, tăng đơn điệu, không có NaN/Inf | README gói; kiểm tra file |
| Mã luồng | `train_00`…`train_12`, `val_00`…`val_04`, sau này có `test_00`…`test_04` | README gói |
| Định nghĩa cú ngã | Bắt đầu từ lúc khởi phát chuyển động, kết thúc khi người đã nằm yên trên sàn; thời gian nằm sau đó không tính. Biên có sai số ±0,5–1 s. | README gói |
| Cờ chất lượng | `quality = low` (phần dư dưới ngưỡng 5,12e-6, chủ yếu ở Dataset 3) và `needs_review`. Các cú ngã này vẫn nằm trong luồng. Trên test, chúng là vùng `ignore`. | README gói; `evaluate.py` |
| Split | Train, val và test dùng những người khác nhau. Người trong bộ ngã và người trong bộ ADL không được ghép với nhau. | Trang challenge; README gói |
| File nộp | CSV gồm `stream_id, t_start, t_end, score`, tùy chọn thêm `t_alarm`. Yêu cầu `t_start < t_end`, cả hai nằm trong luồng, và giá trị hữu hạn. | README gói; `evaluate.py`; `sample_submission.csv` |
| Ghép dự đoán với nhãn | Ghép 1-1, tham lam theo `score` giảm dần. Là TP nếu tIoU ≥ θ. Dự đoán không ghép được mà chạm vùng `ignore` (nới rộng ±2 s) thì bị bỏ qua, không tính FP. | `evaluate.py` |
| Điểm xếp hạng | Trung bình F1 sự kiện tại θ = 0,1 / 0,3 / 0,5. Hòa điểm thì so trung vị \|sai số start\| tại θ = 0,3 (nhỏ hơn thắng), rồi đến số báo động giả mỗi giờ tại θ = 0,3 | README gói; `evaluate.py` |
| Chấm val giống test | `python -m abc2027.evaluate --gt streams_events.csv --pred <csv> --streams streams --ignore-flagged --out report.json` | README gói |
| Baseline tham khảo | Energy: f1_mean 0,315. Logistic với cửa sổ 4 s: f1_mean 0,258. Cả hai được tune ngay trên val, nên không phải ước lượng điểm test. | README gói |
| Giấy phép | CC BY-NC 4.0. Phải trích dẫn 4 bài báo nêu trong README. | LICENSE, README gói |
| Quy định dữ liệu | **Cấm** dùng bản ghi UWB nguồn của LIARA (bộ ngã và bộ ADL gốc, hoặc bản sao bên ngoài) cho cả train lẫn suy luận. Các đội đứng đầu phải nộp code và model để kiểm tra. | README gói |
| Quy định đội | Không giới hạn số thành viên. Chỉ supervisor được ở nhiều đội. Cấm chia sẻ code hoặc dữ liệu ra ngoài đội. Sinh viên thuộc lab tổ chức không được tham gia. | Trang Registration |
| Paper | Tiếng Anh, 8–12 trang kể cả tài liệu tham khảo, theo format ABC. Đăng trên IJABC (J-STAGE). Ít nhất một thành viên phải trình bày. | Trang Registration |
| Lịch | Tutorial và mở dữ liệu: 13/10/2026. Gửi test data: 20/10/2026. Đóng đăng ký: 27/10/2026. Nộp kết quả: 03/11/2026. Nộp paper: 10/11/2026. Trả review: 17/11/2026. Camera-ready: 08/12/2026. Hội nghị: 19–22/01/2027 (Sydney, hybrid). | autocare.ai/abc2027 |
| Hỏi đáp | `abc2027@autocare.ai`, tiêu đề "ABC Challenge" | Trang FAQ |

## Còn cần xác nhận với ban tổ chức

| Hạng mục | Lý do |
|---|---|
| Thư mục `segments/` trong gói (mảng nguồn đã căn chỉnh) có thuộc diện "source recordings" bị cấm không | Pipeline hiện **không** đọc thư mục này. |
| Kênh nộp kết quả và số lần được nộp | README gói ghi "to be announced". |
| Múi giờ của deadline challenge | Trang hội nghị chỉ ghi AoE cho các deadline của paper. |
| Link đăng ký đúng | Trang Registration có một Google Form và một nút trỏ tới CMT của ABC2025. |
| Có được dùng `streams_composition.csv` để chia fold khi train không | Nhóm chỉ dùng file này để chia CV theo người, không dùng làm đặc trưng. Khi suy luận trên test thì không cần đến nó. |

## Quyết định của nhóm

| Ngày | Quyết định | Lý do |
|---|---|---|
| 2026-10-22 | Đánh giá val bằng `evaluate.py` chính thức với `ignore_flagged = true` | Giống cách chấm test |
| 2026-10-22 | Chọn mô hình, ngưỡng và hậu xử lý chỉ dựa trên dự đoán out-of-fold của **train**; val chỉ dùng để báo cáo | Tránh tối ưu quá mức trên val, vì baseline chính thức lại tune trên val |
| 2026-10-22 | Chia fold CV trong train theo người (`subject_group`), không theo luồng | Một file ADL nguồn xuất hiện trong 2–13 luồng train |
| 2026-10-22 | Không đọc `segments/` | Quy định cấm dùng bản ghi nguồn |

Mỗi thay đổi protocol hoặc cách tính metric cần được báo cho cả nhóm, vì nó có thể làm các kết quả cũ không còn so sánh trực tiếp được.
