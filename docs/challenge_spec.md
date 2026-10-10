# Đặc tả UWB-FallTrack — bản cần xác nhận

Tài liệu này là nơi ghi lại những yêu cầu chính thức của challenge để cả nhóm dùng chung. Chỉ chuyển một mục từ “cần xác nhận” sang “đã xác nhận” khi có nguồn (hướng dẫn, trang challenge, dữ liệu mẫu hoặc trao đổi với ban tổ chức).

## Thông tin đã biết ở mức tổng quan

- Tên challenge: **UWB-FallTrack**.
- Chủ đề được mô tả công khai: **Continuous Fall Detection in UWB Radar Streams**.
- Thông tin trên là mô tả định hướng, chưa đủ để xác định định dạng tín hiệu, protocol hay metric.
- Nguồn tham khảo ban đầu: [thông báo tổng quan ABC 2027](https://www.informatika.unimal.ac.id/berita/abc-2027-challenges-peluang-kompetisi-computer-vision-dan-artificial-intelligence-tingkat-internasional) và [trang ABC Research](https://abc-research.github.io/).

## Ghi nhận từ gói dữ liệu cục bộ (2026-10-10)

Máy khởi tạo workspace có gói `abc2027_uwb_train_v1` tại `data/uwb_falltrack/abc2027_uwb_train_v1/`. README và LICENSE đi kèm mô tả gói là dữ liệu train/validation UWB-FallTrack, gồm các luồng HDF5 với trường `cir`, thời gian `t` và serial radar; tài liệu cũng nêu một số chi tiết về nhãn, split và evaluator. Đây là **tuyên bố trong README/LICENSE của gói cục bộ**, chưa được đối chiếu độc lập với tài liệu challenge chính thức. Gói dữ liệu và tài liệu kèm theo nằm trong `data/` và không được đưa vào repository này; mỗi thành viên cần dùng bản dữ liệu được cấp phép. Đã kiểm tra README gói và LICENSE cục bộ ngày 2026-10-10.

Các con số protocol, metric và quy tắc submission từ README gói không được xem là xác nhận chính thức cho tới khi ban tổ chức xác nhận hoặc có tài liệu challenge chính thức kèm theo.

## Những điều cần xác nhận

| Hạng mục | Trạng thái | Giá trị / ghi chú | Nguồn và ngày kiểm tra |
|---|---|---|---|
| Phiên bản bộ dữ liệu và nguồn tải | Cần xác nhận |  |  |
| Định dạng tín hiệu và cấu trúc tệp | Cần xác nhận |  |  |
| Đơn vị thời gian, tần số lấy mẫu, đồng bộ | Cần xác nhận |  |  |
| Định nghĩa lớp và cách gắn nhãn | Cần xác nhận |  |  |
| Cách chia train / validation / test | Cần xác nhận |  |  |
| Metric chính thức và cách tính | Cần xác nhận |  |  |
| Định dạng tệp dự đoán cần nộp | Cần xác nhận |  |  |
| Baseline hoặc mã tham khảo | Cần xác nhận |  |  |
| Quy định thư viện, tài nguyên và suy luận | Cần xác nhận |  |  |
| Deadline và kênh nộp hiện hành | Cần xác nhận |  |  |

## Quyết định của nhóm

Ghi quyết định, lý do và ngày chốt tại đây. Mỗi thay đổi protocol hoặc cách tính metric cần được báo cho cả nhóm vì nó có thể làm các kết quả cũ không còn so sánh trực tiếp được.
