# Đặc tả UWB-FallTrack — bản cần xác nhận

Tài liệu này là nơi ghi lại những yêu cầu chính thức của challenge để cả nhóm dùng chung. Chỉ chuyển một mục từ “cần xác nhận” sang “đã xác nhận” khi có nguồn (hướng dẫn, trang challenge, dữ liệu mẫu hoặc trao đổi với ban tổ chức).

## Thông tin đã biết ở mức tổng quan

- Tên challenge: **UWB-FallTrack**.
- Chủ đề được mô tả công khai: **Continuous Fall Detection in UWB Radar Streams**.
- Thông tin trên là mô tả định hướng, chưa đủ để xác định định dạng tín hiệu, protocol hay metric.
- Nguồn tham khảo ban đầu: [thông báo tổng quan ABC 2027](https://www.informatika.unimal.ac.id/berita/abc-2027-challenges-peluang-kompetisi-computer-vision-dan-artificial-intelligence-tingkat-internasional) và [trang ABC Research](https://abc-research.github.io/).

## Cập nhật từ trang challenge chính thức (2026-10-20)

Đã đối chiếu các trang challenge UWB-FallTrack do ABC/UQAC công bố ngày 2026-10-20:

- [Tổng quan UWB-FallTrack](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/the-challenge) xác nhận bài toán dùng luồng UWB liên tục, không cắt sẵn, phát hiện cú ngã và ước lượng ranh giới thời gian; trang mô tả mục tiêu theo hướng bảo vệ quyền riêng tư.
- [Dataset và tutorial](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/dataset-and-tutorial) mô tả 3 radar Novelda XeThru X4M200 gắn tường, CIR 368 giá trị cùng timestamp và radar ID, tần số xấp xỉ 50 Hz mỗi radar, 13 luồng train (~6,2 giờ), 5 luồng validation (~2 giờ) và ground truth thời gian cho 478 cú ngã. Đây là thông tin công bố chính thức ở mức mô tả; tài liệu công khai chưa đủ để xác định chính xác kiểu dữ liệu, tên trường, đơn vị timestamp hoặc schema tệp.
- [Registration và evaluation](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/registration) liệt kê hit/miss/false alarm, sai số thời điểm bắt đầu/kết thúc, tIoU và detection latency. Trang chưa công bố ngưỡng tIoU, cách tổng hợp/xếp hạng hoặc một metric cuối cùng.
- [Trang ABC 2027](https://autocare.ai/abc2027/) liệt kê các mốc challenge, nhưng trang UWB vẫn ghi đăng ký đang mở trong khi trang lịch có mốc đóng đăng ký ngày 2026-10-13. Vì vậy trạng thái deadline hiện hành chưa được xác nhận nhất quán.
- Lịch trên trang ABC 2027 ghi dữ liệu test sẽ được gửi cho người tham gia vào ngày 2026-10-20. Các trang công khai chưa có liên kết tải test hoặc endpoint chấm kết quả; gói dữ liệu cục bộ hiện chỉ có train/validation. Đầu mối workspace xác nhận chưa nhận gói test qua kênh riêng ngày 2026-10-20. Đây là trạng thái nhóm tự ghi nhận, không xác nhận tình trạng phát hành cho các đội khác.

Các thông tin chi tiết hơn trong README và mã của gói dữ liệu cục bộ vẫn được ghi riêng là **tuyên bố của package**, không thay thế các điểm protocol còn thiếu từ ban tổ chức. Đặc biệt, metric mean event F1 tại các ngưỡng tIoU 0.1/0.3/0.5 và cờ `--ignore-flagged` hiện chỉ được package mô tả; chưa có xác nhận trên các trang challenge chính thức đã kiểm tra.

## Ghi nhận từ gói dữ liệu cục bộ (2026-10-10)

Máy khởi tạo workspace có gói `abc2027_uwb_train_v1` tại `data/uwb_falltrack/abc2027_uwb_train_v1/`. README và LICENSE đi kèm mô tả gói là dữ liệu train/validation UWB-FallTrack, gồm các luồng HDF5 với trường `cir`, thời gian `t` và serial radar; tài liệu cũng nêu một số chi tiết về nhãn, split và evaluator. Đây là **tuyên bố trong README/LICENSE của gói cục bộ**, chưa được đối chiếu độc lập với tài liệu challenge chính thức. Nhóm đã cung cấp [thư mục Google Drive](https://drive.google.com/drive/folders/182TmF7QxRepxTcQosjwLR5O0KWgSvlgv) để truy cập bản chia sẻ. Đây là nơi lưu trữ do nhóm cung cấp, không phải nguồn challenge chính thức; nội dung thư mục chưa được kiểm tra độc lập. Mỗi thành viên cần đọc README/LICENSE đi kèm và dùng bản dữ liệu được cấp phép. Đã kiểm tra README gói và LICENSE cục bộ ngày 2026-10-10.

Các con số protocol, metric và quy tắc submission từ README gói không được xem là xác nhận chính thức cho tới khi ban tổ chức xác nhận hoặc có tài liệu challenge chính thức kèm theo.

## Những điều cần xác nhận

| Hạng mục | Trạng thái | Giá trị / ghi chú | Nguồn và ngày kiểm tra |
|---|---|---|---|
| Phiên bản bộ dữ liệu và nguồn tải | Cần xác nhận | Trang chính thức mô tả nội dung bộ dữ liệu; URL tải công khai và điều khoản sử dụng chi tiết chưa được xác định | Dataset/tutorial, kiểm tra 2026-10-20 |
| Định dạng tín hiệu và cấu trúc tệp | Đã xác nhận một phần | CIR 368 giá trị, kèm timestamp và radar ID; tên trường, kiểu dữ liệu và schema tệp chưa rõ | Dataset/tutorial, kiểm tra 2026-10-20 |
| Đơn vị thời gian, tần số lấy mẫu, đồng bộ | Đã xác nhận một phần | Khoảng 50 Hz mỗi radar; đơn vị timestamp và chi tiết đồng bộ chưa rõ | Dataset/tutorial, kiểm tra 2026-10-20 |
| Định nghĩa lớp và cách gắn nhãn | Cần xác nhận |  |  |
| Cách chia train / validation / test | Đã xác nhận một phần | 13 luồng train (~6,2 giờ), 5 luồng validation (~2 giờ), ground truth 478 cú ngã. Lịch ABC ghi test dự kiến gửi ngày 2026-10-20; đầu mối workspace xác nhận chưa nhận, schema test chưa rõ | Dataset/tutorial và trang ABC 2027, kiểm tra 2026-10-20; xác nhận trạng thái nhóm ngày 2026-10-20 |
| Bộ test và evaluator công khai | Chưa xác nhận | Không thấy liên kết tải test hoặc endpoint evaluator công khai; gói cục bộ chỉ có train/validation. Đầu mối workspace xác nhận chưa nhận test riêng của đội | Trang ABC 2027, dataset/tutorial và registration, kiểm tra 2026-10-20; kiểm tra gói cục bộ và xác nhận trạng thái nhóm ngày 2026-10-20 |
| Metric chính thức và cách tính | Đã xác nhận một phần | Có hit/miss/false alarm, sai số start/end, tIoU và latency; chưa công bố ngưỡng/công thức xếp hạng | Registration/evaluation, kiểm tra 2026-10-20 |
| Định dạng tệp dự đoán cần nộp | Cần xác nhận | Trang công khai chỉ yêu cầu start/end và prediction cho mỗi sự kiện; schema tệp chưa rõ | Registration, kiểm tra 2026-10-20 |
| Baseline hoặc mã tham khảo | Cần xác nhận |  |  |
| Quy định thư viện, tài nguyên và suy luận | Cần xác nhận |  |  |
| Deadline và kênh nộp hiện hành | Cần xác nhận | Các trang chính thức đang có thông tin trạng thái đăng ký không nhất quán | Trang ABC 2027 và UWB overview, kiểm tra 2026-10-20 |

## Quyết định của nhóm

Ghi quyết định, lý do và ngày chốt tại đây. Mỗi thay đổi protocol hoặc cách tính metric cần được báo cho cả nhóm vì nó có thể làm các kết quả cũ không còn so sánh trực tiếp được.
