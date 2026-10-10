# AGENTS.md

## Phạm vi dự án

Đây là workspace của nhóm ABC UWB Falltrack, hướng tới bài toán phát hiện ngã liên tục từ luồng radar UWB. README mô tả mục tiêu và workflow tổng thể; `GUIDELINES.md` giữ phần phân công công việc cho 5 thành viên. Giữ nội dung phân công chi tiết ở `GUIDELINES.md`, không chuyển ngược vào README nếu người dùng chưa yêu cầu.

## Nguồn sự thật về challenge

- Đọc `README.md`, `GUIDELINES.md` và `docs/challenge_spec.md` trước khi thay đổi workflow, dữ liệu hoặc cách đánh giá.
- Chủ đề tổng quát đã biết là phát hiện ngã liên tục trên luồng radar UWB. Không tự giả định định dạng tín hiệu, schema, nhãn, split, metric, giới hạn suy luận hay định dạng submission.
- Khi có tài liệu chính thức hoặc dữ liệu mới, cập nhật `docs/challenge_spec.md` kèm nguồn và ngày kiểm tra. Phân biệt rõ thông tin đã xác nhận với giả thuyết của nhóm.

## Quy ước dữ liệu và thí nghiệm

- Giữ dữ liệu nguồn nguyên trạng trong `data/raw/`; đặt dữ liệu trung gian và dữ liệu đã xử lý trong `data/interim/` và `data/processed/`.
- Không ghi đè dữ liệu gốc. Ghi lại các bước chuyển đổi để kết quả có thể tái tạo.
- Dùng đúng split và metric được tài liệu challenge quy định. Không lựa chọn mô hình hoặc ngưỡng dựa trên nhãn test.
- Mỗi thí nghiệm cần ghi cấu hình, seed (nếu có), split, cách tiền xử lý, metric và artifact đầu ra.
- Giữ trục thời gian và quy ước timestamp nhất quán trong các bước cửa sổ hóa, ghép dự đoán và tạo submission.
- Gói dữ liệu train/validation hiện có cục bộ dưới `data/uwb_falltrack/abc2027_uwb_train_v1/` và bị `.gitignore` loại khỏi repository. Không đưa dữ liệu, nhãn hoặc bản sao gói lên dịch vụ bên ngoài nếu chưa có yêu cầu rõ ràng; trước khi chia sẻ phải kiểm tra giấy phép, giới hạn dung lượng và quy định challenge.

## Quy ước mã nguồn và tệp

- Mã Python có thể tái sử dụng đặt trong `src/abc_uwb_falltrack/`; lệnh chạy pipeline đặt trong `scripts/`.
- Dùng `notebooks/` cho khám phá và thử nghiệm ban đầu. Chuyển logic dùng lại nhiều lần vào `src/` hoặc `scripts/`.
- Đặt cấu hình có thể chia sẻ trong `configs/`; dùng đường dẫn tương đối, không ghi đường dẫn riêng của máy một thành viên.
- Khai báo thư viện dự án trong `pyproject.toml`. Chỉ thêm thư viện khi có lý do kỹ thuật cụ thể và ghi cách sử dụng khi cần.
- Không đưa checkpoint, dự đoán sinh ra hoặc cache vào mã nguồn; các thư mục artifact tương ứng được quản lý qua `.gitignore`.
- Viết tài liệu dự án bằng tiếng Việt, giữ nguyên các thuật ngữ kỹ thuật tiếng Anh khi cách dịch có thể gây mơ hồ.

## Git và thao tác bên ngoài

- Làm việc cục bộ trong repo hiện tại. Không tạo remote, tạo repository GitHub, commit, push, hoặc gửi dữ liệu lên dịch vụ bên ngoài cho tới khi người dùng yêu cầu rõ ràng.
- Không đưa dữ liệu vào commit một cách tự động. Khi người dùng yêu cầu đưa dữ liệu lên GitHub, trước hết kiểm tra những tệp nào sẽ được thêm và báo cáo các giới hạn liên quan đến kích thước hoặc định dạng.
- Không tự chạy hoặc thêm kiểm thử nếu yêu cầu của người dùng không bao gồm kiểm thử/xác minh.

## Cách thực hiện thay đổi

- Giữ thay đổi tập trung vào yêu cầu hiện tại; tránh tự triển khai mô hình trước khi định dạng dữ liệu và protocol được xác nhận.
- Khi sửa README, giữ mô tả bài toán và workflow tổng thể ở đó; đặt phân công vai trò và nhiệm vụ từng người trong `GUIDELINES.md`.
- Sau khi sửa, tóm tắt tệp đã đổi và trạng thái Git. Nêu rõ nếu chưa chạy kiểm thử hoặc chưa thực hiện thao tác GitHub.
