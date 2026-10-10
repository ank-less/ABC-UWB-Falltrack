# GUIDELINES — Phân công nhóm ABC UWB Falltrack

Tài liệu này đề xuất cách chia công việc cho nhóm 5 người. Nhóm có thể thay tên vai trò theo thế mạnh và lịch của từng thành viên. Mỗi mảng nên có một người chịu trách nhiệm chính, nhưng các quyết định ảnh hưởng tới protocol, dữ liệu, metric và kết quả cuối cần được chia sẻ để cả nhóm cùng nắm.

## Phân công cho 5 thành viên

### Người 1 — Điều phối và đặc tả challenge

**Mục tiêu:** giúp nhóm hiểu đúng yêu cầu của ban tổ chức và làm việc theo cùng một protocol.

**Công việc:**

- Đọc tài liệu chính thức, hướng dẫn tải dữ liệu, baseline và quy định nộp bài.
- Ghi lại định dạng dữ liệu, nhãn, cách chia tập, metric, định dạng kết quả và deadline trong `docs/challenge_spec.md`.
- Đánh dấu rõ thông tin nào đã được xác nhận, thông tin nào còn là câu hỏi mở; ghi nguồn và ngày kiểm tra.
- Chia các mốc công việc theo deadline chính thức và nhắc nhóm cập nhật tiến độ.
- Trước khi nộp, cùng nhóm rà soát checklist về kết quả, tệp dự đoán và paper.

**Bàn giao:** đặc tả challenge có nguồn, danh sách câu hỏi còn mở, lịch mốc và checklist nộp bài.

### Người 2 — Dữ liệu và tín hiệu UWB

**Mục tiêu:** giúp nhóm hiểu dữ liệu và xây đầu vào nhất quán cho các mô hình.

**Công việc:**

- Kiểm kê tệp, schema, nhãn, độ dài bản ghi, thứ tự thời gian và các giá trị thiếu hoặc bất thường.
- Trực quan hóa một số tín hiệu đại diện để nhóm thấy tín hiệu của các hoạt động khác nhau trông như thế nào.
- Kiểm tra metadata về người tham gia, phiên thu thập hoặc môi trường để phát hiện nguy cơ rò rỉ giữa các tập.
- Đề xuất cách nạp dữ liệu, chuẩn hóa và chia cửa sổ sau khi protocol đã rõ.
- Giữ dữ liệu gốc nguyên trạng trong `data/raw/`; đặt các bước chuyển đổi vào pipeline có thể chạy lại.

**Bàn giao:** báo cáo kiểm tra dữ liệu, notebook khám phá và module nạp/tiền xử lý có hướng dẫn sử dụng.

### Người 3 — Baseline và phát triển mô hình

**Mục tiêu:** tạo các mô hình dự đoán có thể so sánh công bằng trên cùng dữ liệu và cùng cách đánh giá.

**Công việc:**

- Phối hợp với Người 2 để thống nhất giao diện dữ liệu đầu vào.
- Dựng baseline đơn giản hoặc triển khai baseline do ban tổ chức cung cấp.
- Sau khi baseline và split ổn định, thử các lựa chọn mô hình hoặc biểu diễn tín hiệu có giả thuyết rõ ràng.
- Lưu cấu hình, seed, trọng số và dự đoán validation theo quy ước của nhóm.
- Trao đổi với Người 4 để chọn thử nghiệm tiếp theo dựa trên nhóm lỗi, không chỉ dựa trên một con số tổng.

**Bàn giao:** mã huấn luyện và suy luận, cấu hình, baseline có thể chạy lại và các mô hình ứng viên được ghi nhận.

### Người 4 — Đánh giá và phân tích lỗi

**Mục tiêu:** đảm bảo các kết quả phản ánh đúng protocol và giúp nhóm hiểu mô hình hoạt động tốt hoặc kém ở đâu.

**Công việc:**

- Đối chiếu cách tính metric với tài liệu chính thức và kiểm tra bằng các ví dụ nhỏ.
- Xây module đánh giá theo split và metric đã xác nhận.
- Phân tích sự kiện bị bỏ sót, hoạt động thường ngày bị nhận nhầm và độ trễ phát hiện nếu dữ liệu cho phép.
- Kiểm tra rò rỉ giữa train và validation, cũng như ảnh hưởng của từng thay đổi trong các thí nghiệm.
- Tóm tắt kết quả thành bảng và biểu đồ để Người 3, Người 5 và cả nhóm dùng chung.

**Bàn giao:** mã đánh giá, bảng/biểu đồ validation, phân tích lỗi và khuyến nghị cho vòng thử nghiệm kế tiếp.

### Người 5 — Tích hợp, tái lập và báo cáo

**Mục tiêu:** nối các phần việc thành một pipeline hoàn chỉnh mà thành viên khác có thể chạy lại.

**Công việc:**

- Thống nhất lệnh chạy, cấu hình và giao diện giữa nạp dữ liệu, huấn luyện, đánh giá và suy luận.
- Tích hợp các module sau khi giao diện giữa chúng đã được thống nhất; phát hiện sớm lỗi phụ thuộc vào máy cá nhân.
- Ghi lại môi trường, phiên bản cấu hình, đường dẫn artifact và các bước tạo kết quả cuối.
- Tổng hợp kết quả và biểu đồ đã được lưu để hỗ trợ paper.
- Soạn phần mô tả phương pháp, tổ chức review nội bộ và nhờ một thành viên khác kiểm tra lệnh chạy/tệp nộp.

**Bàn giao:** pipeline end-to-end, hướng dẫn chạy, bản tổng hợp thí nghiệm, bản thảo và checklist bàn giao.

## Cách phối hợp và bàn giao

1. **Người 1 → cả nhóm:** chốt protocol và chia sẻ các yêu cầu đã xác nhận trước khi nhóm khóa cách chia tập hoặc metric.
2. **Người 2 → Người 3 và Người 4:** bàn giao mô tả dữ liệu, quy ước nạp dữ liệu và split trước khi bắt đầu so sánh mô hình.
3. **Người 3 ↔ Người 4:** Người 3 cung cấp dự đoán và cấu hình; Người 4 đánh giá độc lập, phân tích lỗi rồi gửi khuyến nghị cho vòng tiếp theo.
4. **Người 5 ↔ các thành viên:** tích hợp từng module sớm, ghi lại cách chạy và báo lỗi giao diện để người phụ trách sửa ngay.
5. **Cả nhóm:** review mô hình cuối, metric validation, tệp submission và paper. Một thành viên không trực tiếp tạo tệp cuối nên kiểm tra lại lệnh chạy và định dạng trước khi nộp.

## Quy ước chung

- Mỗi thí nghiệm ghi rõ split, phiên bản tiền xử lý, cấu hình, seed, metric và artifact đầu ra.
- Thay đổi dữ liệu hoặc cách tính metric cần được ghi lại để kết quả cũ còn được diễn giải đúng.
- Câu hỏi chưa có đáp án trong tài liệu chính thức được ghi trong `docs/challenge_spec.md`, không biến suy đoán thành yêu cầu đã xác nhận.
- Lịch nội bộ sẽ được chốt sau khi nhóm kiểm tra lại deadline hiện hành của challenge.
