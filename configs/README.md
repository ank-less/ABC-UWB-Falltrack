# Configurations

Lưu cấu hình thí nghiệm dạng YAML tại đây khi pipeline bắt đầu được xây dựng. Mỗi cấu hình nên ghi rõ split, bước tiền xử lý, tham số mô hình, seed và các đường dẫn tương đối. Không ghi đường dẫn tuyệt đối tới máy cá nhân.

`baseline.yaml` là cấu hình của baseline. Mỗi giá trị được chú thích là *confirmed* (đọc từ README/evaluator của gói hoặc từ chính file dữ liệu) hoặc *provisional* (lựa chọn của nhóm). Các giá trị bắt buộc như `data.rate_hz` hay `windowing.duration_seconds` không có mặc định: nếu bị để trống, pipeline dừng và báo cụ thể giá trị nào thiếu. Để thử nhanh một giá trị khác mà không sửa file, dùng `--override section.key=value`; cấu hình thực tế đã dùng được lưu vào `resolved_config.yaml` của mỗi run.
