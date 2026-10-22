# Bộ dữ liệu cuộc thi

**Thư mục Google Drive:** [Mở thư mục dữ liệu trên Google Drive](https://drive.google.com/drive/folders/182TmF7QxRepxTcQosjwLR5O0KWgSvlgv).

Quyền truy cập phụ thuộc vào cài đặt chia sẻ của chủ sở hữu Drive. Nếu link yêu cầu quyền, hãy liên hệ người quản lý thư mục để được cấp quyền cho tài khoản của bạn.

Tải dữ liệu được cấp phép về và giữ nguyên bản gốc trong `data/raw/`. Các tệp dữ liệu dung lượng lớn không được commit vào Git; repository chỉ lưu hướng dẫn này.

## Tải từ link chính thức của ban tổ chức

Trang "Dataset and Tutorial" của challenge có link Google Drive tới `abc2027_uwb_train_v1.zip` (8,17 GB). Script dưới đây chỉ tải những file cần thiết từ zip đó (dùng HTTP range request, không phải tải cả file zip) và kiểm tra CRC của từng file:

```bash
python scripts/fetch_data.py --subset all     # 18 luồng + CSV + code của gói: khoảng 4,6 GB tải về, 6,8 GB trên đĩa
python scripts/fetch_data.py --subset smoke   # train_00, train_01, val_00, val_01: khoảng 1,1 GB
```

Dữ liệu được đặt vào `data/uwb_falltrack/abc2027_uwb_train_v1/`; đây là đường dẫn mà `configs/baseline.yaml` dùng. Thư mục `segments/` (bản ghi nguồn LIARA đã căn chỉnh) không được tải, vì README của gói cấm dùng bản ghi nguồn để train hoặc suy luận.

## Quyền sử dụng

Thư mục Drive là nơi nhóm chia sẻ dữ liệu, không phải xác nhận độc lập về nguồn gốc hay protocol chính thức của challenge. Trước khi sử dụng hoặc chia sẻ lại, hãy đọc README và LICENSE đi kèm dữ liệu; các tài liệu trong gói nêu giấy phép CC BY-NC 4.0 và các hạn chế liên quan đến bản ghi nguồn LIARA.
