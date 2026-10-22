---
type: project-constraints
project: ABC2027 UWB-FallTrack
updated: 2026-10-19
tags:
  - abc2027
  - uwb
---

# Ràng buộc

## Đã biết

- Chủ đề tổng quát là phát hiện ngã liên tục từ luồng radar UWB.
- Dữ liệu gốc phải được giữ nguyên; dữ liệu trung gian và đã xử lý ở đúng thư mục theo [quy tắc agent](../../AGENTS.md).

## Phụ thuộc protocol và cần xác nhận

- Quyền truy cập, giấy phép và cách sử dụng dữ liệu.
- Kênh cảm biến, tần số lấy mẫu, độ chính xác timestamp và giới hạn tiền xử lý.
- Quy tắc split theo người/phiên và chống rò rỉ dữ liệu.
- Quy định dữ liệu ngoài, pretrained model và augmentation.
- Giới hạn suy luận, tài nguyên tính toán và định dạng đầu ra.
- Deliverable và deadline.

Các chi tiết challenge chỉ có một nguồn chuẩn tại [đặc tả challenge](../challenge_spec.md). Trước khi được xác nhận, giữ nguyên tín hiệu gốc và thứ tự thời gian; không suy ra luật từ dataset không liên quan.
