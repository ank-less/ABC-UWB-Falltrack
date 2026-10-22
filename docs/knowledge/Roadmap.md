---
type: project-roadmap
project: ABC2027 UWB-FallTrack
updated: 2026-10-21
tags:
  - abc2027
  - uwb
---

# Lộ trình

## Trạng thái hiện tại

- Many-body R31–R33 đã hoàn tất train-only nested CV; không promote. R32 đạt mean event F1 0,6305, R33 0,6153. R31 validation diagnostic bằng 0,0000; validation đã tiêu thụ nên R32/R33 không được chấm lại. Mục tiêu validation ≥0,82 chưa đạt/chưa được xác nhận.
- R32 đã khắc phục vị trí reset của fall trong synthetic repack, nhưng chỉ có 13 continuous bundle và 186 valid event. Source-path disjoint chưa chứng minh participant/session disjoint. R33 tăng history 0,4→1,2 s nhưng OOF giảm, false alarms tăng.
- R32 ablation chưa thấy quantum readout feature block giúp điểm: signal-only + QFI decoder đạt 0,6496, full 0,6305, quantum-only 0,1238. Mô phỏng dùng CIR cổ điển và chưa chứng minh quantum advantage.
- Không có tiến trình many-body nào đang chạy. Báo cáo nguyên nhân nằm ở `ABC testing/r32/failure_analysis.md`; chi tiết thí nghiệm tại [[Experiment Log]] và [[Decision Log]].
- Chờ holdout/test mới có participant/session mapping rõ và protocol metric được xác nhận trước khi tuyên bố generalization hoặc mục tiêu 0,82.

- Giai đoạn khảo sát dữ liệu, baseline và train-only OOF đã chạy. Mục tiêu nội bộ hiện là F1 ít nhất 0,75 và các mốc tIoU gần nhau; cách diễn giải tạm thời: F1@0.1/0.3/0.5 đều ≥0.75 và max−min ≤0.10.
- Nested OOF stream-level R16 ghi 0.8796/0.8482/0.7958, nhưng source-overlap audit sau đó phát hiện source ADL trùng/chồng giữa folds; không xem là bằng chứng zero-leakage hoặc generalization.
- R17 được chọn bằng 5-fold GroupKFold OOF trên 13 stream train; chưa phải điểm validation/test.
- Validation cục bộ đã được dùng cho baseline R2, R17, R23, R26 và diagnostic R28 consensus theo yêu cầu. Các diagnostic fit từ train và khóa prediction trước khi evaluator đọc nhãn; validation không còn độc lập. R28 candidate được chọn từ stream-level OOF bị source overlap. Gói test chưa có trong workspace; trạng thái nhận riêng chưa xác nhận. Xem [đặc tả challenge](../challenge_spec.md).
- Đã hoàn tất R18 reservoir và decoder semi-Markov; reservoir cải thiện nhỏ, semi-Markov kém R16 nên không promote. R19 tau sweep không cải thiện nhất quán so với baseline tau 8 trên seed 2028.
- R21 stream weighting tăng pooled F1 lên 0,8847/0,8633/0,7721, nhưng spread 0,1126 vượt mục tiêu tạm 0,10 và một outer-fold tIoU 0,5 xuống 0,7143. R22 thử boundary trim nhưng inner selection giữ nguyên profile cũ, kết quả trùng R21.
- R24 thử readout onset/offset riêng; cả năm outer fold chọn bán kính 0. Nested OOF đạt 0,8847/0,8686/0,7775 (spread 0,1072), nhưng outer-fold F1@0,5 thấp nhất 0,7143; chưa đáp ứng ổn định mục tiêu.
- R25 thử duration contraction, nhưng cả năm fold chọn scale 1,00 và kết quả trùng R24.
- R26 thêm reservoir và stream weighting nhưng đạt spread 0,1223; outer-fold minimum F1@0.5 là 0,72.
- Diagnostic R26 consensus trên validation đạt F1@0.1/0.3/0.5 = 0,8333/0,7576/0,5455; mean 0,7121, spread 0,2879. Chưa đạt mục tiêu và không dùng kết quả này để chỉnh mô hình.
- R27 dùng objective sàn F1 0,75/trần spread 0,10 trong inner OOF: pooled 0,8563/0,8394/0,7549 (spread 0,1014), nhưng outer-fold minimum F1@0.5 là 0,70.
- R28 đã kiểm tra offset start/end; metrics trùng R27 nên không promote.
- Diagnostic R28 consensus fit trên 13 train streams rồi áp lên 5 validation streams đạt 0,8271/0,7820/0,5564, mean 0,7218, spread 0,2707. Đây là điểm package-local không độc lập; candidate selection có thể bị ảnh hưởng bởi source overlap trong OOF.
- R29 stream-level nested OOF ghi 0,8635/0,8468/0,7521 (mean 0,8208, spread 0,1114), nhưng source overlap audit cho thấy đây không phải estimate leak-free.
- Train-only source audit phát hiện 74,9–78,4% held-out ADL source seconds overlap fit streams; toàn bộ 13 train streams nằm trong một overlap component. Stream-level OOF R16–R29 không chứng minh zero leakage.
- Baseline VQC R30 đã chạy source-group OOF trên 467 hash nguồn train, không dùng validation/test. Mean F1 0,1668 và 185,22 false alarms/hour; chưa đạt mục tiêu, chưa dùng QPU. Chi tiết ở [[Experiment Log]].
- Mục tiêu cân bằng tIoU chưa được xác nhận. Tạm dừng promote từ stream-level OOF; cần source-disjoint/purged protocol hợp lệ hoặc holdout/test mới. Không dùng thêm validation để chọn model/ngưỡng.

## Giai đoạn 1 — Xác nhận yêu cầu

- Đọc tài liệu chính thức của UWB-FallTrack.
- Xác nhận đăng ký, supervisor, quyền truy cập và điều khoản sử dụng dữ liệu.
- Ghi nguồn, split, metric, schema submission và deadline vào [đặc tả challenge](../challenge_spec.md).

## Giai đoạn 2 — Kiểm kê dữ liệu

- Lập danh mục tệp và ghi nhận schema, tần số lấy mẫu, kênh, nhãn, dữ liệu thiếu và số sự kiện.
- Vẽ các luồng đại diện cùng khoảng thời gian có nhãn.
- Kiểm tra mất cân bằng lớp và độ phủ theo người/phiên nếu protocol cho phép.

## Giai đoạn 3 — Dựng baseline

- **Đã có baseline và pipeline train-only OOF** trên package cục bộ; các chi tiết chưa xác nhận chính thức vẫn được ghi tại [đặc tả challenge](../challenge_spec.md).
- Tiền xử lý causal, cửa sổ 0,1 s, readout và decoder Ising/DP đã được ghi cấu hình để tái lập.
- Các vòng nested OOF R1–R22, semi-Markov và lựa chọn/diagnostic R17–R23 được ghi trong [[Experiment Log]].

## Giai đoạn 4 — Cải thiện và đánh giá

- Ứng viên R17 đã khóa bằng train-only OOF; không diễn giải OOF đã chọn là điểm generalization.
- Kết quả validation chẩn đoán R17/R23 là hậu kiểm, không độc lập hoàn toàn; không dùng để chọn mô hình/ngưỡng. Không chọn mô hình/ngưỡng từ nhãn test.
- R18 quantum reservoir tăng mean nested OOF nhẹ, tau sweep R19 không có lợi ích nhất quán, semi-Markov thấp hơn R16, R22 boundary trim không cải thiện, và R24 onset/offset heads đều bị inner selection loại bỏ.
- Target-floor ranking đã chạy ở R27–R29, nhưng stream-level folds có source overlap; cần sửa protocol trước khi tiếp tục lựa chọn candidate.
- Chưa promote cấu hình thay thế R17 đã khóa.
- Còn chờ holdout độc lập mới hoặc official evaluator để xác nhận kết quả; validation hiện có đã tiêu thụ.

## Giai đoạn 5 — Bàn giao

- Lệnh tạo dự đoán đã sẵn sàng qua `scripts/predict_holdout.py`, mặc định dùng cấu hình ứng viên R17 và chặn split validation cũ.
- Chưa tạo dự đoán submission vì test stream chưa có; schema/evaluator cuối cùng còn cần xác nhận.
- Chuẩn bị paper và deliverable theo yêu cầu.
- Lưu mã, cấu hình, kết quả và cách chạy để tái lập.
