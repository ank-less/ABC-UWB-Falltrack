---
type: project-goals
project: ABC2027 UWB-FallTrack
updated: 2026-10-21
tags:
  - abc2027
  - uwb
---

# Mục tiêu

## Mục tiêu chính

Phát triển và nộp một hệ thống phát hiện ngã liên tục từ luồng thời gian radar UWB, tuân thủ protocol chính thức của UWB-FallTrack.

## Điều kiện thành công

- Dùng đúng split và quy trình đánh giá do ban tổ chức quy định.
- Thiết lập baseline minh bạch trước khi tinh chỉnh mô hình phức tạp.
- Báo cáo metric và định dạng đầu ra được yêu cầu.
- Ghi lại tiền xử lý, huấn luyện, validation và giới hạn để người khác có thể tái lập.

## Trạng thái hiện tại — 2026-10-21

- Many-body R31–R33 đã hoàn tất train-only; không promote. R32 đạt nested source-group OOF mean event F1 **0,6305**, R33 **0,6153**. R31 có validation diagnostic 0,0000 nhưng validation đã tiêu thụ; R32/R33 không được chấm validation. Mục tiêu validation ≥0,82 chưa đạt/chưa được xác nhận.
- R31–R33 source-path grouping chưa chứng minh participant/session disjoint. R32/R33 dùng synthetic continuous bundles; 13 bundle, 186 valid event và 173 flagged. R33 thêm history 1,2 s nhưng OOF giảm và false alarms tăng. Không còn tiến trình many-body chạy.
- R32 ablation signal-only readout + QFI decoder đạt 0,6496; quantum-only readout + QFI decoder 0,1238; full pipeline 0,6305. Chưa có bằng chứng quantum feature block cải thiện prediction trong cấu hình hiện tại.
- Các bằng chứng và giới hạn được ghi trong [[Experiment Log]], [[Decision Log]] và báo cáo local `ABC testing/r32/failure_analysis.md`.
- Cần holdout/test mới, participant/session mapping và protocol evaluator chính thức để xác nhận generalization. Validation cũ đã dùng nhiều lần, không dùng lại để chọn model/ngưỡng.

- Mục tiêu many-body theo yêu cầu người dùng là **mean event F1 validation ≥0,82**. R31–R33 đã được triển khai và ghi rõ kết quả không đạt; hướng tới mục tiêu không đồng nghĩa với kết quả đã xác nhận. Chi tiết tại [[Decision Log]].

- Đã có baseline và nhiều vòng train-only GroupKFold OOF. Candidate stream-balanced R23 được chọn bằng OOF trên 13 stream train, fit lại trên toàn bộ train và chấm một lượt chẩn đoán trên validation; kết quả không đạt mục tiêu nên chưa promote.
- Mục tiêu nội bộ mới theo yêu cầu người dùng là F1 ít nhất 0.75 và kết quả giữa các mốc tIoU không chênh quá lớn. Cách diễn giải tạm để phát triển là F1@0.1/0.3/0.5 đều ≥0.75 và max−min ≤0.10; đây không phải protocol chính thức.
- Điểm R17 được dùng để chọn cấu hình từ bốn biến thể train OOF; đây không phải phép đo độc lập và không chứng minh mục tiêu trên validation/test.
- Năm stream validation đã được dùng cho baseline R2, R17 và các lượt diagnostic R23, R26, R28 theo yêu cầu. Nhãn chỉ được evaluator dùng sau khi dự đoán đã khóa; các kết quả không được dùng để chỉnh/chọn mô hình. Vì vậy validation không còn là holdout độc lập.
- Gói cục bộ hiện chỉ có train/validation, chưa có test stream. Việc nhận test riêng và protocol nộp/chấm chính thức vẫn cần đối chiếu trong [đặc tả challenge](../challenge_spec.md).
- R17 đạt mean event F1 0.6957 trên lượt validation chẩn đoán theo package evaluator, ngay dưới mục tiêu nội bộ 0.70. Do split này đã được dùng cho R2, kết quả chỉ mang tính tham khảo, không xác nhận generalization độc lập. Mục tiêu và metric vẫn chưa được xác nhận là tiêu chí xếp hạng chính thức. Chưa tạo submission vì chưa có test stream mới.
- Stream-level nested OOF R16 ghi F1@0.1/0.3/0.5 = 0.8796/0.8482/0.7958, nhưng source-overlap audit sau đó cho thấy split theo `stream_id` tái sử dụng source ADL giữa fit và held-out. Không xem R16 hay OOF R17–R29 là bằng chứng zero-leakage/generalization. R17 validation diagnostic vẫn thấp hơn mục tiêu 0.75; không mở lại validation để chọn cấu hình.
- Nested OOF R19/R20 trên split seed mới đạt lần lượt 0.8640/0.8427/0.7520 (spread 0.1120) và 0.8617/0.8404/0.7553 (spread 0.1064); tau 4/8/16 không cho lợi ích ổn định so với baseline tau 8.
- Nested OOF R21 với stream weighting đạt 0.8847/0.8633/0.7721, mean 0.8400, spread 0.1126. Cả ba pooled threshold vượt 0.75, nhưng spread vượt tiêu chí tạm 0.10 và outer-fold tIoU 0.5 min là 0.7143; chưa xác nhận mục tiêu ổn định, chưa promote.
- Decoder semi-Markov nested OOF train-only đạt 0.7532/0.7013/0.5714, mean 0.6753; thấp hơn R16 ở cả ba ngưỡng nên không promote.
- Validation diagnostic R17 đạt 0.8116/0.7536/0.5217 (mean 0.6957); diagnostic R23 đạt 0.8143/0.7143/0.5286 (mean 0.6857). R23 không cải thiện mean và cả hai lượt đều dưới mục tiêu; không dùng lại validation đã tiêu thụ để chọn model.
- Nested OOF R22 thử adaptive boundary trim nhưng inner selection giữ profile cũ ở cả năm fold; metric trùng R21, nên không cải thiện.
- R24 nested OOF đạt 0.8847/0.8686/0.7775 (mean 0.8436, spread 0.1072); pooled F1 cả ba ngưỡng vượt 0.75 nhưng spread còn cao hơn ngưỡng tạm 0.10 và outer-fold minimum F1@0.5 là 0.7143. Cả năm fold chọn boundary radius 0, nên không promote onset/offset heads.
- R25 thử contraction hệ số 0,80–1,00 nhưng cả năm fold chọn 1,00, nên không cải thiện R24.
- R26 thêm quantum reservoir và stream weighting: 0.8989/0.8723/0.7766 (mean 0.8493, spread 0.1223); outer-fold minimum F1@0.5 là 0.72, nên chưa đạt tiêu chí ổn định.
- Theo yêu cầu, đã fit cấu hình đồng thuận R26 trên toàn bộ train rồi chấm validation một lượt: mean F1 0.7121; F1@0.1/0.3/0.5 = 0.8333/0.7576/0.5455, spread 0.2879. Nhãn validation không tham gia fit/chọn; do validation đã dùng ở các lượt trước, đây chỉ là diagnostic, không độc lập và không dùng để tinh chỉnh.
- R27 với inner-selection target-floor đạt pooled OOF F1@0.1/0.3/0.5 = 0.8563/0.8394/0.7549, spread 0.1014; outer-fold minimum F1@0.5 là 0.70. Chưa đạt tiêu chí ổn định. R28 đang thử offset biên trong inner OOF, không dùng validation.
- R28 đổi profile offset biên nhưng pooled stream-level OOF trùng R27. Consensus R28 được fit trên train rồi chấm validation: mean F1 0.7218; F1@0.1/0.3/0.5 = 0.8271/0.7820/0.5564, spread 0.2707. Đây là package-local diagnostic đã tiêu thụ; lựa chọn model dựa trên OOF source-overlap-contaminated.
- R29 stream-level OOF ghi 0.8635/0.8468/0.7521 (mean 0.8208, spread 0.1114); source audit cho thấy đây không phải estimate leak-free. Không promote và không dùng để kết luận đạt mục tiêu.
- Train-only source audit cho thấy outer held-out folds có 74.9–78.4% ADL source-time overlap với fit streams; cả 13 stream train nằm trong một component. Do đó các OOF R16–R29 theo `stream_id` không chứng minh zero leakage. Cần protocol source-disjoint/purged hợp lệ hoặc holdout/test mới trước khi tiếp tục model selection.
- Baseline VQC 4 qubit R30 đã chạy 5-fold OOF theo 467 hash nguồn train, không dùng validation/test; mean F1 **0,1668**, F1@0.1/0.3/0.5 **0,1926/0,1729/0,1349**, false alarms/hour **185,22**. Chưa đạt sàn 0,75; phép đo source-disjoint chỉ là package-local và VQC chạy bằng statevector NumPy, không dùng QPU. Xem [[Experiment Log]].
- Cần test stream hoặc holdout độc lập mới để xác nhận generalization; validation cũ đã tiêu thụ.
- Các artifact/config và kết quả R16–R29 được ghi trong [[Experiment Log]]. Chưa có test stream hoặc holdout độc lập mới; mục tiêu generalization vẫn chưa được xác nhận.

Chi tiết R16/R17, cấu hình và artifact nằm trong [[Experiment Log]] và [[Decision Log]].
