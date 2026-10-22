---
type: experiment-log
project: ABC2027 UWB-FallTrack
updated: 2026-10-20
tags:
  - abc2027
  - uwb
---

# Nhật ký thí nghiệm

Ghi kết quả theo [[Templates/Experiment Template]], gồm split, tiền xử lý, seed, metric và artifact. Protocol challenge chỉ có tại [đặc tả challenge](../challenge_spec.md); các kết quả dưới đây dùng evaluator cục bộ của package và được ghi nhãn tương ứng.

---

## Experiment — 2026-10-20 — quantum-diff-logistic-oof-r1

### Câu hỏi / giả thuyết

Đặc trưng chuyển động theo thời gian kết hợp observables của density matrix 4 qubit có tạo được baseline event-level vượt detector năng lượng đơn giản không?

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; chỉ 13 stream train, 5-fold GroupKFold theo `stream_id`. Validation chưa được mở. Hàng train có `quality=low`, `needs_review` hoặc `ignore` được mask cùng vùng đệm evaluator. Metric dùng evaluator nằm trong package; xem trạng thái protocol tại [đặc tả challenge](../challenge_spec.md).

### Tiền xử lý và cửa sổ hóa

Sai phân nhân quả giữa các frame CIR liên tiếp để triệt nền tĩnh; RMS theo 5 frame (10 Hz), chia 368 range bin thành 16 nhóm. Không dùng `streams_composition.csv` trong trích xuất đặc trưng hoặc dự đoán.

### Mô hình và cấu hình

Amplitude encoding 4 qubit trên profile năng lượng chuyển động của từng radar; density matrix hỗn hợp ba radar, entropy, purity, entanglement entropy, temporal overlap và Pauli-Z observables. Readout logistic regression, `class_weight=balanced`, `C=1.0`, `max_iter=250`. Decoder là bộ lọc DP Ising nhân quả; dò bias, transition cost và min duration trên OOF.

### Seed ngẫu nhiên

2027 cho thứ tự group fold và readout.

### Metric theo protocol

Mean event F1 tại tIoU 0.1/0.3/0.5 theo evaluator package với semantics `--ignore-flagged`; đây là metric package-local, chưa được xác nhận là công thức xếp hạng chính thức.

### Kết quả

OOF mean event F1 = **0.4964**; F1@0.1/0.3/0.5 = 0.5714/0.5286/0.3893; median absolute start error tại 0.3 = 0.20 s; false alarms/hour tại 0.3 = 36.56. Readout chạm giới hạn 250 vòng lặp ở cả 5 fold.

### Phân tích lỗi

OOF có 226 FP và 38 FN tại tIoU 0.3; nhiều dự đoán rơi vào ADL, `Autre/Empty` và vùng trước/sau event.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_oof.yaml](../../configs/quantum_oof.yaml)
- Mã: [quantum_features.py](../../src/abc_uwb_falltrack/quantum_features.py), [oof.py](../../src/abc_uwb_falltrack/oof.py), [run_oof.py](../../scripts/run_oof.py)
- Artifact cục bộ bị ignore: `data/processed/quantum_diff_oof_20261020_r1/`
- Lệnh: `./.venv/bin/python -u scripts/run_oof.py --config configs/quantum_oof.yaml`

### Bước tiếp theo

Thử readout phi tuyến, chỉ dựa trên OOF train.

---

## Experiment — 2026-10-20 — quantum-diff-histgb-oof-r2

### Câu hỏi / giả thuyết

Readout phi tuyến có thể phân biệt động học fall với ADL/`Autre/Empty` tốt hơn logistic trên cùng biểu diễn quantum-inspired không?

### Phiên bản dữ liệu và split

Giống r1: 13 stream train, 5-fold GroupKFold theo `stream_id`; validation không dùng. Gắn cờ dữ liệu và evaluator theo mô tả package, chưa xem là protocol xếp hạng đã xác nhận.

### Tiền xử lý và cửa sổ hóa

Giữ nguyên đặc trưng/cache của r1. Trong Error Audit train, đọc các hàng train trong `streams_composition.csv` để gán false alarm vào loại segment; file này không tham gia feature extraction/prediction.

### Mô hình và cấu hình

Giữ nguyên state preparation và quantum observables. Readout `HistGradientBoostingClassifier`: learning rate 0.08, 120 vòng, 15 leaf nodes, min leaf 100, L2=2.0, max bins 63, positive weight 5.0. Decoder chọn bằng OOF: bias 0.3246787, transition cost 1.0, min duration 0.8 s.

### Seed ngẫu nhiên

2027.

### Metric theo protocol

Evaluator package với `--ignore-flagged`, tính trên prediction OOF train; metric package-local. Xem [đặc tả challenge](../challenge_spec.md).

### Kết quả

OOF mean event F1 = **0.7322**; F1@0.1/0.3/0.5 = 0.7909/0.7557/0.6499; median absolute start error tại 0.3 = 0.18 s; false alarms/hour tại 0.3 = 9.87. Mốc nội bộ 0.70 đã đạt trên OOF, chưa phải trên validation.

### Phân tích lỗi

Tại tIoU 0.3 có 61 FP và 36 FN. Error Audit train gán 34 FP vào segment fall, 19 vào `Autre/Empty` và 8 vào ADL; trong 34 FP ở segment fall, 13 chạm event nhưng IoU chưa đạt, 11 nằm trước và 10 nằm sau event. Kết quả gợi ý boundary/context vẫn là nguồn lỗi chính.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_oof.yaml](../../configs/quantum_histgb_oof.yaml)
- Artifact OOF và audit train cục bộ bị ignore: `data/processed/quantum_diff_histgb_oof_20261020_r2/`
- Lệnh OOF: `./.venv/bin/python -u scripts/run_oof.py --config configs/quantum_histgb_oof.yaml`
- Lệnh Error Audit train: `./.venv/bin/python scripts/audit_oof.py --artifact-dir data/processed/quantum_diff_histgb_oof_20261020_r2`

### Bước tiếp theo

Đóng băng readout/decoder từ OOF, rồi chạy một lần final holdout theo evaluator package.

---

## Experiment — 2026-10-20 — quantum-diff-histgb-final-holdout

### Câu hỏi / giả thuyết

Mô hình đã khóa từ OOF có đạt mốc nội bộ mean event F1 0.70 trên 5 stream validation của package không?

### Phiên bản dữ liệu và split

Fit trên toàn bộ 13 stream train. Dự đoán trên 5 stream validation; nhãn validation chỉ được đọc ở lần chạy cuối của evaluator package. Không dùng điểm này để tinh chỉnh. Đây là package-local holdout, chưa được xác nhận là điểm challenge chính thức.

### Tiền xử lý và cửa sổ hóa

Sai phân CIR nhân quả, RMS theo block 5 frame, cùng quantum-inspired density-matrix features như r2. Không đọc `streams_composition.csv` khi predict.

### Mô hình và cấu hình

HistGradientBoosting readout với cùng tham số r2; decoder khóa từ OOF: bias 0.3246787, transition cost 1.0, min duration 0.8 s. Seed 2027.

### Seed ngẫu nhiên

2027.

### Metric theo protocol

Chạy evaluator package với `--ignore-flagged` một lần trên 5 stream validation; công thức chỉ được package mô tả, chưa xác nhận là ranking chính thức.

### Kết quả

Mean event F1 = **0.6306**; F1@0.1/0.3/0.5 = 0.7297/0.6757/0.4865. Tại tIoU 0.3: TP=50, FP=31, FN=17; median absolute start error=0.48 s; false alarms/hour=15.13. Mốc validation nội bộ 0.70 **chưa đạt**. Kết quả này được giữ làm holdout cuối; không tinh chỉnh tiếp dựa trên nó.

### Phân tích lỗi

Khoảng cách OOF-to-holdout cho thấy khả năng tổng quát hóa sang 5 stream holdout còn yếu, đặc biệt ở tIoU 0.5. Không phân tích lỗi theo từng nhãn validation và không thay đổi mô hình sau lần chấm này.

### Artifact và lệnh tái lập

- Cấu hình khóa: [quantum_histgb_final.yaml](../../configs/quantum_histgb_final.yaml)
- Mã dự đoán: [predict_holdout.py](../../scripts/predict_holdout.py)
- Dự đoán, checkpoint cục bộ và report evaluator bị ignore: `data/processed/quantum_diff_histgb_final_20261020/`
- Dự đoán: `./.venv/bin/python -u scripts/predict_holdout.py --config configs/quantum_histgb_final.yaml`
- Đánh giá: `PYTHONPATH=data/uwb_falltrack/abc2027_uwb_train_v1/src .venv/bin/python -m abc2027.evaluate --gt data/uwb_falltrack/abc2027_uwb_train_v1/streams_events.csv --pred data/processed/quantum_diff_histgb_final_20261020/validation_predictions.csv --streams data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --ignore-flagged --out data/processed/quantum_diff_histgb_final_20261020/holdout_metrics_package_local.json`

### Bước tiếp theo

Không mở lại validation này để tinh chỉnh. Muốn kiểm chứng phiên bản cải tiến cần một holdout mới, chưa dùng trong phát triển, hoặc đánh giá test chính thức của ban tổ chức.

---

## Experiment — 2026-10-20 — quantum-diff-histgb-oof-r3-temporal-quantum-ablation

### Câu hỏi / giả thuyết

Fidelity riêng từng radar theo độ trễ 0,1/0,3 s và entropy/purity của density matrix cửa sổ quá khứ 0,4/0,8 s có cải thiện readout trên cùng biểu diễn motion không?

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; 13 stream train, 5 fold nhóm theo `stream_id`, seed 2027. Không đọc validation hoặc `streams_composition.csv` trong feature extraction/prediction. Cờ quality/needs_review/ignore được mask theo quy tắc package.

### Tiền xử lý và cửa sổ hóa

Giữ sai phân CIR nhân quả, block 5 frame/10 Hz và 4-qubit amplitude encoding. Thêm fidelity cùng radar theo lag 1/3 block và von Neumann entropy/purity của density matrix thời gian cuộn chỉ dùng trạng thái hiện tại/quá khứ.

### Mô hình và cấu hình

Giữ nguyên HistGradientBoosting r2 và decoder grid cũ để cô lập ablation đặc trưng.

### Seed ngẫu nhiên

2027.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; đây là protocol package-local, chưa được xác nhận là công thức xếp hạng chính thức.

### Kết quả

OOF mean event F1 = **0.7276**; F1@0.1/0.3/0.5 = 0.7868/0.7513/0.6447. Tại tIoU 0.3: TP/FP/FN = 148/60/38; median start error = 0.18 s; false alarms/hour = 9.71. So với r2, ablation này không cải thiện primary mean F1.

### Phân tích lỗi

Audit train-only ghi nhận 60 FP: 38 fall-source, 15 other, 7 ADL; có 17 fall-source FP overlap event nhưng dưới IoU 0.3. Không chọn nhánh temporal này làm ứng viên chính.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_oof_r3.yaml](../../configs/quantum_histgb_oof_r3.yaml)
- Artifact OOF bị ignore: `data/processed/quantum_diff_histgb_oof_20261020_r3/`
- Lệnh: `./.venv/bin/python -u scripts/run_oof.py --config configs/quantum_histgb_oof_r3.yaml`

### Bước tiếp theo

Mở rộng decoder search trên train OOF và đánh giá bằng nested grouped CV.

---

## Experiment — 2026-10-20 — quantum-diff-histgb-oof-r4-expanded-decoder

### Câu hỏi / giả thuyết

Decoder r2 chạm biên trên của transition cost và min duration. Grid rộng hơn có thể giảm báo động giả mà không làm mất quá nhiều event.

### Phiên bản dữ liệu và split

13 stream train, cùng 5 fold và seed 2027 như r3. Chỉ dùng train OOF logits/nhãn; validation không được đọc.

### Tiền xử lý và cửa sổ hóa

Dùng feature set r3. Không đổi cách trích đặc trưng hoặc readout.

### Mô hình và cấu hình

Giữ HistGradientBoosting base; mở rộng decoder thành 7 quantile × 5 transition costs × 6 min durations. Tham số được chọn bằng điểm OOF train.

### Seed ngẫu nhiên

2027.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; kết quả OOF này có thiên lệch chọn tham số vì cùng OOF predictions vừa dùng chọn vừa báo cáo.

### Kết quả

Mean event F1 = **0.7553**; F1@0.1/0.3/0.5 = 0.8421/0.7756/0.6482. Lựa chọn: bias 0.30921, transition cost 3.0, min duration 1.0 s. Tại tIoU 0.3: TP/FP/FN = 140/35/46; median start error 0.28 s; false alarms/hour 5.66.

### Phân tích lỗi

FP giảm còn 35 nhưng FN tăng lên 46; start error tăng so với r2. Audit train-only: 19 fall-source, 14 other và 2 ADL FP. Đây là trade-off nhạy ngưỡng, không phải bằng chứng holdout tổng quát hóa.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_oof_r4.yaml](../../configs/quantum_histgb_oof_r4.yaml)
- Artifact và audit bị ignore: `data/processed/quantum_diff_histgb_oof_20261020_r4/`
- Lệnh OOF: `./.venv/bin/python -u scripts/run_oof.py --config configs/quantum_histgb_oof_r4.yaml`
- Lệnh audit train: `./.venv/bin/python scripts/audit_oof.py --artifact-dir data/processed/quantum_diff_histgb_oof_20261020_r4`

### Bước tiếp theo

Đánh giá chọn decoder lồng trong GroupKFold để tách dữ liệu chọn và dữ liệu chấm.

---

## Experiment — 2026-10-20 — quantum-diff-histgb-nested-oof-r5

### Câu hỏi / giả thuyết

Mức điểm r4 có giữ được khi chọn decoder trên inner OOF và chỉ chấm trên outer train streams chưa tham gia fit/tuning không?

### Phiên bản dữ liệu và split

Nested GroupKFold theo `stream_id`: 5 outer fold × 4 inner fold trên 13 stream train. Validation không được mở. Metric chỉ trên outer held-out train streams.

### Tiền xử lý và cửa sổ hóa

Dùng r3 causal fidelity/rolling density features. Cache được tạo riêng theo feature version.

### Mô hình và cấu hình

HistGradientBoosting base. Decoder được chọn từ compact grid 3 quantiles × 3 transition costs × 2 min durations bên trong mỗi outer train split.

### Seed ngẫu nhiên

Outer seed 2027; inner seed là 3027 cộng chỉ số outer fold.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; đây là nested train OOF, không phải validation challenge.

### Kết quả

Nested mean event F1 = **0.7289**; F1@0.1/0.3/0.5 = 0.8160/0.7520/0.6187. Tại tIoU 0.3: TP/FP/FN = 141/48/45; median start error 0.24 s; false alarms/hour 7.76.

### Phân tích lỗi

Outer-fold mean F1 dao động từ 0.6771 đến 0.7692. Điểm thấp hơn OOF đã dùng chọn decoder, xác nhận cần báo riêng selection score và nested estimate.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_oof.yaml](../../configs/quantum_histgb_nested_oof.yaml)
- Mã nested OOF: [run_nested_oof.py](../../scripts/run_nested_oof.py)
- Artifact bị ignore: `data/processed/quantum_diff_histgb_nested_oof_20261020/`
- Lệnh: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_oof.yaml`

### Bước tiếp theo

So sánh readout regularization bên trong nested CV và khảo sát motion/static state hybrid nhân quả.

---

## Experiment — 2026-10-20 — quantum-diff-histgb-nested-readout-sweep-r6

### Câu hỏi / giả thuyết

Chọn giữa năm mức regularization/cân bằng lớp bằng inner OOF có giảm độ biến thiên và false alarms trên outer train streams?

### Phiên bản dữ liệu và split

13 stream train; 5 outer × 3 inner group folds. Toàn bộ readout và decoder selection nằm trong inner folds; outer labels chỉ được chấm một lần. Không đọc validation.

### Tiền xử lý và cửa sổ hóa

Dùng r3 motion/quantum feature set, 5 frame mỗi bước 0,1 s.

### Mô hình và cấu hình

Inner sweep gồm base và từng thay đổi độc lập: max leaves 7, min leaf 200, L2=8, positive weight=3. Decoder grid 18 cấu hình dùng chung cho các biến thể.

### Seed ngẫu nhiên

2027; seed inner thay đổi theo outer fold để tái lập.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; nested outer train OOF.

### Kết quả

Mean event F1 = **0.7331**; F1@0.1/0.3/0.5 = 0.8033/0.7535/0.6427. Tại tIoU 0.3: TP/FP/FN = 136/39/50; median start error 0.23 s; false alarms/hour 6.31. Inner selection chọn base ở hai outer fold, lower positive weight ở ba fold.

### Phân tích lỗi

Train-only composition audit: 39 FP gồm 20 fall-source, 16 other và 3 ADL; 12 fall-source FP overlap event nhưng dưới IoU 0.3. Điểm mean tương đương r5 nhưng false alarms/hour và F1@0.5 tốt hơn; không chọn theo outer score cho bước tuning tiếp theo.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_readout_sweep.yaml](../../configs/quantum_histgb_nested_readout_sweep.yaml)
- Mã: [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_histgb_nested_readout_sweep_20261020/`
- Lệnh: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_readout_sweep.yaml`

### Bước tiếp theo

Kiểm tra profile tĩnh bằng background EMA riêng từng stream; không dùng metadata composition trong inference.

---

## Experiment — 2026-10-20 — quantum-static-hybrid-nested-r7

### Câu hỏi / giả thuyết

Một nhánh profile tĩnh nhân quả có giúp nhận ra thay đổi postural/landing khi motion yếu, bổ sung cho state 4-qubit chỉ mã hóa motion không?

### Phiên bản dữ liệu và split

13 stream train; nested 5 outer × 4 inner GroupKFold. Baseline EMA khởi tạo từ frame đầu và reset theo stream. Validation, nhãn validation và `streams_composition.csv` không tham gia trích xuất đặc trưng.

### Tiền xử lý và cửa sổ hóa

Giữ sai phân CIR và quantum motion state. Thêm contrast tuyệt đối so với EMA causal tau=8 s; cập nhật baseline sau từng block hiện tại. Ghép motion/static amplitude state thành state 5-qubit với `gamma=1.0`; thêm flag-Z, flag entanglement, hybrid density entropy/purity và fidelity theo thời gian.

### Mô hình và cấu hình

HistGradientBoosting base; decoder chọn trong inner OOF với grid 18 cấu hình. Không chọn tau/gamma bằng validation.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; nested outer train OOF.

### Kết quả

Nested mean event F1 = **0.8000**; F1@0.1/0.3/0.5 = 0.8623/0.8260/0.7117. Tại tIoU 0.3: TP/FP/FN = 159/40/27; median start error 0.20 s; false alarms/hour 6.47. Outer-fold mean F1 nằm trong khoảng 0.7463–0.8783.

### Phân tích lỗi

Audit chỉ trên train composition: 40 FP gồm 22 fall-source, 13 ADL, 5 other; 11 fall-source FP overlap event nhưng chưa đạt IoU 0.3. Nhánh tăng recall rõ nhưng cũng tạo thêm FP ở ADL, nên cần giữ nguyên nhãn “ablation chưa kiểm chứng trên holdout”.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static.yaml](../../configs/quantum_histgb_nested_static.yaml)
- Mã feature: [quantum_features.py](../../src/abc_uwb_falltrack/quantum_features.py)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_static_hybrid_nested_oof_20261020/`
- Lệnh: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static.yaml`

### Bước tiếp theo

Kiểm tra regularization bên trong nested CV để giảm false alarms mà không dùng outer/validation labels cho chọn tham số.

---

## Experiment — 2026-10-20 — quantum-static-hybrid-nested-readout-sweep-r8

### Câu hỏi / giả thuyết

Readout regularization được chọn bên trong inner folds có giảm false alarms của nhánh static hybrid mà vẫn giữ điểm event-level không?

### Phiên bản dữ liệu và split

13 stream train; 5 outer × 3 inner group folds. Không đọc validation. Năm readout candidates và decoder đều chỉ chọn trong từng outer train split.

### Tiền xử lý và cửa sổ hóa

Như r7: per-stream causal EMA tau=8 s, static/motion 5-qubit state, gamma=1.0.

### Mô hình và cấu hình

Base HistGradientBoosting cùng bốn ablation một biến: max leaves 7, min leaf 200, L2=8 và positive weight=3. Decoder grid 18 cấu hình.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; nested outer train OOF.

### Kết quả

Mean event F1 = **0.7844**; F1@0.1/0.3/0.5 = 0.8424/0.8098/0.7011. Tại tIoU 0.3: TP/FP/FN = 149/33/37; median start error 0.18 s; false alarms/hour 5.34. Lower positive weight được chọn ở ba outer folds, base một fold, stronger L2 một fold.

### Phân tích lỗi

Audit train-only: 33 FP gồm 18 fall-source, 11 ADL, 4 other; 9 fall-source FP overlap event nhưng IoU dưới 0.3. So với r7, F1 chính thấp hơn 0.0156 nhưng giảm FP và cải thiện start error/false alarms. Đây là trade-off giữa recall của base và độ thưa cảnh báo của regularized selection.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static_readout_sweep.yaml](../../configs/quantum_histgb_nested_static_readout_sweep.yaml)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_static_hybrid_nested_readout_sweep_20261020/`
- Lệnh: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static_readout_sweep.yaml`

### Bước tiếp theo

Tạo cấu hình ứng viên cuối từ full train OOF; chờ một holdout mới hoặc evaluator test chính thức để kiểm chứng.

---

## Experiment — 2026-10-20 — quantum-static-hybrid-full-train-oof-candidate

### Câu hỏi / giả thuyết

Sau khi nested CV xác nhận nhánh static hybrid trên train streams, cấu hình full-train nào được khóa để chấm một holdout mới?

### Phiên bản dữ liệu và split

13 stream train, 5-fold GroupKFold, seed 2027. Chỉ train OOF được dùng chọn decoder. Validation đã được dùng trước đó cho một cấu hình cũ và không được mở lại.

### Tiền xử lý và cửa sổ hóa

Motion sai phân CIR + causal per-stream static-profile EMA tau=8 s; hybrid state 5-qubit, gamma=1.0. `streams_composition.csv` chỉ được đọc trong audit train sau OOF, không tham gia feature/predict.

### Mô hình và cấu hình

HistGradientBoosting base; full OOF decoder search. Cấu hình khóa: bias 0.429099, transition cost 2.0, min duration 1.0 s.

### Seed ngẫu nhiên

2027.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; full-train OOF là kết quả có selection optimism. Nested estimate r7 là bằng chứng tổng quát hóa train-only mạnh hơn.

### Kết quả

Full-train OOF mean event F1 = **0.8087**; F1@0.1/0.3/0.5 = 0.8798/0.8415/0.7049. Tại tIoU 0.3: TP/FP/FN = 154/26/32; median start error 0.20 s; false alarms/hour 4.21. Kết quả validation mục tiêu vẫn **chưa được xác nhận** cho candidate này.

### Phân tích lỗi

Audit train-only: 26 FP gồm 17 fall-source, 7 ADL, 2 other; 10 fall-source FP overlap event nhưng chưa đạt IoU 0.3. Không xem/chấm lại validation sau khi chọn cấu hình.

### Artifact và lệnh tái lập

- Cấu hình OOF: [quantum_histgb_static_oof.yaml](../../configs/quantum_histgb_static_oof.yaml)
- Cấu hình ứng viên holdout mới: [quantum_histgb_static_final_candidate.yaml](../../configs/quantum_histgb_static_final_candidate.yaml)
- Inference train-only: [predict_holdout.py](../../scripts/predict_holdout.py) nhận `--streams-dir`, `--split-name` và `--output-dir` để chấm một thư mục holdout mới.
- Artifact bị ignore: `data/processed/quantum_diff_static_hybrid_oof_20261020/`
- Lệnh OOF: `./.venv/bin/python -u scripts/run_oof.py --config configs/quantum_histgb_static_oof.yaml`
- Lệnh train error audit: `./.venv/bin/python scripts/audit_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_oof_20261020`
- Mẫu lệnh dự đoán cho holdout mới: `./.venv/bin/python -u scripts/predict_holdout.py --config configs/quantum_histgb_static_final_candidate.yaml --streams-dir data/uwb_falltrack/new_holdout/streams --split-name test`

### Bước tiếp theo

Chưa tạo submission hay điểm validation cho candidate. Cần holdout mới chưa dùng hoặc official test evaluator để xác nhận mốc 0.70 mà không tái sử dụng năm stream validation đã tiêu thụ.

---

## Experiment — 2026-10-20 — quantum-static-hybrid-nested-boundary-gamma-r9

### Câu hỏi / giả thuyết

Chọn static/motion weighting và các dịch chuyển biên nhỏ hoàn toàn bên trong inner OOF có tăng tIoU nghiêm ngặt mà vẫn giảm false alarm trên outer train streams không?

### Phiên bản dữ liệu và split

13 stream train; 5 outer × 4 inner GroupKFold, seed 2027. Mỗi outer fold chọn gamma, readout và decoder trên inner folds rồi chấm một lần trên các stream outer giữ lại. Không mở validation.

### Tiền xử lý và cửa sổ hóa

Sai phân CIR và static-profile contrast với EMA causal tau=8 s, reset theo stream; các state 5-qubit được tạo với gamma ∈ {1.0, 1.5, 2.0}. Bước cửa sổ là 5 frame/0,1 s. `streams_composition.csv` chỉ dùng sau dự đoán trong audit train.

### Mô hình và cấu hình

HistGradientBoosting base. Inner decoder search gồm logit quantiles 0.95/0.975/0.99, transition cost 1/2/3, min duration 0.8/1.2 s, start offsets −0.2/−0.1/0 s và end offsets −0.3/−0.2/−0.1/0 s. Alarm vẫn lấy từ thời điểm decoder phát hiện trước dịch biên.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; nested outer train OOF. Đây là metric package-local, chưa xác nhận là ranking chính thức.

### Kết quả

Nested mean event F1 = **0.8228**; F1@0.1/0.3/0.5 = 0.8737/0.8421/0.7526; F1@0.7 = 0.4737. Tại tIoU 0.3: TP/FP/FN = 160/34/26; median absolute start error 0.17 s; false alarms/hour 5.50. So với nested static-hybrid r7, mean F1 tăng 0.0228, F1@0.5 tăng 0.0409, FP giảm 6 và FN giảm 1. Gamma 2.0 được chọn ở ba outer folds; gamma 1.5 ở hai folds. Mọi fold chọn start offset −0.2 s; end offset −0.1 s ở một fold và 0 s ở bốn folds.

### Phân tích lỗi

Audit train-only gán 18 FP vào fall-source, 14 ADL và 2 other; 8 fall-source FP chạm event nhưng chưa đạt IoU 0.3, 5 nằm trước và 5 sau. Tại theta 0.5, signed start error trung vị là −0.02 s nhưng end vẫn muộn +0.24 s; tỉ lệ duration dự đoán/ground truth trung vị 1.136. Đây là tín hiệu để xem xét hiệu chỉnh end theo độ dài dự đoán trong một vòng nested mới, không phải kết luận từ validation.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static_boundary.yaml](../../configs/quantum_histgb_nested_static_boundary.yaml)
- Mã: [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_static_hybrid_nested_boundary_gamma_20261020/`
- Lệnh nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static_boundary.yaml`
- Lệnh audit train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_boundary_gamma_20261020`

### Bước tiếp theo

Nếu inner folds ủng hộ, thử end offset theo duration dự đoán với một số ngưỡng nhỏ và khóa quy tắc trước khi chấm outer fold. Không dùng lại validation đã tiêu thụ; chưa tạo submission.

---

## Experiment — 2026-10-20 — quantum-static-hybrid-nested-asymmetric-exit-r10

### Câu hỏi / giả thuyết

Exit penalty riêng, thấp hơn entry penalty, có thể rút ngắn event tail mà không dịch start thêm; positive weight=4 có giảm ADL false alarms mà vẫn giữ recall không?

### Phiên bản dữ liệu và split

13 stream train; nested 5 outer × 4 inner GroupKFold, seed 2027. Inner folds chọn gamma, positive weight, decoder và offsets; không mở validation.

### Tiền xử lý và cửa sổ hóa

Như r9: sai phân CIR cộng static contrast causal EMA tau=8 s; gamma {1.0, 1.5, 2.0}; block 5 frame/0,1 s. Không dùng composition trong inference; composition chỉ đọc trong train audit.

### Mô hình và cấu hình

HistGradientBoosting so sánh positive weight 5 và 4. Ising dùng entry cost {2,3}, exit cost {0.5,1,1.5,2,3}; start offset khóa −0.2 s và end offset 0 s để cô lập tác động penalty.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; nested outer train OOF, metric package-local.

### Kết quả

Nested mean event F1 = **0.8174**; F1@0.1/0.3/0.5 = 0.8777/0.8404/0.7340; F1@0.7 = 0.4787. Tại tIoU 0.3: TP/FP/FN = 158/32/28; median absolute start error 0.18 s; false alarms/hour 5.18. So với r9, FP giảm 2 nhưng FN tăng 2, mean F1 giảm 0.0054, F1@0.5 giảm 0.0186; chỉ số false alarms/hour tốt hơn 0.32.

### Phân tích lỗi

Audit train-only có 18 fall-source, 12 ADL và 2 other FP. Positive weight 4 được chọn ở ba folds, weight 5 ở hai; exit thấp hơn entry ở hai folds, bằng nhau một fold và cao hơn ở hai folds, nên giả thuyết exit penalty thấp chưa được ủng hộ ổn định. R9 giữ điểm mean/tIoU .5 và start error tốt hơn; r10 chỉ có lợi thế nhỏ về false alarms/hour.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static_asym_exit.yaml](../../configs/quantum_histgb_nested_static_asym_exit.yaml)
- Mã: [oof.py](../../src/abc_uwb_falltrack/oof.py), [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_static_hybrid_nested_asym_exit_20261020/`
- Lệnh nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static_asym_exit.yaml`
- Lệnh audit train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_asym_exit_20261020`

### Bước tiếp theo

Giữ r9 làm ứng viên train-only mạnh hơn hiện tại; chưa tạo submission vì validation đã tiêu thụ và điểm mục tiêu cho phiên bản mới chưa được xác nhận. Nếu có holdout mới, đánh giá một lần sau khi khóa cấu hình.

---

## Experiment — 2026-10-20 — quantum-static-hybrid-nested-duration-profile-r11

### Câu hỏi / giả thuyết

End offset được chọn theo duration dự đoán có giảm các interval dài quá mức và tăng F1 tại tIoU nghiêm ngặt so với một offset dùng chung không?

### Phiên bản dữ liệu và split

13 stream train; nested 5 outer × 4 inner GroupKFold, seed 2027. Gamma và duration profile được chọn chỉ từ inner OOF trước khi chấm outer fold. Validation không được mở.

### Tiền xử lý và cửa sổ hóa

Sai phân CIR cùng static contrast causal EMA tau=8 s, reset theo stream; gamma candidates 1.5 và 2.0; block 5 frame/0,1 s. Duration cutoff dùng predicted raw interval; không dùng nhãn hoặc composition trong inference.

### Mô hình và cấu hình

HistGradientBoosting base. Giữ start offset −0.2 s và so sánh uniform end với end offset −0.1/−0.2/−0.3 s cho predicted duration >1.8 s; cutoff và offset được chọn hoàn toàn bằng inner OOF. `t_alarm` giữ theo thời điểm detector phát hiện.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold.

### Metric theo protocol

Evaluator package với semantics `--ignore-flagged`; nested outer train OOF. Kết quả package-local, chưa xác nhận ranking chính thức.

### Kết quả

Nested mean event F1 = **0.8342**; F1@0.1/0.3/0.5 = 0.8796/0.8429/0.7801; F1@0.7 = 0.5026. Tại tIoU 0.3: TP/FP/FN = 161/35/25; median absolute start error 0.16 s; false alarms/hour 5.66. So với r9, mean F1 tăng 0.0114, F1@0.5 tăng 0.0275, FN giảm 1 và median start error giảm 0.01 s; false alarms/hour tăng 0.16. Inner selection chọn profile end −0.2 s với threshold 1.8 s ở ba folds, −0.3 s ở một fold và uniform ở một fold.

### Phân tích lỗi

Audit train-only gán 19 FP vào fall-source, 14 ADL và 2 other; 9 fall-source FP overlap event dưới IoU 0.3. Ở theta 0.5, median signed start/end error là −0.02/+0.20 s, so với −0.02/+0.24 s ở r9; median predicted/GT duration ratio tăng nhẹ từ 1.136 lên 1.139. R11 là kết quả development-only vì nhiều vòng lặp đã dùng các train folds để định hướng giả thuyết.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static_duration_profile.yaml](../../configs/quantum_histgb_nested_static_duration_profile.yaml)
- Mã: [oof.py](../../src/abc_uwb_falltrack/oof.py), [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_static_hybrid_nested_duration_profile_20261020/`
- Lệnh nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static_duration_profile.yaml`
- Lệnh audit train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_duration_profile_20261020`

### Bước tiếp theo

Đánh giá một quy tắc hai chế độ với cutoff dự đoán lấy từ quantile duration của inner OOF, để tránh cố định cutoff từ audit outer. Chưa tạo submission; cần holdout mới để xác nhận target validation.


## Experiment — 2026-10-20 — quantum-static-hybrid-nested-duration-quantiles-r12

### Câu hỏi / giả thuyết

Cutoff độ dài dự đoán lấy từ quantile inner OOF theo từng outer fold có giữ/cải thiện F1 nghiêm ngặt so với cutoff cố định 1.8 s từ inner search không?

### Phiên bản dữ liệu và split

13 stream train; nested 5 outer × 4 inner GroupKFold, seed 2027. Chọn feature variant và decoder bằng inner OOF; ngưỡng q25/q75 được tính từ thời lượng raw interval dự đoán trong inner OOF của candidate đó rồi khóa trước khi chấm outer fold. `validation_used=false`, `composition_file_used=false`, dùng `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

Giữ pipeline r11: sai phân CIR và static contrast causal EMA tau=8 s, reset theo stream; gamma 1.5/2.0; block 5 frame/0,1 s. Không dùng composition trong inference.

### Mô hình và cấu hình

HistGradientBoosting base. So sánh uniform, global end trim và profile adaptive q25/q75. Adaptive offsets: short start −0.3/end 0; middle start/end −0.2; long start −0.2/end −0.3 s. Các cutoff được outer fold chọn từ inner OOF, lần lượt khoảng [1.4, 2.2], [1.3, 2.2], [1.4, 2.3] và [1.4, 2.4] s; một fold chọn uniform.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold.

### Metric theo protocol

Evaluator package-local với semantics `--ignore-flagged`; điểm nested train OOF, chưa xác nhận ranking chính thức.

### Kết quả

Nested mean event F1 = **0.8355**; F1@0.1/0.3/0.5 = 0.8825/0.8460/0.7781; F1@0.7 = 0.5326. Tại tIoU 0.3: TP/FP/FN = 162/35/24; median absolute start error 0.18 s; false alarms/hour 5.66. So với r11, mean F1 tăng 0.0013 và F1@0.3 tăng 0.0030, nhưng F1@0.5 giảm 0.0020 và start error tăng 0.02 s; false alarms/hour không đổi. Đây là khác biệt nhỏ, không khẳng định cải thiện tổng quát hóa.

### Phân tích lỗi

Audit train-only ghi nhận 19 fall-source, 14 ADL và 2 other FP; tại theta 0.3 có 24 FN. Ở theta 0.5, median signed start/end error là −0.04/+0.12 s và median predicted/GT duration ratio là 1.070, so với −0.02/+0.20 s và 1.139 ở r11. Kết quả gợi ý end boundary và độ dài được hiệu chỉnh tốt hơn, trong khi median absolute start error ở theta 0.3 xấu hơn nhẹ. R12 là development-only vì các vòng lặp đã dùng train OOF để định hướng giả thuyết.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static_duration_quantiles.yaml](../../configs/quantum_histgb_nested_static_duration_quantiles.yaml)
- Mã: [oof.py](../../src/abc_uwb_falltrack/oof.py), [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_static_hybrid_nested_duration_quantiles_20261020/`
- Lệnh nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static_duration_quantiles.yaml`
- Lệnh audit train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_duration_quantiles_20261020`

### Bước tiếp theo

Dùng r12 làm baseline train-only cho một thí nghiệm thêm tương phản fidelity lag1−lag3 theo từng radar; giữ feature này trong lựa chọn inner OOF và không truy cập validation.


## Experiment — 2026-10-20 — quantum-static-hybrid-nested-fidelity-contrast-r13

### Câu hỏi / giả thuyết

Ba feature delta fidelity `lag1 − lag3` theo từng radar có giúp readout phân biệt pha động học ngắn với chuyển động lặp lại trong ADL, trên baseline r12 không?

### Phiên bản dữ liệu và split

13 stream train; nested 5 outer × 4 inner GroupKFold, cùng seed/split với r12. Mỗi outer fold chọn gamma, có/không có delta fidelity và decoder bằng inner OOF; khóa lựa chọn trước outer scoring. `validation_used=false`, `composition_file_used=false`, dùng `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

Giữ pipeline r12: sai phân CIR, static contrast causal EMA tau=8 s reset theo stream, gamma 1.5/2.0 và block 5 frame/0,1 s. Delta chỉ dùng fidelity của motion state hiện tại với state trễ 1 và 3 bước; không dùng tương lai. History và startup padding giữ nguyên, cache có feature version riêng.

### Mô hình và cấu hình

HistGradientBoosting base và toàn bộ decoder/duration search r12 giữ nguyên. Thêm ba cột `state_fidelity_lag1_minus_lag3_r{0,1,2}`; giữ nguyên sáu feature fidelity nguồn. So sánh hai gamma với delta tắt/bật trong inner selection.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold.

### Metric theo protocol

Evaluator package-local với semantics `--ignore-flagged`; nested train OOF, chưa xác nhận ranking chính thức.

### Kết quả

Nested mean event F1 = **0.8316**; F1@0.1/0.3/0.5 = 0.8789/0.8474/0.7684; F1@0.7 = 0.5316. Tại tIoU 0.3: TP/FP/FN = 161/33/25; median absolute start error 0.18 s; false alarms/hour 5.34. So với r12, mean F1 giảm 0.0039 và F1@0.5 giảm 0.0096; F1@0.3 tăng 0.0014, start error giữ 0.18 s và false alarms/hour giảm 0.32. Delta được chọn trong 2/5 folds; outer mean F1 của hai fold đó giảm lần lượt 0.0126 và 0.0087 so với r12, ba fold còn lại trùng r12.

### Phân tích lỗi

Audit train-only ghi nhận 18 fall-source, 13 ADL và 2 other FP; tại theta 0.3 có 25 FN. Có 8 fall-source FP overlap event nhưng dưới IoU 0.3. Feature delta chưa cải thiện nhất quán metric chính hoặc tIoU 0.5. Kết quả development-only.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static_fidelity_contrast.yaml](../../configs/quantum_histgb_nested_static_fidelity_contrast.yaml)
- Mã: [quantum_features.py](../../src/abc_uwb_falltrack/quantum_features.py), [oof.py](../../src/abc_uwb_falltrack/oof.py), [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_static_hybrid_nested_fidelity_contrast_20261020/`
- Lệnh nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static_fidelity_contrast.yaml`
- Lệnh audit train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_fidelity_contrast_20261020`

### Bước tiếp theo

Không bật delta fidelity trong ứng viên hiện tại. Giữ r12 làm baseline train-only có mean F1 cao nhất; thiết kế giả thuyết tiếp theo từ audit train-only, không dùng validation.


## Experiment — 2026-10-20 — quantum-static-hybrid-nested-duration-endtrim-r14

### Câu hỏi / giả thuyết

End boundary còn trễ nhẹ ở tIoU 0.5; thêm một profile adaptive q25/q75 trim đầu ra thêm 0.1 s có tăng F1 nghiêm ngặt mà không làm xấu metric chính hay tie-breakers tại tIoU 0.3 không?

### Phiên bản dữ liệu và split

13 stream train; nested 5 outer × 4 inner GroupKFold, cùng seed/split với r12. Chọn gamma và decoder bằng inner OOF; q25/q75 tính lại từ raw predicted durations của inner OOF trong từng outer fold. `validation_used=false`, `composition_file_used=false`, dùng `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

Giữ pipeline r12: sai phân CIR và static contrast causal EMA tau=8 s, reset theo stream; gamma 1.5/2.0; block 5 frame/0,1 s. Fidelity contrast R13 tắt.

### Mô hình và cấu hình

HistGradientBoosting base và các candidate r12 giữ nguyên. Thêm duy nhất profile adaptive q25/q75 với offsets short start/end = −0.3/−0.1 s; middle = −0.2/−0.3 s; long = −0.2/−0.4 s. So với adaptive r12, end được trim thêm 0.1 s ở cả ba bin.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold.

### Metric theo protocol

Evaluator package-local với semantics `--ignore-flagged`; nested train OOF, chưa xác nhận ranking chính thức.

### Kết quả

Nested mean event F1 = **0.8390**; F1@0.1/0.3/0.5 = 0.8825/0.8460/0.7885; F1@0.7 = 0.5431. Tại tIoU 0.3: TP/FP/FN = 162/35/24; median absolute start error 0.18 s; false alarms/hour 5.66. So với r12, mean F1 tăng 0.0035 và F1@0.5 tăng 0.0104; F1@0.3 và cả hai tie-breakers tại 0.3 giữ nguyên. Profile mới được chọn ở 2/5 folds; hai outer fold đó tăng F1@0.5 lần lượt 0.0222 và 0.0313 so với r12; ba fold còn lại trùng r12.

### Phân tích lỗi

Audit train-only cho thấy tại theta 0.5 TP/FP/FN = 151/46/35, so với 149/48/37 ở r12. Median signed end error giảm từ +0.12 xuống +0.10 s, predicted/GT duration ratio giảm từ 1.070 xuống 1.049. Absolute end error median vẫn 0.20 s; tại theta 0.3 các counts và tie-breakers không đổi. R14 là development-only.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static_duration_endtrim.yaml](../../configs/quantum_histgb_nested_static_duration_endtrim.yaml)
- Mã: [oof.py](../../src/abc_uwb_falltrack/oof.py), [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact và audit train bị ignore: `data/processed/quantum_diff_static_hybrid_nested_duration_endtrim_20261020/`
- Lệnh nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static_duration_endtrim.yaml`
- Lệnh audit train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_duration_endtrim_20261020`

### Bước tiếp theo

Giữ profile mới trong ứng viên train-only hiện tại. Không tạo submission hoặc tuyên bố đạt validation target; cần một holdout độc lập chưa dùng hoặc official evaluator để xác nhận.


## Experiment — 2026-10-20 — quantum-static-hybrid-nested-duration-endtrim-r15

### Câu hỏi / giả thuyết

End boundary của các sự kiện match tại tIoU 0.5 còn trễ median +0.10 s. Trim thêm 0.1 s ở cả ba duration bin có giúp tăng F1 nghiêm ngặt không?

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; 13 stream train, nested 5 outer × 4 inner GroupKFold theo `stream_id`, seed 2027. Profile chỉ được chọn bằng inner OOF; outer fold được chấm sau khi khóa lựa chọn. `validation_used=false`; `composition_file_used=false` trong mô hình. Không đọc validation.

### Tiền xử lý và cửa sổ hóa

Giữ nguyên R14: sai phân CIR và static contrast causal EMA tau=8 s reset theo stream; gamma candidates 1.5/2.0; block 5 frame/0,1 s. Feature cache R14 được tái sử dụng; không dùng composition trong extraction/prediction.

### Mô hình và cấu hình

HistGradientBoosting và toàn bộ decoder R14 giữ nguyên. Thêm duy nhất profile adaptive q25/q75 `adaptive_q25_q75_end_minus_0_2`, với start offset short/middle/long = −0.3/−0.2/−0.2 s và end offset = −0.2/−0.4/−0.5 s. Ngưỡng q25/q75 được tính từ raw predicted durations trong inner OOF của fold.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold như cấu hình nested OOF.

### Metric theo protocol

Evaluator package-local với semantics `--ignore-flagged`; nested train OOF. Metric này chưa được xác nhận là ranking chính thức.

### Kết quả

Nested mean event F1 = **0.8390**; F1@0.1/0.3/0.5 = **0.8825/0.8460/0.7885**. Tại tIoU 0.3: TP/FP/FN = 162/35/24; median absolute start error = 0.18 s; false alarms/hour = 5.66. Tất cả năm outer fold và aggregate trùng R14. Profile mới không được inner selection chọn ở fold nào; giữ nguyên lựa chọn gamma và duration profile R14.

### Phân tích lỗi

Train-only audit giữ nguyên kết quả R14: tại theta 0.3 có 19 false alarm từ fall-source, 14 ADL và 2 other; 24 event chưa match. Tại theta 0.5, median signed end error vẫn +0.10 s, median absolute end error 0.20 s và predicted/GT duration ratio 1.049. R15 không cải thiện boundary hoặc F1. Audit kiểm tra lại partition outer/inner và coverage logits; các cờ leakage đều đạt, validation không được dùng. `streams_composition.csv` chỉ được đọc trong audit lỗi train.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_static_duration_endtrim_r15.yaml](../../configs/quantum_histgb_nested_static_duration_endtrim_r15.yaml)
- Mã: [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact train-only bị ignore: `data/processed/quantum_diff_static_hybrid_nested_duration_endtrim_r15_20261020/`
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_static_duration_endtrim_r15.yaml`
- Audit lỗi train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_duration_endtrim_r15_20261020`

### Bước tiếp theo

Không giữ profile mới vì không được chọn và không đổi outer metric. R14 vẫn là train-only candidate đứng đầu; không dùng lại validation đã tiêu thụ. Mọi tiếp tục tuning trên 13 stream này chỉ là phát triển có nguy cơ selection optimism; cần test package hoặc holdout mới để xác nhận mục tiêu.


## Experiment — 2026-10-20 — quantum-static-hybrid-nested-rho-similarity-r16

### Câu hỏi / giả thuyết

Chuẩn hóa temporal overlap của density matrix theo purity hiện tại và quá khứ có bổ sung tín hiệu hữu ích ngoài `rho_temporal_overlap` và `rho_purity` hiện có không?

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; 13 stream train, nested 5 outer × 4 inner GroupKFold theo `stream_id`, seed 2027. Feature variant và decoder được chọn trong inner OOF; outer fold chỉ được chấm sau khi khóa lựa chọn. `validation_used=false`, `composition_file_used=false` trong OOF. Evaluator package-local dùng semantics `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

Giữ R14: sai phân CIR và static contrast causal EMA tau=8 s reset theo stream; gamma 1.5/2.0; block 5 frame/0,1 s. Feature mới là normalized Hilbert–Schmidt similarity giữa density matrix hiện tại và trạng thái trễ một bước, chuẩn hóa bằng căn tích purity hai thời điểm, clip về [0,1]. Khởi tạo trung tính bằng 1 ở frame đầu; giữ state qua ranh giới chunk và reset theo stream. Đây không phải Uhlmann fidelity.

### Mô hình và cấu hình

HistGradientBoosting và decoder search của R14. So sánh gamma 1.5/2.0, mỗi biến thể có hoặc không có `rho_temporal_similarity_normalized`; chi tiết tại [quantum_histgb_nested_rho_similarity_r16.yaml](../../configs/quantum_histgb_nested_rho_similarity_r16.yaml). Feature mới được inner selection chọn ở 2/5 outer folds.

### Seed ngẫu nhiên

2027; inner seed thay đổi theo outer fold như cấu hình nested OOF.

### Metric theo protocol

Evaluator package-local với semantics `--ignore-flagged`; metric nested train OOF, chưa được xác nhận là ranking chính thức. Report ghi tất cả kiểm tra split, namespace và coverage OOF là true.

### Kết quả

Nested mean event F1 = **0.8412**; F1@0.1/0.3/0.5 = **0.8796/0.8482/0.7958**. So với R14 (0.8390; 0.8825/0.8460/0.7885), mean tăng 0.0022, F1@0.1 giảm 0.0029, F1@0.3 tăng 0.0022 và F1@0.5 tăng 0.0073. Tại tIoU 0.3: TP/FP/FN = 162/34/24; median absolute start error = 0.18 s; false alarms/hour = 5.50 (R14: 162/35/24, 0.18 s, 5.66).

### Phân tích lỗi

Audit train-only tại theta 0.5 ghi nhận TP/FP/FN = 152/44/34; median signed end error +0.04 s và median absolute end error 0.20 s; predicted/GT duration ratio median 1.042. False alarm theo loại: 18 fall-source, 14 ADL và 2 other, so với 19/14/2 ở R14. Đây là mức tăng nhỏ trên cùng hệ thống split đã định hướng các vòng lặp; có thể có experiment-level selection optimism. Không xem đây là bằng chứng tổng quát hóa.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_rho_similarity_r16.yaml](../../configs/quantum_histgb_nested_rho_similarity_r16.yaml)
- Mã: [quantum_features.py](../../src/abc_uwb_falltrack/quantum_features.py), [oof.py](../../src/abc_uwb_falltrack/oof.py), [run_nested_oof.py](../../scripts/run_nested_oof.py), [audit_nested_oof.py](../../scripts/audit_nested_oof.py)
- Artifact OOF và audit train-only bị ignore: `data/processed/quantum_diff_static_hybrid_nested_rho_similarity_r16_20261020/`
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_rho_similarity_r16.yaml`
- Audit train-only: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_rho_similarity_r16_20261020`

### Bước tiếp theo

Giữ R14 làm anchor đã có trước ablation và R16 làm train-only candidate có mean OOF cao nhất hiện ghi nhận. Không mở lại validation đã tiêu thụ; cần test package mới hoặc holdout độc lập chưa dùng để xác nhận mốc 0.70.


## Experiment — 2026-10-20 — quantum-static-hybrid-final-selection-r17

### Câu hỏi / giả thuyết

Sau R16, chọn một cấu hình cố định để có thể fit trên toàn bộ train và dự đoán holdout mới. So sánh bốn biến thể feature đã định trước bằng 5-fold GroupKFold OOF trên train; không dùng validation.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; 13 stream train, 5-fold GroupKFold theo `stream_id`, seed 2027. `validation_used=false`; `composition_file_used=false` được kiểm tra bắt buộc sau từng lượt. Gói hiện không có test stream. Validation cũ không được mở lại.

### Tiền xử lý và cửa sổ hóa

Sai phân CIR và static contrast causal EMA tau=8 s reset theo stream; block 5 frame/0,1 s. So sánh gamma 1,5/2,0 có hoặc không có normalized Hilbert–Schmidt temporal similarity. Dùng feature cache train-only đã tạo cho R16.

### Mô hình và cấu hình

HistGradientBoosting cố định như R16. Decoder Ising/DP được chọn trên OOF logits theo package evaluator tại theta 0,3 và tie-breaker được package mô tả. So sánh bốn feature candidates; kết quả cao nhất sinh [quantum_histgb_final_candidate_r17.yaml](../../configs/quantum_histgb_final_candidate_r17.yaml). Cấu hình chọn `gamma=2.0` cùng normalized rho similarity, transition cost 3, min duration 0,8 s và profile `adaptive_q25_q75_end_minus_0_1`.

### Seed ngẫu nhiên

2027 cho GroupKFold và readout.

### Metric theo protocol

Package evaluator cục bộ với semantics `--ignore-flagged`. Chưa xác nhận các metric/ranking này là ranking cuối cùng của challenge; xem [đặc tả challenge](../challenge_spec.md).

### Kết quả

Ứng viên được chọn có mean event F1 **0,8520**; F1@0,1/0,3/0,5 = **0,8930/0,8663/0,7968**. Tại theta 0,3: TP/FP/FN = **162/26/24**, median absolute start error 0,16 s, false alarms/hour 4,21. Ba biến thể còn lại đạt mean OOF 0,8396 (`gamma=1.5`), 0,8495 (`gamma=2.0`) và 0,8396 (`gamma=1.5` + normalized rho).

Đây là chọn cấu hình trên cùng 13 stream train sau nhiều vòng thử nghiệm. Điểm 0,8520 có selection optimism, không phải validation/test score và không xác nhận mục tiêu generalization ≥0,70.

### Phân tích lỗi

R17 có 26 false alarm và 24 miss tại theta 0,3 trên OOF; số liệu chỉ mô tả train-fold predictions. Không đọc `streams_composition.csv` trong lựa chọn cấu hình, nên không phân loại lỗi theo nguồn hoạt động ở lượt này.

### Artifact và lệnh tái lập

- Cấu hình chọn ứng viên: [quantum_histgb_final_selection_r17.yaml](../../configs/quantum_histgb_final_selection_r17.yaml)
- Cấu hình dự đoán holdout: [quantum_histgb_final_candidate_r17.yaml](../../configs/quantum_histgb_final_candidate_r17.yaml)
- Mã: [select_final_candidate.py](../../scripts/select_final_candidate.py), [predict_holdout.py](../../scripts/predict_holdout.py), [oof.py](../../src/abc_uwb_falltrack/oof.py)
- OOF artifacts và summary train-only bị ignore: `data/processed/quantum_diff_static_hybrid_final_selection_r17/`
- Lệnh: `./.venv/bin/python -u scripts/select_final_candidate.py --config configs/quantum_histgb_final_selection_r17.yaml`

### Bước tiếp theo

Giữ R17 làm cấu hình ứng viên train-only. Chưa tạo submission: thư mục test chưa có cục bộ và validation đã tiêu thụ. Cần holdout mới được phép dùng hoặc official evaluator để đo độc lập; nếu test chỉ có nhãn ẩn, gửi cấu hình đã khóa mà không tune theo kết quả.


## Experiment — 2026-10-20 — r17-validation-diagnostic-once

### Câu hỏi / giả thuyết

Chấm một lần cấu hình R17 đã khóa trên năm stream validation để xem OOF và validation khác nhau ra sao. Kết quả chỉ mang tính chẩn đoán; không chọn lại mô hình, feature, decoder hay ngưỡng.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; fit trên 13 stream train, dự đoán năm stream validation. Validation đã được dùng cho baseline R2 trước đó nên lượt này không phải đánh giá holdout hoàn toàn độc lập. Không đọc `streams_composition.csv`; nhãn validation chỉ được evaluator đọc sau khi đã tạo dự đoán.

### Tiền xử lý và cửa sổ hóa

Theo cấu hình R17: CIR 50 Hz, causal static contrast EMA tau=8 s, gamma=2.0, normalized rho similarity và pooling 5 frame/0,1 s.

### Mô hình và cấu hình

HistGradientBoosting, seed 2027; decoder R17 đã khóa (`bias=0.43045`, transition/entry/exit cost 3, min duration 0,8 s và profile `adaptive_q25_q75_end_minus_0_1`). Xem [cấu hình R17](../../configs/quantum_histgb_final_candidate_r17.yaml).

### Seed ngẫu nhiên

2027.

### Metric theo protocol

Evaluator package-local với `--ignore-flagged`; chưa xác nhận đây là công thức xếp hạng chính thức.

### Kết quả

Sinh 86 sự kiện dự đoán. Mean event F1 **0,6957**; F1@0,1/0,3/0,5/0,7 = **0,8116/0,7536/0,5217/0,2754**. Tại theta 0,3: TP/FP/FN = **52/19/15**, precision 0,7324, recall 0,7761, FA/h 9,27, median absolute start/end error 0,52/0,41 s. Kết quả dưới mục tiêu nội bộ mean F1 0,70 một chút; F1 giảm mạnh ở tIoU 0,5 và 0,7, cho thấy localization yếu trên validation.

Đối chiếu R2 trên cùng validation/evaluator: mean F1 0,6306, F1@0,3 0,6757 và FA/h 15,13. R17 cao hơn ở các chỉ số này, nhưng điểm validation đã được xem cho R2 trước đó; do đó không xem phép so sánh này là holdout hoàn toàn độc lập. OOF R17 0,8520 cao hơn đáng kể mean validation 0,6957, cho thấy generalization gap hoặc khác biệt phân phối giữa các stream; số stream ít nên chưa xác định được nguyên nhân.

### Phân tích lỗi

Ở theta 0,3, 19 false positive và 15 false negative; 52 sự kiện ground truth cùng 15 dự đoán bị ignore theo cờ chất lượng package. F1@0,5 giảm xuống 0,5217 và FA/h tăng lên 17,08. Không phân tích theo composition vì không dùng `streams_composition.csv`.

### Artifact và lệnh tái lập

- Dự đoán, metadata và metric được lưu dưới `data/processed/quantum_histgb_final_candidate_r17_validation_diagnostic_20261020/` (artifact cục bộ bị ignore).
- Tạo dự đoán: `./.venv/bin/python -u scripts/predict_holdout.py --config configs/quantum_histgb_final_candidate_r17.yaml --streams-dir data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --split-name val --output-dir data/processed/quantum_histgb_final_candidate_r17_validation_diagnostic_20261020 --validation-diagnostic`.
- Chấm metric: `./.venv/bin/python data/uwb_falltrack/abc2027_uwb_train_v1/src/abc2027/evaluate.py --gt data/uwb_falltrack/abc2027_uwb_train_v1/streams_events.csv --pred data/processed/quantum_histgb_final_candidate_r17_validation_diagnostic_20261020/val_predictions.csv --streams data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --ignore-flagged --out data/processed/quantum_histgb_final_candidate_r17_validation_diagnostic_20261020/val_metrics_package_local.json`.

### Bước tiếp theo

Không dùng lại validation để chỉnh mô hình hoặc ngưỡng. Giữ kết quả này như chẩn đoán hậu kiểm; chờ test/holdout mới hoặc official evaluator để xác nhận.


## Experiment — 2026-10-20 — quantum-reservoir-channel-r18

### Câu hỏi / giả thuyết

Một reset-channel quantum reservoir bốn qubit cho mỗi radar có bổ sung động học hữu ích ngoài các feature density-matrix overlap hiện có hay không? So sánh các biến thể R16 và phiên bản ghép reservoir trên cùng các split nested OOF.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; chỉ 13 stream `train_00`–`train_12`, 5-fold GroupKFold theo `stream_id`, inner 4-fold, seed 2027. Outer held-out stream IDs trùng chính xác với R16. Report xác nhận `validation_used=false` và `composition_file_used=false`; không đọc validation stream/labels hoặc `streams_composition.csv`.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, sai phân nhân quả, pooling 5 frame/0,1 s; static contrast EMA tau=8 s. Feature candidates gồm bốn biến thể R16 và bốn biến thể paired có thêm reservoir. Reservoir nhận xác suất 16 bin của motion state; dùng reset-channel memory tau=0,8 s, phase scale π/2 và ring ZZ entangler angle π/8. Trạng thái density được khởi tạo/reset về `|+>^4` và được giữ riêng theo radar, qua chunk nhưng reset giữa các stream. Readout gồm 48 Pauli expectation mỗi timestep (X/Y/Z và cyclic ZZ, 4 qubit/radar).

### Mô hình và cấu hình

HistGradientBoosting giữ nguyên thông số R16. Decoder Ising/DP được chọn trên inner OOF logits với nguyên search space R16. Không dùng kết quả outer fold để chọn thêm tham số.

### Seed ngẫu nhiên

2027 cho GroupKFold và readout.

### Metric theo protocol

Package evaluator cục bộ, ranking là mean event F1 tại tIoU 0,1/0,3/0,5; tie-breaker median absolute start error và false alarms/hour tại theta 0,3. Chưa xác nhận evaluator package-local là ranking chính thức của challenge.

### Kết quả

R18 đạt mean event F1 **0,8447**; F1@0,1/0,3/0,5 = **0,8901/0,8534/0,7906**. So với nested OOF R16 **0,8412** và **0,8796/0,8482/0,7958**, delta lần lượt là **+0,0035** mean, **+0,0105** tại 0,1, **+0,0052** tại 0,3 và **−0,0052** tại 0,5. Median absolute start error giảm **0,18 → 0,16 s**; false alarms/hour giảm **5,50 → 5,34**.

Inner selection chọn một biến thể có reservoir ở **4/5** outer folds. Fold mean F1 đổi lần lượt **−0,0308, −0,0045, +0,0232, +0,0236, 0,0000** so với R16 trên đúng các held-out streams; fold cuối giữ baseline gamma 1,5. Đây là cải thiện aggregate nhỏ và không đồng đều; F1@0,5 giảm nhẹ, còn nhiều vòng thử nghiệm đã dùng cùng train streams nên vẫn có selection optimism.

### Phân tích lỗi

Ở theta 0,3, R18 có TP/FP/FN **162/34/24**, median absolute start error **0,16 s**, false alarms/hour **5,34**. Không đọc composition nên không phân loại false positives theo nguồn hoạt động.

### Artifact và lệnh tái lập

- Config: [quantum_histgb_nested_quantum_reservoir_r18.yaml](../../configs/quantum_histgb_nested_quantum_reservoir_r18.yaml)
- OOF report, logits, predictions và cấu hình dùng: `data/processed/quantum_diff_static_hybrid_nested_quantum_reservoir_r18_20261020/` (artifact cục bộ bị ignore)
- Lệnh: `./.venv/bin/python scripts/run_nested_oof.py --config configs/quantum_histgb_nested_quantum_reservoir_r18.yaml`
- Baseline feature caches R16 được dùng lại; cache reservoir có version `quantum-reservoir-channel-v1` và lưu riêng theo từng candidate.

### Bước tiếp theo

Giữ reservoir ở chế độ opt-in để tiếp tục thử nghiệm train-only; chưa đổi cấu hình final đã khóa vì cải thiện nhỏ, tIoU 0,5 giảm và chưa có holdout độc lập. Inference path trong `predict_holdout.py` đã nhận flag/parameters và giữ mặc định cũ khi flag tắt. Cần test package mới, holdout mới hoặc official evaluator để xác nhận khả năng tổng quát.


## Experiment — 2026-10-20 — causal-semimarkov-r17-no-promote

### Câu hỏi / giả thuyết

Một causal semi-Markov filter với hidden event-age states và duration prior fit theo train scope có cải thiện temporal decoding so với Ising/DP R16 hay không? Giả thuyết không được ủng hộ: nested OOF hoàn tất thành công, nhưng kết quả thấp hơn R16 ở cả ba ngưỡng tIoU.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; chỉ 13 stream `train_00`–`train_12`, grouped nested OOF với 5 outer fold và 4 inner fold, seed 2027. Mỗi duration prior inner fold chỉ fit trên đúng các stream inner-train; outer prior chỉ fit trên outer-train. Report xác nhận `validation_used=false` và `composition_file_used=false`; không đọc validation streams/labels hoặc `streams_composition.csv`.

### Tiền xử lý và cửa sổ hóa

50 Hz CIR, causal static contrast EMA với tau=8 s, pooling 5 frame thành bước 0,1 s. Dùng bốn feature variant gamma 1,5/2,0, có hoặc không normalized rho similarity; cache feature train-only của pipeline hiện có được dùng lại.

### Mô hình và cấu hình

HistGradientBoosting theo cấu hình R16. Decoder tùy chọn `causal_semimarkov` dùng forward filter log-space trên event-age state đến 8 s; emission là margin của readout, event entry/exit có transition costs, hazard lấy từ lognormal duration prior robust fit trong fold scope, sau đó causal hysteresis tạo event intervals. Inner OOF chọn bias, entry/exit cost, duration penalty, threshold và min duration. Xem [cấu hình R17 semi-Markov](../../configs/quantum_histgb_nested_semimarkov_r17.yaml).

### Seed ngẫu nhiên

2027 cho grouped folds và readout.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`; ranking là mean event F1 tại tIoU 0,1/0,3/0,5. Đây là train-only nested OOF; chưa xem validation hoặc test.

### Kết quả

Run hoàn tất với nested OOF mean event F1 **0,6753**; F1@0,1/0,3/0,5 = **0,7532/0,7013/0,5714**. Outer-fold mean ± population std lần lượt là **0,7562 ± 0,0472**, **0,7001 ± 0,0407**, **0,5581 ± 0,1095**; min/max là **0,6761–0,8214**, **0,6479–0,7500**, **0,4231–0,7436**. Tại theta 0,3: TP/FP/FN **108/14/78**, precision **0,8852**, recall **0,5806**, false alarms/hour **2,26**, median absolute start error **0,21 s**.

So với nested OOF R16 trên cùng 13 train streams (**0,8412** mean; F1@0,1/0,3/0,5 **0,8796/0,8482/0,7958**), R17 thấp hơn tương ứng khoảng **0,1659** mean và **0,1263/0,1469/0,2244** tại ba ngưỡng. Vì vậy thí nghiệm không đạt tiêu chí cải thiện và không được promote; `scripts/predict_holdout.py` được giữ nguyên.

Các cờ leakage trong report đều true: outer/inner folds partition đúng train scope, mỗi train stream xuất hiện đúng một lần trong outer OOF, predictions chỉ tham chiếu train IDs, inner duration priors chỉ fit inner-train và outer duration priors chỉ fit outer-train.

### Phân tích lỗi

Recall 0,3 là 0,5806 với 78 false negatives; localization nghiêm ngặt yếu hơn rõ rệt tại tIoU 0,5. Outer-fold std ở tIoU 0,5 là 0,1095, cho thấy độ dao động đáng kể trên số stream nhỏ. Tham số decoder được chọn khác nhau giữa các fold, nên chưa có bằng chứng duration prior/event-age model ổn định hoặc giúp tổng quát hóa; không dùng kết quả này để chọn cấu hình inference.

### Artifact và lệnh tái lập

- Config: [quantum_histgb_nested_semimarkov_r17.yaml](../../configs/quantum_histgb_nested_semimarkov_r17.yaml)
- Report, `config_used.json`, train-only predictions và train-only logits nằm trong `data/processed/quantum_diff_static_hybrid_nested_semimarkov_r17_20261020/` (artifact cục bộ bị ignore).
- Lệnh: `./.venv/bin/python scripts/run_nested_oof.py --config configs/quantum_histgb_nested_semimarkov_r17.yaml`

### Bước tiếp theo

Không promote decoder và không thêm đường inference vào `predict_holdout.py`. Giữ config/artifact làm kết quả train-only âm tính; mọi thử nghiệm tiếp theo cần dùng nested grouped OOF mới và giữ duration-prior fit trong đúng fold scope.


## Experiment — 2026-10-20 — multiscale-static-floor-first-r19

### Câu hỏi / giả thuyết

Static-background time constant tau khác 8 s có thể cải thiện tIoU nghiêm ngặt hoặc độ cân bằng giữa các ngưỡng không? Thay đổi thứ hai là chọn feature/decoder bên trong inner OOF theo F1 thấp nhất trong ba ngưỡng, sau đó theo spread nhỏ hơn rồi mean F1, để khớp mục tiêu nội bộ.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; chỉ 13 stream train, nested 5 outer × 4 inner GroupKFold theo `stream_id`, seed 2028. Đây là split seed mới so với R16/R18. `validation_used=false`, `composition_file_used=false`; evaluator package-local dùng `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, sai phân causal và static contrast EMA causal, reset theo stream; block 5 frame/0,1 s. Inner search 12 feature variants từ static tau 4/8/16 s, gamma 1,5/2,0 và normalized rho similarity tắt/bật. Mỗi stream có feature cache riêng cho từng variant.

### Mô hình và cấu hình

HistGradientBoosting giữ tham số R16. Decoder Ising/DP và duration profiles giữ search space R16. Inner selection objective `minimum_f1_then_spread`: tối đa F1 nhỏ nhất ở tIoU 0,1/0,3/0,5; tie-break giảm spread, rồi tối đa mean F1, sau đó dùng median start error và false alarms/hour. Thay đổi objective chỉ áp dụng trong config R19; package evaluator vẫn báo metric chuẩn của package.

### Seed ngẫu nhiên

2028 cho GroupKFold và readout; inner seed thay đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với semantics `--ignore-flagged`; ranking package-local là mean event F1 tại tIoU 0,1/0,3/0,5. Công thức challenge chính thức chưa được xác nhận.

### Kết quả

Nested OOF đạt mean event F1 **0,8196**; F1@0,1/0,3/0,5 = **0,8640/0,8427/0,7520**, pooled spread **0,1120**. Tại theta 0,3: TP/FP/FN **158/31/28**, median absolute start/end error **0,18/0,22 s**, false alarms/hour **5,01**. Outer-fold mean ± population std: **0,8566 ± 0,0428 / 0,8262 ± 0,0577 / 0,7574 ± 0,0659**; min/max tại tIoU 0,5 là **0,6931–0,8667**. Vì pooled spread lớn hơn 0,10 và một outer fold dưới 0,75 tại mọi ngưỡng, kết quả chưa đạt mục tiêu ổn định.

Inner selection chọn tau 8 s ở 2/5 outer folds, tau 16 s ở 3/5; tau 4 s không được chọn. R19 chỉ là một vòng development train-only, nên không kết luận tau 16 tốt hơn nếu chưa có so sánh cùng split với baseline tau 8.

### Phân tích lỗi

Ở aggregate, yếu nhất vẫn là tIoU 0,5, cách tIoU 0,1 **0,1120**. Kết quả tau khác nhau đáng kể theo fold, với fold giữ lại `train_10`/`train_12` có mean F1 **0,7381**. Mức điểm aggregate R19 thấp hơn R16 nhưng dùng seed khác, nên cần chạy baseline tau 8 trên đúng seed 2028 và cùng selection objective trước khi quy thay đổi cho feature.

### Artifact và lệnh tái lập

- Config: [quantum_histgb_nested_multiscale_static_r19.yaml](../../configs/quantum_histgb_nested_multiscale_static_r19.yaml)
- Report, config dùng, logits, predictions train-only và feature cache: `data/processed/quantum_diff_static_hybrid_nested_multiscale_static_r19_20261020/` (artifact cục bộ bị ignore)
- Lệnh: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_multiscale_static_r19.yaml`
- Leakage checks trong report đều true; report không tham chiếu validation hoặc composition.

### Bước tiếp theo

Chạy baseline R20 tau 8 s trên seed 2028 với cùng inner selection objective để ghép cặp so sánh. Không dùng validation; không promote R19.


## Experiment — 2026-10-20 — static-tau8-floor-first-seed2028-r20

### Câu hỏi / giả thuyết

Baseline tau 8 s trên đúng split seed 2028 và cùng tiêu chí inner selection có ổn định hơn vòng tau sweep R19 không? Đây là đối chứng để tách hiệu ứng static tau khỏi khác split và khác tiêu chí lựa chọn.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; chỉ 13 stream train, nested 5 outer × 4 inner GroupKFold theo `stream_id`, seed 2028. Cùng outer/inner splits, evaluator và `minimum_f1_then_spread` objective như R19. `validation_used=false`, `composition_file_used=false`; dùng `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, sai phân causal, static contrast EMA causal tau cố định 8 s, reset theo stream; block 5 frame/0,1 s. Inner search 4 feature variants từ gamma 1,5/2,0 và normalized rho similarity tắt/bật. Tái sử dụng đúng feature caches của R19 tau 8 s.

### Mô hình và cấu hình

HistGradientBoosting theo R16; Ising/DP decoder search theo R16. Inner selection ưu tiên F1 thấp nhất tại tIoU 0,1/0,3/0,5, tie-break theo spread, mean F1, median start error và false alarms/hour.

### Seed ngẫu nhiên

2028 cho GroupKFold và readout; inner seed thay đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với semantics `--ignore-flagged`; đây là metric package-local và chưa được xác nhận là ranking chính thức.

### Kết quả

Nested OOF mean event F1 **0,8191**; F1@0,1/0,3/0,5 = **0,8617/0,8404/0,7553**, pooled spread **0,1064**. Tại theta 0,3: TP/FP/FN **158/32/28**, median absolute start/end error **0,18/0,21 s**, false alarms/hour **5,18**. Outer-fold mean ± population std lần lượt **0,8552 ± 0,0267 / 0,8270 ± 0,0457 / 0,7624 ± 0,0559**; min/max tại tIoU 0,5 là **0,7059–0,8525**.

So với R19 cùng split, R20 mean F1 thấp hơn **0,0004** và F1@0,1/0,3 thấp hơn **0,0023/0,0022**; F1@0,5 cao hơn **0,0033**, pooled spread nhỏ hơn **0,0053**. Fold-wise kết quả trái chiều; tau 16 chỉ tăng rõ ở một fold. R19 tau sweep không tạo cải thiện nhất quán; giữ tau 8 làm cấu hình feature cho vòng tiếp theo.

### Phân tích lỗi

Pooled F1@0,5 vừa trên 0,75 nhưng spread **0,1064** vượt mục tiêu tạm 0,10; outer-fold min ở tIoU 0,5 là 0,7059. Inner selection objective chưa đảm bảo ngưỡng ngoài fold nên không xem kết quả là đã đạt mục tiêu ổn định.

### Artifact và lệnh tái lập

- Config: [quantum_histgb_nested_rho_similarity_r20_seed2028.yaml](../../configs/quantum_histgb_nested_rho_similarity_r20_seed2028.yaml)
- Report, config dùng, logits, predictions train-only và tái sử dụng feature cache: `data/processed/quantum_diff_static_hybrid_nested_rho_similarity_r20_seed2028_20261020/` (artifact cục bộ bị ignore)
- Lệnh: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_rho_similarity_r20_seed2028.yaml`
- Leakage checks trong report đều true; validation và composition không được dùng.

### Bước tiếp theo

Thử trọng số sample theo train stream để giảm chi phối của stream dài, chọn exponent hoàn toàn trong inner OOF. Giữ validation đóng.


## Experiment — 2026-10-20 — stream-balanced-readout-r21

### Câu hỏi / giả thuyết

Các stream dài có thể chi phối số frame huấn luyện và làm readout kém ổn định theo stream. Thử sample weighting theo stream với exponent 0/0,5/1,0, được chọn trong inner OOF, có thể cải thiện generalization và các metric tIoU.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; 13 stream train, nested 5 outer × 4 inner GroupKFold theo `stream_id`, seed 2028. Cùng splits và `minimum_f1_then_spread` objective như R20. Validation/composition không tham gia training hoặc selection; `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, sai phân causal, static contrast EMA tau 8 s reset theo stream, block 5 frame/0,1 s. Dùng 4 feature variant R20: gamma 1,5/2,0 × normalized rho similarity tắt/bật; tái sử dụng R20 feature cache train-only.

### Mô hình và cấu hình

HistGradientBoosting R16. Mỗi inner fit thử sample weight per-frame, partial stream balance (exponent 0,5) và equal stream balance (exponent 1,0). Stream factor được tính chỉ từ số frame usable trong các stream của fit scope; positive class weight 5 vẫn giữ nguyên. Decoder search theo R20. Selection ưu tiên F1 thấp nhất tại ba ngưỡng, tiếp theo spread và mean.

### Seed ngẫu nhiên

2028 cho GroupKFold và readout; inner seed đổi theo outer fold.

### Metric theo protocol

Package evaluator local với `--ignore-flagged`; nested OOF train-only, chưa xác nhận là ranking chính thức.

### Kết quả

Nested OOF mean event F1 **0,8400**; F1@0,1/0,3/0,5 = **0,8847/0,8633/0,7721**, spread **0,1126**. So với R20, mean tăng **0,0209**; F1 tăng **0,0230/0,0228/0,0168**. Tại theta 0,3: TP/FP/FN **161/26/25**, median absolute start/end error **0,18/0,20 s**, false alarms/hour **4,21** (R20: 158/32/28, 0,18/0,21 s, 5,18). Inner selection chọn equal stream weighting ở **4/5** folds. Outer-fold mean ± population std: **0,8949 ± 0,0440 / 0,8685 ± 0,0249 / 0,7956 ± 0,0697**; min F1@0,5 **0,7143**.

Aggregate thresholds đều trên 0,75 nhưng spread pooled vẫn lớn hơn mục tiêu tạm 0,10, và hiệu ứng outer fold trái chiều. Không promote làm candidate đã khóa.

### Phân tích lỗi

Train-only boundary audit: tại tIoU 0,5 median signed start error **−0,08 s** (dự đoán sớm), end error **+0,06 s** (dự đoán muộn), duration ratio median **1,093**. Tín hiệu hỗ trợ thử hiệu chỉnh ranh giới causal trong inner OOF. Audit gán 14 false alarms vào segment fall, 11 ADL và 1 other; không dùng thông tin đó trong feature/readout selection.

### Artifact và lệnh tái lập

- Config: [quantum_histgb_nested_stream_balance_r21_seed2028.yaml](../../configs/quantum_histgb_nested_stream_balance_r21_seed2028.yaml)
- Report, train-only logits/predictions, config, feature cache và [nested_error_audit_train_only.json](../../data/processed/quantum_diff_static_hybrid_nested_stream_balance_r21_seed2028_20261020/nested_error_audit_train_only.json): `data/processed/quantum_diff_static_hybrid_nested_stream_balance_r21_seed2028_20261020/` (artifact cục bộ bị ignore)
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_stream_balance_r21_seed2028.yaml`
- Train-only audit: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_stream_balance_r21_seed2028_20261020`
- Toàn bộ leakage checks trong report đều true; validation và composition không dùng cho training/selection.

### Bước tiếp theo

Thử boundary-profile trim được chọn bên trong inner OOF để khắc phục start hơi sớm và end hơi muộn. Giữ data scope train-only; không chạy lại validation.


## Experiment — 2026-10-20 — adaptive-boundary-trim-r22

### Câu hỏi / giả thuyết

Train-only audit R21 cho thấy start dự đoán sớm khoảng 0,08 s, end muộn khoảng 0,06 s và predicted duration dài hơn median 9,3%. Các adaptive profile dịch start muộn và end sớm 0,1/0,2 s có thể cải thiện F1 ở tIoU 0,5 mà vẫn giữ ba ngưỡng cân bằng không?

### Phiên bản dữ liệu và split

13 stream train trong package cục bộ; nested 5 outer × 4 inner GroupKFold theo `stream_id`, seed 2028. Dùng đúng split, feature candidates, stream-weight candidates và inner objective `minimum_f1_then_spread` của R21. Validation/composition không được dùng; evaluator package-local `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

50 Hz CIR, sai phân causal, static contrast EMA tau 8 s, gamma 1,5/2,0, normalized rho similarity tắt/bật, bước 0,1 s. Tái sử dụng feature caches train-only R20.

### Mô hình và cấu hình

HistGradientBoosting R16, sample weights exponent 0/0,5/1,0 theo stream được chọn trong inner OOF, Ising/DP decoder. Giữ bốn duration profiles R21 và thêm hai adaptive profile: trim cả start/end 0,1 s và 0,2 s theo ba duration bin. Inner OOF chọn readout, decoder và profile trước khi chấm outer fold.

### Seed ngẫu nhiên

2028 cho GroupKFold và readout; inner seed đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`, nested OOF train-only; chưa xác nhận là metric xếp hạng chính thức.

### Kết quả

R22 cho **đúng cùng kết quả outer OOF như R21**: mean event F1 **0,8400**; F1@0,1/0,3/0,5 = **0,8847/0,8633/0,7721**, spread **0,1126**. Cả năm fold chọn đúng cùng readout/decoder/profile như R21; hai profile trim mới không được chọn. Vì vậy boundary trim cố định/adaptive đơn giản không cải thiện kết quả trong nested selection.

### Phân tích lỗi

F1@0,5 aggregate vẫn chỉ 0,7721 và spread vẫn vượt mức tạm 0,10; outer-fold min F1@0,5 là 0,7143. Không có bằng chứng các profile trim mới tổng quát hóa tốt hơn. Leakage checks của report đều true, `validation_used=false`, `composition_file_used=false`.

### Artifact và lệnh tái lập

- Config: [quantum_histgb_nested_boundary_trim_r22_seed2028.yaml](../../configs/quantum_histgb_nested_boundary_trim_r22_seed2028.yaml)
- Report, train-only logits/predictions, config và cache: `data/processed/quantum_diff_static_hybrid_nested_boundary_trim_r22_seed2028_20261020/` (artifact cục bộ bị ignore)
- Lệnh: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_boundary_trim_r22_seed2028.yaml`
- Tất cả leakage checks outer/inner split và train-only predictions đều true.

### Bước tiếp theo

Không promote profile trim R22. Vì threshold decoder không dịch biên đủ nhất quán, thử readout riêng dự đoán onset/offset từ nhãn train, với mọi boundary head và tham số được fit/chọn trong đúng inner fold. Không mở validation.


## Experiment — 2026-10-20 — stream-balanced-final-validation-diagnostic-r23

### Câu hỏi / giả thuyết

Sau khi thêm stream weighting, mô hình fit trên toàn bộ train có chuyển được cải thiện train OOF sang validation cục bộ hay không? Đây là lượt chẩn đoán theo yêu cầu; không dùng điểm validation để chọn lại mô hình.

### Phiên bản dữ liệu và split

Chọn feature/decoder bằng 5-fold GroupKFold OOF trên 13 stream train, seed 2028; readout dùng stream-weight exponent 1,0 theo lựa chọn train-only R21. Fit cuối trên cả 13 stream train, dự đoán 5 stream validation rồi mới chấm bằng package evaluator `--ignore-flagged`. Validation đã được dùng cho R2 và R17 nên kết quả R23 không phải holdout độc lập.

### Tiền xử lý và cửa sổ hóa

50 Hz CIR; sai phân nhân quả; static contrast EMA tau 8 s, gamma 2,0; block 5 frame/0,1 s. Không fit scaler/PCA hay tham số thống kê từ validation; EMA được cập nhật causal riêng trong từng stream. Không đọc `streams_composition.csv` khi dự đoán.

### Mô hình và cấu hình

HistGradientBoosting với positive weight 5 và stream-weight exponent 1,0; feature variant gamma 2,0 được chọn bằng mean F1 train OOF. Decoder Ising/DP được khóa từ train OOF. Điểm OOF của candidate được chọn là 0,8436, F1@0,1/0,3/0,5 = 0,8954/0,8686/0,7668; spread 0,1287. Vì vậy điểm train OOF này không đáp ứng điều kiện cân bằng tạm thời spread ≤0,10.

### Seed ngẫu nhiên

2028 cho GroupKFold và readout.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`; metric này chưa được xác nhận là công thức xếp hạng chính thức. Nhãn validation chỉ được evaluator đọc sau khi dự đoán đã cố định.

### Kết quả

Validation diagnostic đạt mean event F1 **0,6857**; F1@0,1/0,3/0,5 = **0,8143/0,7143/0,5286**, spread **0,2857**. Tại theta 0,3: TP/FP/FN **50/23/17**, precision **0,6849**, recall **0,7463**, false alarms/hour **11,23**, median absolute start/end error **0,41/0,41 s**. Mô hình chưa đạt mục tiêu nội bộ F1 ≥0,75 ở cả ba ngưỡng và điểm ở ngưỡng 0,5 còn thấp. So với validation R17 (mean 0,6957; F1@0,1/0,3/0,5 = 0,8116/0,7536/0,5217), R23 nhỉnh hơn ở 0,1 và 0,5 nhưng thấp hơn mean; không kết luận cải thiện vì validation đã được dùng trước đó.

### Phân tích lỗi

Metric giảm rõ khi tIoU tăng: F1 từ 0,8143 tại 0,1 xuống 0,5286 tại 0,5. Median absolute sai số start/end ở tIoU 0,3 đều 0,41 s; điều này phù hợp với việc định vị ranh giới còn chưa chính xác, nhưng không dùng các nhãn validation để tinh chỉnh.

### Artifact và lệnh tái lập

- Cấu hình chọn train-only: [quantum_histgb_final_selection_stream_balance_r23.yaml](../../configs/quantum_histgb_final_selection_stream_balance_r23.yaml)
- Cấu hình candidate đã khóa: [quantum_histgb_final_candidate_stream_balance_r23.yaml](../../configs/quantum_histgb_final_candidate_stream_balance_r23.yaml)
- OOF selection summary: `data/processed/quantum_diff_static_hybrid_final_selection_stream_balance_r23_20261020/candidate_selection_summary.json`
- Dự đoán, model và evaluator report cục bộ: `data/processed/quantum_diff_static_stream_balance_validation_diagnostic_r23_20261020/`
- Chọn trên train: `./.venv/bin/python -u scripts/select_final_candidate.py --config configs/quantum_histgb_final_selection_stream_balance_r23.yaml`
- Fit trên train rồi dự đoán validation: `./.venv/bin/python -u scripts/predict_holdout.py --config configs/quantum_histgb_final_candidate_stream_balance_r23.yaml --streams-dir data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --split-name val --validation-diagnostic --output-dir data/processed/quantum_diff_static_stream_balance_validation_diagnostic_r23_20261020`
- Chấm một lần: `PYTHONPATH=data/uwb_falltrack/abc2027_uwb_train_v1/src .venv/bin/python -m abc2027.evaluate --gt data/uwb_falltrack/abc2027_uwb_train_v1/streams_events.csv --pred data/processed/quantum_diff_static_stream_balance_validation_diagnostic_r23_20261020/val_predictions.csv --streams data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --ignore-flagged --out data/processed/quantum_diff_static_stream_balance_validation_diagnostic_r23_20261020/val_metrics_package_local.json`
- Prediction metadata xác nhận đủ 13 train stream, `prediction_labels_read=false`, `composition_file_used_for_prediction=false`; evaluator đọc nhãn chỉ để tính metric sau dự đoán.

### Bước tiếp theo

Không dùng validation R23 để chọn mô hình, ngưỡng hoặc feature. Tiếp tục nested OOF train-only cho readout onset/offset; chờ test hoặc holdout mới để đánh giá độc lập.


## Experiment — 2026-10-20 — nested-boundary-readout-r24-seed2028

### Câu hỏi / giả thuyết

Hai readout onset/offset fit từ nhãn train và một bước hiệu chỉnh ranh giới causal được chọn trên inner OOF có giúp F1 ở tIoU 0,5 tăng mà không làm các ngưỡng tIoU mất cân bằng không? Kết quả không ủng hộ giữ hai head: cả năm outer fold chọn bán kính hiệu chỉnh bằng 0.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; nested 5 outer × 4 inner GroupKFold theo `stream_id`, seed 2028. Tất cả feature/readout/decoder/radius được chọn trong inner OOF của outer-train. Không đọc validation; không dùng `streams_composition.csv` trong fit/selection. Evaluator package-local dùng `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, sai phân nhân quả, block 5 frame/0,1 s, static contrast EMA tau 8 s. Inner selection so sánh gamma 1,5/2,0 và normalized rho temporal similarity bật/tắt. EMA reset riêng từng stream.

### Mô hình và cấu hình

Event readout HistGradientBoosting như R21/R22; sample-weight stream exponent 0/0,5/1,0 được chọn trong inner OOF. Hai boundary head HistGradientBoosting riêng cho onset/offset dùng positive band 0,3 s, positive weight 20, 7 leaf, min leaf 30; fit đúng inner-train/outer-train scope. Refinement chỉ dịch start muộn trong phạm vi tới `t_alarm`, dịch end sớm trong phạm vi dự đoán; không thay `t_alarm`. Inner search chọn trong radius 0–0,5 s, cùng Ising/DP và hai duration profile. Mục tiêu chọn `minimum_f1_then_spread`.

### Seed ngẫu nhiên

GroupKFold/readout seed 2028; inner seed đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`, trên nested outer OOF train-only; metric chưa được xác nhận là metric xếp hạng chính thức.

### Kết quả

Nested OOF mean event F1 **0,8436**; F1@0,1/0,3/0,5 = **0,8847/0,8686/0,7775**, spread **0,1072**. Ba pooled F1 đều vượt 0,75 nhưng spread vượt tiêu chí phát triển tạm **0,10**; outer-fold minimum F1@0,5 là **0,7143**, nên chưa xem là đạt ổn định. Các outer fold lần lượt đạt mean 0,8471/0,7959/0,8552/0,8556/0,9247.

Cả năm fold chọn boundary radius **0,0 s**. Vì vậy hai head không cải thiện inner selection và không được promote. So với R22, search space duration profile cũng đã thay đổi nên không quy phần tăng nhỏ của aggregate F1 cho boundary head.

### Phân tích lỗi

Train-only audit tại tIoU 0,3: TP/FP/FN **162/25/24**, median absolute start/end error **0,18/0,20 s**; 13 false alarms phủ lên segment fall, 11 lên ADL và 1 lên other. Tại tIoU 0,5, median signed start error là **−0,08 s**, end error **+0,08 s**; predicted/ground-truth duration ratio median **1,1165**. Đây là bằng chứng để thử co interval theo tỷ lệ trong inner OOF, không phải tham số được áp trực tiếp từ toàn bộ OOF.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_boundary_readout_r24_seed2028.yaml](../../configs/quantum_histgb_nested_boundary_readout_r24_seed2028.yaml)
- Report, dự đoán/logits OOF và train-only error audit: `data/processed/quantum_diff_static_hybrid_nested_boundary_readout_r24_seed2028_20261020/`
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_boundary_readout_r24_seed2028.yaml`
- Audit scope/leakage và lỗi train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_boundary_readout_r24_seed2028_20261020`
- Report runtime và audit xác nhận validation không được dùng; fit/scores boundary khớp scope inner/outer; composition chỉ được đọc trong train error audit.

### Bước tiếp theo

Không dùng boundary heads R24. Thử hiệu chỉnh tỷ lệ độ dài interval được chọn riêng trên inner OOF; chỉ giữ nếu cải thiện outer OOF mà đạt các tiêu chí F1 và spread. Validation hiện có đã tiêu thụ; không chấm lại.


## Experiment — 2026-10-20 — nested-duration-contraction-r25-seed2028

### Câu hỏi / giả thuyết

OOF audit R24 cho thấy interval dự đoán thường dài hơn ground truth. Co đối xứng quanh tâm khoảng dự đoán theo hệ số 0,80–1,00, chọn trong inner OOF, có làm F1@0,5 tăng và giảm spread giữa ba ngưỡng không? Không: mọi outer fold chọn scale 1,00.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; nested 5 outer × 4 inner GroupKFold theo `stream_id`, seed 2028. Không dùng validation/composition khi fit hoặc selection; package evaluator `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

Giữ nguyên R24: CIR 50 Hz, sai phân causal, block 5 frame/0,1 s, EMA tau 8 s; gamma 1,5/2,0 và normalized rho temporal similarity được lựa chọn trong inner OOF.

### Mô hình và cấu hình

HistGradientBoosting và Ising/DP; stream-weight exponent 0/0,5/1,0 và decoder được chọn trong inner OOF. Sau khi dựng interval, áp hệ số contraction quanh trung điểm; interval không được ngắn hơn `min_duration_s`, `t_alarm` giữ nguyên. Ứng viên scale: 0,80/0,85/0,90/0,95/1,00.

### Seed ngẫu nhiên

GroupKFold/readout seed 2028; inner seed đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`, nested outer OOF train-only; chưa được xác nhận là metric xếp hạng chính thức.

### Kết quả

Tất cả năm outer fold chọn **scale 1,00**. Metric trùng R24: mean event F1 **0,8436**; F1@0,1/0,3/0,5 = **0,8847/0,8686/0,7775**, spread **0,1072**. Outer-fold minimum F1@0,5 là **0,7143**. Không có cải thiện từ contraction.

### Phân tích lỗi

Train-only error audit cho cùng kết quả R24: theta 0,3 có TP/FP/FN **162/25/24**, median absolute start/end error **0,18/0,20 s**; median duration ratio ở theta 0,5 **1,1165**. Dù median interval hơi dài, co đối xứng không giúp qua nested selection; sai số khác nhau giữa stream/event nên hệ số chung không đủ.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_duration_contraction_r25_seed2028.yaml](../../configs/quantum_histgb_nested_duration_contraction_r25_seed2028.yaml)
- Report, logits/predictions OOF và train-only error audit: `data/processed/quantum_diff_static_hybrid_nested_duration_contraction_r25_seed2028_20261020/`
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_duration_contraction_r25_seed2028.yaml`
- Audit leakage/scope và lỗi train: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_duration_contraction_r25_seed2028_20261020`
- Report xác nhận `validation_used=false`, mọi outer/inner folds là train streams và scale được chọn từ cấu hình inner OOF.

### Bước tiếp theo

Không promote contraction. R18 từng cho pooled F1@0,5 cao hơn nhưng dùng objective mean-F1 và seed khác; chạy lại các feature quantum reservoir cùng stream weighting dưới objective `minimum_f1_then_spread` trong nested OOF train-only. Không dùng lại validation.


## Experiment — 2026-10-20 — nested-reservoir-balance-r26-seed2028

### Câu hỏi / giả thuyết

Quantum reservoir R18 có giúp F1 thấp nhất giữa các ngưỡng khi kết hợp stream weighting và selection ưu tiên minimum F1/spread không? Kết quả tăng mean nhẹ nhưng spread xấu hơn R25; chưa promote.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; nested 5 outer × 4 inner GroupKFold trên 13 stream train, seed 2028. Không dùng validation hay composition trong fit/selection; evaluator package-local `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, sai phân nhân quả, pooling 5 frame/0,1 s, EMA tau 8 s. Feature candidates ghép gamma 1,5/2,0, normalized rho temporal similarity bật/tắt và reset-channel quantum reservoir bật/tắt; reservoir memory tau 0,8 s, phase scale π/2, ring ZZ angle π/8. Cache feature R16/R18 được tái dùng.

### Mô hình và cấu hình

HistGradientBoosting; stream-weight exponent 0/0,5/1,0; Ising/DP decoder. Feature/readout/decoder chỉ được chọn từ inner OOF bằng `minimum_f1_then_spread`. Có 8 feature variants × 3 readout weight candidates; không bật boundary heads hay duration contraction.

### Seed ngẫu nhiên

GroupKFold/readout seed 2028; inner seed đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`, nested outer OOF train-only; metric chưa được xác nhận là ranking chính thức.

### Kết quả

Nested OOF mean F1 **0,8493**; F1@0,1/0,3/0,5 = **0,8989/0,8723/0,7766**, spread **0,1223**. So với R25 cùng seed, F1@0,1 và 0,3 tăng nhưng F1@0,5 gần như giữ nguyên; độ vênh tăng. Outer-fold minimum F1@0,5 là **0,7200**. Reservoir được chọn trong 3/5 outer fold; các fold khác giữ baseline, nên cải thiện chưa ổn định.

### Phân tích lỗi

Train-only audit tại theta 0,3: TP/FP/FN **164/26/22**, median absolute start/end error **0,18/0,19 s**; 15 false alarms phủ lên segment fall, 10 lên ADL và 1 lên other. Tại theta 0,5 median duration ratio là **1,076**. Biên thời gian cải thiện nhẹ so với R25 nhưng không tăng F1@0,5 đủ để cân bằng tIoU.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_reservoir_balance_r26_seed2028.yaml](../../configs/quantum_histgb_nested_reservoir_balance_r26_seed2028.yaml)
- Report, logits/predictions OOF và train-only audit: `data/processed/quantum_diff_static_hybrid_nested_reservoir_balance_r26_seed2028_20261020/`
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_reservoir_balance_r26_seed2028.yaml`
- Audit: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_reservoir_balance_r26_seed2028_20261020`
- Runtime report và audit xác nhận các split/logit/prediction chỉ chứa train; `validation_used=false`, `composition_file_used=false`.

### Bước tiếp theo

Không promote R26 cho mục tiêu cân bằng. Thay ranking `minimum_f1_then_spread` bằng objective ràng buộc trực tiếp: ưu tiên candidate đạt F1 tối thiểu 0,75 ở cả ba tIoU và spread ≤0,10; nếu không có candidate thỏa, tối đa hóa F1 thấp nhất rồi giảm spread. Chỉ áp dụng chọn lựa bên trong inner OOF.


## Experiment — 2026-10-20 — r26-consensus-validation-diagnostic

### Câu hỏi / giả thuyết

Một cấu hình đồng thuận chọn hoàn toàn từ inner OOF của R26, sau khi fit readout trên toàn bộ train, cho kết quả chẩn đoán nào trên validation đã dùng trước đó? Không sử dụng nhãn validation để chọn hoặc sửa mô hình.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; fit trên 13 stream train, dự đoán 5 stream validation. Candidate `gamma_1_5_qreservoir/partial_stream_balance` được chọn từ tổng hợp train-only: trong 5 outer-train inner selections, 4 đáp ứng sàn F1 0,75 và spread ≤0,10; trung bình minimum F1 là 0,7718, spread trung bình 0,0870. Decoder được tổng hợp bằng median/mode của 5 lựa chọn inner OOF. Validation chỉ được evaluator đọc sau khi CSV dự đoán đã khóa.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, sai phân nhân quả, pooling 5 frame/0,1 s, EMA tau 8 s và gamma 1,5. Quantum reservoir reset-channel với memory tau 0,8 s, phase scale π/2, ring ZZ angle π/8. Không đọc `streams_composition.csv` trong dự đoán.

### Mô hình và cấu hình

HistGradientBoosting, stream-weight exponent 0,5; Ising/DP với bias median 0,2664, transition/entry/exit cost 3,0, minimum duration 1,2 s. Duration profile adaptive q25/q75, tham số profile và ngưỡng duration lấy bằng mode/median từ lựa chọn inner OOF R26. Cấu hình: [quantum_histgb_final_candidate_r26_consensus.yaml](../../configs/quantum_histgb_final_candidate_r26_consensus.yaml).

### Seed ngẫu nhiên

2028 cho CV/readout.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`; chưa được xác nhận là metric xếp hạng chính thức.

### Kết quả

Validation diagnostic đạt mean event F1 **0,7121**; F1@0,1/0,3/0,5 = **0,8333/0,7576/0,5455**, spread **0,2879**. Tại theta 0,3: TP/FP/FN **50/15/17**, precision **0,7692**, recall **0,7463**, false alarms/hour **7,32**, median absolute start/end error **0,43/0,36 s**. So với diagnostic R23, mean F1 tăng từ 0,6857 lên 0,7121; tuy vậy F1@0,5 vẫn thấp và spread còn lớn. Chưa đạt mục tiêu tạm F1 ≥0,75 tại cả ba ngưỡng và spread ≤0,10.

### Phân tích lỗi

Điểm giảm chủ yếu khi yêu cầu overlap chặt hơn: F1@0,5 là 0,5455, thấp hơn 0,7576 tại tIoU 0,3. Đây là dấu hiệu lỗi định vị ranh giới/sự kiện còn đáng kể; validation đã bị dùng ở các vòng trước nên không dùng kết quả này để tinh chỉnh.

### Artifact và lệnh tái lập

- Dự đoán, model, metadata và evaluator report: `data/processed/quantum_diff_static_hybrid_validation_diagnostic_r26_consensus_20261020/`
- Fit trên train rồi dự đoán validation: `./.venv/bin/python -u scripts/predict_holdout.py --config configs/quantum_histgb_final_candidate_r26_consensus.yaml --streams-dir data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --split-name val --validation-diagnostic`
- Chấm sau khi CSV đã khóa: `PYTHONPATH=data/uwb_falltrack/abc2027_uwb_train_v1/src ./.venv/bin/python -m abc2027.evaluate --gt data/uwb_falltrack/abc2027_uwb_train_v1/streams_events.csv --pred data/processed/quantum_diff_static_hybrid_validation_diagnostic_r26_consensus_20261020/val_predictions.csv --streams data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --ignore-flagged --out data/processed/quantum_diff_static_hybrid_validation_diagnostic_r26_consensus_20261020/val_metrics_package_local.json`
- Metadata xác nhận fit trên 13 train streams; `prediction_labels_read=false`, `composition_file_used_for_prediction=false`; evaluator format validator cũng không đọc nhãn.

### Bước tiếp theo

Không dùng điểm validation này để chọn mô hình/ngưỡng hoặc thay đổi pipeline. Tiếp tục lựa chọn bằng nested OOF train-only; cần test/holdout độc lập mới để xác nhận generalization.


## Experiment — 2026-10-20 — nested-target-floor-r27-seed2028

### Câu hỏi / giả thuyết

Thay objective chọn inner candidate từ `minimum_f1_then_spread` sang ràng buộc trực tiếp F1 tối thiểu 0,75 và spread tối đa 0,10 có làm pooled outer OOF đạt mục tiêu cân bằng không? Kết quả gần đạt pooled target nhưng chưa ổn định ở mọi outer fold.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; nested 5 outer × 4 inner GroupKFold theo `stream_id` trên 13 train streams, seed 2028. Không dùng validation hoặc composition khi fit/selection; package evaluator `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

Giữ R26: CIR 50 Hz, sai phân nhân quả, pooling 5 frame/0,1 s, EMA tau 8 s; gamma 1,5/2,0, normalized rho temporal similarity và reservoir reset-channel được chọn trong inner OOF.

### Mô hình và cấu hình

HistGradientBoosting với stream-weight exponent 0/0,5/1,0; Ising/DP decoder. Candidate được xếp ưu tiên nếu cả ba F1@0,1/0,3/0,5 ≥0,75 và spread ≤0,10; nếu không, xếp theo F1 thấp nhất rồi spread. Không bật boundary readout hoặc contraction.

### Seed ngẫu nhiên

GroupKFold/readout seed 2028; inner seed đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`, nested outer OOF train-only; metric chưa được xác nhận là ranking chính thức.

### Kết quả

Nested OOF pooled mean F1 **0,8169**; F1@0,1/0,3/0,5 = **0,8563/0,8394/0,7549**, spread **0,1014**. Pooled scores vượt sàn 0,75 nhưng spread còn vượt trần 0,10 một ít. Outer-fold mean F1 theo ngưỡng là **0,8569/0,8304/0,7641**; outer-fold minimum F1@0,5 là **0,7000**, nên chưa đạt ổn định qua stream groups.

### Phân tích lỗi

Nested leakage/scope audit hoàn tất và các check partition, fit scope, OOF coverage, validation isolation và composition isolation đều true. Error audit tại theta 0,3: TP/FP/FN **149/20/37**, median absolute start/end error **0,18/0,20 s**; 13 false alarms trùng vùng fall nhưng dưới IoU, 6 trên ADL và 1 other. Signed boundary error median cho thấy start dự đoán sớm khoảng 0,10 s và end muộn khoảng 0,04 s; đây là giả thuyết để thử offset boundary quanh các profile hiện tại trong R28, lựa chọn vẫn nằm trong inner OOF.

### Artifact và lệnh tái lập

- Cấu hình: [quantum_histgb_nested_target_floor_r27_seed2028.yaml](../../configs/quantum_histgb_nested_target_floor_r27_seed2028.yaml)
- Report, OOF logits/predictions và train-only audit: `data/processed/quantum_diff_static_hybrid_nested_target_floor_r27_seed2028_20261020/`
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_target_floor_r27_seed2028.yaml`
- Audit: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_target_floor_r27_seed2028_20261020`

### Bước tiếp theo

R28 mở rộng lựa chọn offset biên trong inner OOF để kiểm tra giả thuyết start sớm/end muộn; không dùng validation đã tiêu thụ.


## Experiment — 2026-10-20 — nested-boundary-shift-r28-seed2028

### Câu hỏi / giả thuyết

R27 error audit cho thấy start hơi sớm/end hơi muộn. Mở rộng duration profiles với start shift +0,1 s và/hoặc end shift −0,1 s có tăng F1@0,5 hoặc giảm spread không? Không tăng metric pooled.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; nested 5 outer × 4 inner GroupKFold theo `stream_id` trên 13 train streams, seed 2028. Validation và composition không tham gia fit/selection; package evaluator `--ignore-flagged`.

### Tiền xử lý và cửa sổ hóa

Giữ nguyên R27: CIR 50 Hz, sai phân nhân quả, block 5 frame/0,1 s, EMA tau 8 s; feature gamma, normalized rho similarity và reservoir được chọn trong inner OOF.

### Mô hình và cấu hình

HistGradientBoosting với stream-weight exponent 0/0,5/1,0; target-floor objective; Ising/DP. Decoder search thêm global/adaptive profiles có start muộn hơn 0,1 s và adaptive profile có end sớm hơn 0,1 s. Cấu hình: [quantum_histgb_nested_boundary_shift_r28_seed2028.yaml](../../configs/quantum_histgb_nested_boundary_shift_r28_seed2028.yaml).

### Seed ngẫu nhiên

GroupKFold/readout seed 2028; inner seed đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`, nested outer OOF train-only; metric chưa được xác nhận là ranking chính thức.

### Kết quả

Pooled nested OOF trùng R27: mean F1 **0,8169**; F1@0,1/0,3/0,5 = **0,8563/0,8394/0,7549**, spread **0,1014**. Outer-fold minimum F1@0,5 vẫn **0,7000**. Một outer fold chọn profile offset mới, nhưng pooled event F1 không đổi.

### Phân tích lỗi

Train-only audit: TP/FP/FN tại theta 0,3 **149/20/37**, median absolute start/end **0,18/0,20 s**. Signed error median cải thiện từ khoảng −0,10/+0,04 s ở R27 thành −0,08/+0,02 s, nhưng event matching không đổi. Per-stream audit cho thấy nhiều miss ở tIoU 0,5 không có dự đoán chồng lấp; R29 chuyển sang tăng độ nhạy readout.

### Artifact và lệnh tái lập

- Report, OOF logits/predictions và audit: `data/processed/quantum_diff_static_hybrid_nested_boundary_shift_r28_seed2028_20261020/`
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_boundary_shift_r28_seed2028.yaml`
- Audit: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_boundary_shift_r28_seed2028_20261020`
- Audit xác nhận `validation_used=false`, `composition_file_used=false` và đúng scope train.

### Bước tiếp theo

R29 so sánh positive weight 5/8 và các ngưỡng logit thấp hơn trong nested OOF train-only.


## Experiment — 2026-10-20 — r28-consensus-validation-diagnostic

### Câu hỏi / giả thuyết

Fit candidate consensus đã chọn hoàn toàn từ R28 train-only nested OOF trên toàn bộ train rồi áp vào validation để xem kết quả. Đây là diagnostic được yêu cầu; không dùng nhãn validation để chọn hoặc chỉnh mô hình.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; fit trên 13 train streams, dự đoán 5 validation streams. Validation đã được dùng ở R2/R17/R23/R26 nên metric này không độc lập. Candidate được tổng hợp từ stream-level OOF; source-overlap audit sau đó cho thấy khoảng 75% held-out ADL source-time đã có trong fit streams, nên lựa chọn candidate có thể lạc quan.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, block 5 frame/0,1 s, EMA tau 8 s, static motion gamma 1,5; normalized rho similarity và quantum reservoir không bật. Cache train dùng các feature train-only đã lưu; validation được trích xuất riêng.

### Mô hình và cấu hình

HistGradientBoosting (positive weight 5, stream-weight exponent 0,5) với Ising/DP. Feature/readout/profile được chọn bằng mode, tham số số bằng median từ năm outer-fold inner selections của R28. Cấu hình: [quantum_histgb_final_candidate_r28_consensus.yaml](../../configs/quantum_histgb_final_candidate_r28_consensus.yaml).

### Seed ngẫu nhiên

2028.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`; metric chưa được xác nhận là ranking challenge chính thức.

### Kết quả

Mean event F1 **0,7218**; F1@0,1/0,3/0,5 = **0,8271/0,7820/0,5564**, spread **0,2707**. Tại theta 0,3: TP/FP/FN **52/14/15**, precision/recall **0,7879/0,7761**, false alarms/hour **6,83**, median absolute start/end error **0,31/0,38 s**. Mean vượt 0,70; tiêu chí nội bộ ≥0,75 ở cả ba ngưỡng và spread ≤0,10 chưa đạt.

### Phân tích lỗi

F1 giảm rõ ở tIoU 0,5; có 30 FN và 29 FP tại ngưỡng này. Prediction interval chưa đủ chặt hoặc nhiều miss ở tIoU 0,5; không điều chỉnh mô hình từ các nhãn validation đã xem.

### Artifact và lệnh tái lập

- Prediction CSV, model fit trên train, metadata và metric: `data/processed/quantum_diff_static_hybrid_validation_diagnostic_r28_consensus_20261020/`
- Fit trên train rồi dự đoán validation: `./.venv/bin/python -u scripts/predict_holdout.py --config configs/quantum_histgb_final_candidate_r28_consensus.yaml --streams-dir data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --split-name val --validation-diagnostic`
- Chấm sau khi CSV đã khóa: `./.venv/bin/python data/uwb_falltrack/abc2027_uwb_train_v1/src/abc2027/evaluate.py --gt data/uwb_falltrack/abc2027_uwb_train_v1/streams_events.csv --pred data/processed/quantum_diff_static_hybrid_validation_diagnostic_r28_consensus_20261020/val_predictions.csv --streams data/uwb_falltrack/abc2027_uwb_train_v1/streams/val --ignore-flagged --out data/processed/quantum_diff_static_hybrid_validation_diagnostic_r28_consensus_20261020/val_metrics_package_local.json`
- Metadata xác nhận `prediction_labels_read=false`, `composition_file_used_for_prediction=false`; evaluator chỉ đọc nhãn sau khi dự đoán đã khóa.

### Bước tiếp theo

Không tái sử dụng validation cho lựa chọn. Tiếp tục mục tiêu bằng train-only nested OOF; cần holdout/test mới để có đánh giá độc lập.


## Experiment — 2026-10-20 — nested-readout-sensitivity-r29-seed2028

### Câu hỏi / giả thuyết

Tăng positive weight từ 5 lên 8 và mở logit quantile thấp hơn có cải thiện F1 ở tIoU 0,5 và giảm độ vênh giữa ba ngưỡng? Toàn bộ lựa chọn chỉ trong inner OOF train.

### Phiên bản dữ liệu và split

Package cục bộ `abc2027_uwb_train_v1`; 5 outer × 4 inner GroupKFold theo `stream_id`, 13 train streams, seed 2028. Validation không được dùng. `streams_composition.csv` không được dùng trong fit/selection; composition train được đọc sau đó trong posthoc source-overlap audit.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz, block 5 frame/0,1 s, EMA tau 8 s; 8 feature variants gồm gamma 1,5/2,0, normalized rho similarity và quantum reservoir.

### Mô hình và cấu hình

HistGradientBoosting với positive weight 5/8 và stream-weight exponent 0/0,5/1,0; logit quantiles 0,90/0,95/0,975/0,99; Ising/DP. Selection objective `target_floor_then_spread`. Cấu hình: [quantum_histgb_nested_readout_sensitivity_r29_seed2028.yaml](../../configs/quantum_histgb_nested_readout_sensitivity_r29_seed2028.yaml).

### Seed ngẫu nhiên

2028; inner seed thay đổi theo outer fold.

### Metric theo protocol

Package evaluator cục bộ với `--ignore-flagged`, nested outer OOF train-only; metric chưa được xác nhận là ranking challenge chính thức.

### Kết quả

Pooled stream-level nested OOF mean event F1 **0,8208**; F1@0,1/0,3/0,5 = **0,8635/0,8468/0,7521**, spread **0,1114**. Outer-fold mean lần lượt **0,8631/0,8368/0,7611**, outer minimum F1@0,5 **0,70**. So với R27, F1@0,1/0,3 tăng **0,0072/0,0074**, F1@0,5 giảm **0,0028**, spread tăng **0,0100**. Positive weight 8 được chọn ở 1/5 fold. Source-overlap audit sau đó cho thấy stream-level folds không độc lập ở mức source data; không diễn giải các điểm này là estimate leak-free.

### Phân tích lỗi

Train-only audit tại theta 0,3: TP/FP/FN **152/21/34**, median absolute start/end error **0,19/0,18 s**, false alarms/hour **3,40**. Tại theta 0,5: TP/FP/FN **135/38/51**. F1@0,5 vừa qua sàn pooled nhưng outer minimum và độ vênh chưa đạt mục tiêu.

### Artifact và lệnh tái lập

- Report, OOF logits/predictions và audit: `data/processed/quantum_diff_static_hybrid_nested_readout_sensitivity_r29_seed2028_20261020/`
- Nested OOF: `./.venv/bin/python -u scripts/run_nested_oof.py --config configs/quantum_histgb_nested_readout_sensitivity_r29_seed2028.yaml`
- Leakage/error audit: `./.venv/bin/python scripts/audit_nested_oof.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_readout_sensitivity_r29_seed2028_20261020`
- Source-overlap audit (train composition only): `./.venv/bin/python scripts/audit_stream_source_overlap.py --artifact-dir data/processed/quantum_diff_static_hybrid_nested_readout_sensitivity_r29_seed2028_20261020`
- Partition/coverage audit xác nhận stream-ID scope và `validation_used=false`; source audit cho thấy **74,9–78,4%** held-out ADL source seconds overlap fit streams, và 13 train streams nằm trong một connected component.

### Bước tiếp theo

Không promote R29. Xem toàn bộ stream-level OOF R16–R29 là source-leakage-contaminated cho tới khi có protocol source-disjoint/purged hợp lệ. Không dùng validation đã tiêu thụ để chỉnh.


## Experiment — 2026-10-20 — source-group-vqc-baseline-r30

### Câu hỏi / giả thuyết

Một variational quantum classifier (VQC) nhỏ có tạo được baseline frame-level cho phát hiện ngã liên tục khi đánh giá theo nguồn radar độc lập không?

### Phiên bản dữ liệu và split

Gói tái đóng train-only `quantum_source_group_cv_package_20261020`: 467 pseudo-stream, mỗi pseudo-stream ứng với đúng một SHA-256 hash của file nguồn train; các clip cùng nguồn được ghép lại và crossfade 0,2 s được áp lại. Chạy 5-fold group OOF, seed 2028. Kiểm tra manifest và `streams_composition.csv` xác nhận mỗi pseudo-stream có một hash nguồn riêng và không có hash giao nhau giữa fit/held-out. Validation/test không dùng; nhãn fold held-out chỉ được evaluator đọc sau khi dự đoán đã khóa. Đây là source-file-disjoint, không khẳng định subject-disjoint nếu nhiều file thuộc cùng một người; metric trên pseudo-stream là phép đo nội bộ, không đồng nhất hoàn toàn với stream liên tục chính thức.

### Tiền xử lý và cửa sổ hóa

CIR 50 Hz; sai phân nhân quả, 5 frame/0,1 s, 48 motion-bin đầu vào (16 range bin × 3 radar). Trong mỗi fold, StandardScaler và PCA 4 thành phần được fit riêng trên các ví dụ train đã lấy mẫu, rồi PC chuẩn hóa và ánh xạ vào góc RY. Lấy tối đa 6 frame cho mỗi lớp trên mỗi nhóm nguồn; loss cân bằng hai lớp.

### Mô hình và cấu hình

VQC 4 qubit, 2 lớp RY/RZ trainable và vòng CNOT có hướng; đầu ra là trung bình kỳ vọng Pauli-Z, ánh xạ sang xác suất fall. Statevector được mô phỏng chính xác bằng NumPy trên CPU, optimizer SPSA chạy 100 bước; không dùng QPU. Decoder baseline cố định: xác suất ≥0,5, nối gap ≤0,2 s, bỏ đoạn <0,6 s. Không dò ngưỡng hoặc decoder bằng nhãn OOF.

### Seed ngẫu nhiên

Seed fold 2028–2032 cho lấy mẫu/SPSA; thứ tự fold dùng seed 2028.

### Metric theo protocol

Evaluator cục bộ của package với `ignore_flagged=true`; metric mean event F1 tại tIoU 0,1/0,3/0,5 chưa được xác nhận là ranking chính thức. Không dùng validation hoặc test.

### Kết quả

Mean event F1 **0,1668**; F1@0,1/0,3/0,5 = **0,1926/0,1729/0,1349**, spread **0,0576**. Tại tIoU 0,3: TP/FP/FN **123/1114/63**, false alarms/hour **185,22**. Mean F1 theo fold dao động **0,1347–0,2487**. Sàn 0,75 chưa đạt; spread thấp không bù được F1 thấp.

### Phân tích lỗi

Ngưỡng cố định 0,5 và bộ nối đoạn đơn giản tạo nhiều false alarm. Đây là baseline chưa được hiệu chỉnh; không kết luận VQC nói chung không phù hợp và không so trực tiếp với điểm trên validation/OOF stream-level đã có source overlap. Loss cân bằng train giảm nhẹ trong cả năm fold.

### Artifact và lệnh tái lập

- Cấu hình: [vqc_source_group_baseline_r30.yaml](../../configs/vqc_source_group_baseline_r30.yaml)
- Runner: [run_vqc_source_group_baseline.py](../../scripts/run_vqc_source_group_baseline.py)
- Report, OOF predictions/probabilities, fold models và feature cache: `data/processed/source_group_vqc_baseline_r30_20261020/`
- Chạy: `./.venv/bin/python -u scripts/run_vqc_source_group_baseline.py --config configs/vqc_source_group_baseline_r30.yaml`

### Bước tiếp theo

Trước khi phát triển thêm VQC, so sánh với readout cổ điển trên đúng các source-group folds và cùng cách tạo đoạn; nếu tiếp tục hiệu chỉnh, chọn ngưỡng/decoder chỉ trong fold train, không dựa vào nhãn held-out hoặc validation.

## Experiment — 2026-10-20 — manybody-source-nested-r31

### Câu hỏi / giả thuyết

Thay biểu diễn phụ thuộc range/radar cụ thể bằng các invariant của complex analytic encoding và many-body quantum reservoir, kết hợp soft core/onset/offset heads và QFI-anchored exact semi-Markov decoder, có cải thiện phát hiện/định vị ngã? Mục tiêu mới mean validation ≥0,82 là giả thuyết cần đo, không phải kết quả được bảo đảm bởi toán học lượng tử.

### Phiên bản dữ liệu và split

Train-only source-group repack: 467 hash đường dẫn source, 359 sự kiện (kể cả flagged), 6,0144 giờ, lấy từ 13 train streams. 5 outer × 3 inner folds, không trùng source-path group. Chưa chứng minh subject/session disjoint hoặc content-disjoint giữa hai đường dẫn khác nhau. Repack nối clip và fade lại nên đây là metric pseudo-stream nội bộ. Validation đã được xem trước đây; chỉ chấm diagnostic một lần sau khóa pipeline và không dùng để chọn cấu hình. Xem [đặc tả challenge](../challenge_spec.md).

### Tiền xử lý và cửa sổ hóa

Causal EMA nền 2 s, FIR endpoint của Hilbert cửa sổ quá khứ 33 frame, CIR 50 Hz. Central range moments/dispersion/center velocity và centered complex Fourier moments, aggregate mean/std/min/max qua radar. Không đưa absolute CoM hoặc fixed range-bin identities vào readout. Translation invariance giả định tín hiệu dịch chuyển không bị clipping tại mép range. Mỗi feature row pool 5 frame, timestamp là mẫu cuối block; không giấu look-ahead ở midpoint. Raw pure-state geometry tại lag 0,06/0,16/0,40 s; mixed-density lag 0,1/0,2/0,4 s.

### Mô hình và cấu hình

Ba spin 1/2, tensor Hilbert dimension 8; XXZ all-to-all + identical fields, `U=expm(-iHdt)`, complex128. Reservoir dùng CPTP convex injection/unitary memory, density smoothing 0,1/0,25/0,5 s. Partial trace singles/pairs; pure Schmidt/Meyer–Wallach, mixed pair negativity, purity/mixedness, basis L1 coherence, mixed-state Bures và temporal QFI/QGT finite difference. Tổng 220 feature. Coherence phụ thuộc basis; không khẳng định arbitrary-unitary invariance của toàn vector.

Readout HistGradientBoostingRegressor ba đầu, soft labels và mask flagged ±2 s; chỉ valid positives ghi đè ignore zone lân cận theo evaluator. Hai cấu hình 7/15 leaves với L2 40/20, 80 iterations/64 bins, tối đa 24k sampled rows, group balancing và train-only transforms. Thêm causal temporal mean/delta 0,4 s cho 660 cột đầu vào; các density scales vẫn giữ đủ. 14 cấu hình decoder (theta 0,25–0,85), exact DP trên mọi đoạn được phép, duration prior học từ fit events của từng inner fold. Confidence chỉ dùng heads/QFI/duration, không biết tIoU thật. Decoder backtrack offline, chưa xác nhận latency online. Final config là mode của các inner selections.

Compute pilot 60k rows/1.540 cột bị dừng trước khi có inner/outer score vì fit quá chậm; log giữ riêng. Thu gọn row budget/temporal summaries để giảm chi phí, chưa có validation R31 hoặc outer score tham gia quyết định này.

Decoder-grid pilot dải theta 0,25–0,45 có inner optimum ở mép trên, P/R@0,3 khoảng 0,322/0,879 và 276 FP/131 TP trong outer-fit inner OOF. Vì grid bị cắt ngắn, mở rộng đến 0,85 rồi chạy lại nested CV; không dùng validation. Một outer diagnostic của pilot đã xuất trước khi dừng, nên ghi nhận train development reuse/selection optimism; không gọi các lần chạy lại là test độc lập. Inner-head cache mới giữ config/code/source-manifest hashes và fit/heldout IDs để lặp decoder không phải fit lại mô hình.

### Seed ngẫu nhiên

2028; offset riêng cho inner/outer/final fit được runner lưu trong metadata.

### Metric theo protocol

Package evaluator `--ignore-flagged`, mean event F1 ở tIoU 0,1/0,3/0,5, median start error và FA/h ở 0,3. Đây là metric package-local, trạng thái chính thức xem [đặc tả challenge](../challenge_spec.md).

### Kết quả

Nested source-group OOF hoàn tất: mean F1 **0,600522**; F1@0,1/0,3/0,5 = **0,631854/0,616188/0,553525**. Tại 0,3: TP/FP/FN **118/79/68**, median start **0,14 s**, FA/h **13,1352**. Inner mode chọn readout 15 lá/L2 20 và decoder theta 0,75, boundary weight 0,15, QFI weight 0,04; final duration median fit từ train là 1,82 s.

Ablation trên cùng5fold: signal52+QFI decoder mean **0,601667** (0,640/0,625/0,540), quantum168+QFI decoder mean **0,003546** (0,010638/0/0). Cả hai giữ hyperparameters được chọn cho pipeline đầy đủ, không tune riêng. Chưa thấy lợi ích bổ sung của quantum features trong readout; quantum-only thất bại trong setting này, không phải chứng minh trần mô hình quantum.

Sau khi kiểm tra 30 inner fit scopes, source-path disjointness, coverage và SHA-256, pipeline/model đã khóa rồi prediction CSV khóa trước evaluator đọc nhãn. Validation diagnostic duy nhất R31: mean F1 **0,000000**; F1 ở cả ba ngưỡng đều 0; TP/FP/FN tại 0,3 **0/25/67**, 1 prediction bị ignore, FA/h **12,2021**, không có median start/end vì không có TP. **Không đạt 0,82, không cải thiện baseline R28 validation 0,7218, không promote.** Không điều chỉnh ngưỡng/model từ kết quả validation này.

Cùng model/ngưỡng R31 áp lại 13 train streams gốc cho mean F1 **0,039052**, TP/FP/FN **5/48/181** tại0,3. Đây là in-sample diagnostic, không OOF/test; cho thấy thất bại không chỉ nằm ở validation.

Train-only context audit: **186/186** valid falls của repack start trước30s sau reset, so với **5/186** trên train gốc. Cùng186event/3.591pairedfeature rows có median |SMD| **0,874**,149/219feature không hằng>|0,5|. Causal energy contrast mean thay đổi từ+0,267 sang−0,962 (SMD−2,626). Mapping359event sai số tối đa1,8e-15s;26event bao phủ13train có raw CIR giống hệt. Bằng chứng hỗ trợ reset/context confounding, không có lỗi lệch nhãn rõ ràng; chưa tách tác động nhân quả riêng của từng cơ chế. Không dùng validation cho diagnostic này.

Kiểm tra toán synthetic đã đạt: sáu hoán vị radar sai khác tối đa 2,67e-13; dịch range 20 bins không clip 1,78e-13; thay future prefix và thay I/O chunk sai khác 0; Bell negativity 0,5; GHZ Meyer–Wallach 1 và pair negativity 0. DP so với scalar reference độc lập đạt 120 trường hợp irregular grid/gap/block. Kiểm tra mask, future independence, fit scope, fit/predict/pickle và row cap đều đạt. Các check này không chứng minh generalization.

### Phân tích lỗi

Audit lịch sử: 0,8520 thuộc R17 OOF, còn 0,7218 thuộc R28 validation. Cặp R17 là 0,8520→0,6957; R28 nested OOF→consensus validation là 0,8169→0,7218. Với R28, recall@0,1 0,8172→0,8209 gần như giữ nguyên, precision giảm; F1@0,5 giảm 0,7549→0,5564. Start/end median @0,3 tăng 0,18/0,20→0,31/0,38 s và FA/h 3,24→6,83. Đây là dấu hiệu localization yếu hơn và nhiều FA hơn, chưa phân chia được nguyên nhân bằng causal ablation. OOF cũ bị ADL source overlap; không dùng làm bằng chứng độc lập hoặc làm mốc so trực tiếp với source-group OOF mới.

### Artifact và lệnh tái lập

- Mã: `src/abc_uwb_falltrack/manybody_quantum.py`, `manybody_readout.py`, `manybody_decoder.py`.
- Cấu hình: [manybody_source_nested_r31.yaml](../../configs/manybody_source_nested_r31.yaml).
- Runner: [run_manybody_nested.py](../../scripts/run_manybody_nested.py).
- Tiến trình, historical audit, math checks, decoder checks, cache/model/predictions: `ABC testing/r31/` (local, ignored).
- Train: `./.venv/bin/python -u scripts/run_manybody_nested.py --config configs/manybody_source_nested_r31.yaml --stage train`.
- Chỉ sau pipeline lock: stage `predict-validation`, sau prediction lock: stage `evaluate-validation`; CSV tại `submissions/`.
- Không dùng QPU, không upload dữ liệu, không commit/push/GitHub. Đã chạy kiểm tra số học theo yêu cầu kiểm toán/kiểm chứng, không thêm formal test suite.

### Bước tiếp theo

R31 đã hoàn tất nested CV, khóa pipeline/model/config, chấm một lượt validation diagnostic và chạy hai ablation readout trên cùng folds. Không chạy lại validation hoặc chọn lại cấu hình theo điểm đã tiêu thụ. R32/R33 tiếp tục train-only trên source-disjoint context package; xem các mục sau.


## Experiment — 2026-10-21 — manybody-source-context-nested-r32

### Câu hỏi / giả thuyết

Khắc phục reset/context confounding quan sát trên train bằng cách thay 467 source stream cô lập bằng 13 bundle liên tục có cả ADL, fall và hoạt động khác. Giữ nguyên lõi many-body và chọn readout/decoder hoàn toàn trong train; không chấm lại validation đã tiêu thụ.

### Phiên bản dữ liệu và split

Tái đóng từ train-only source package; không đọc validation hoặc original segment arrays. Mỗi source path thuộc đúng một trong 13 bundle. Nested CV gồm 5 outer × 3 inner fold theo bundle. Wrapper xác nhận union các source hash thực tế không giao nhau ở cả hai cấp. Không chứng minh subject/session disjoint vì chưa có mapping được xác nhận. Đây là protocol nội bộ trên synthetic continuous bundles, không thay thế [đặc tả challenge](../challenge_spec.md).

### Tiền xử lý và cửa sổ hóa

Causal encoder/QRC giữ như R31. Ghép clip liên tục trong từng bundle, giữ nguyên ánh xạ timestamp/nhãn và bảo vệ biên khi crossfade 0,2 s. Không dùng elapsed time hoặc composition khi inference.

### Mô hình và cấu hình

Giữ 220 feature quantum/signal, 660 cột sau temporal transform, HistGradientBoosting với ba đầu ra mềm và exact semi-Markov DP. Giới hạn 24k hàng train; hai readout có 7/15 lá, L2 40/20. Mở rộng tìm kiếm theta 0,15–0,95 bên trong inner CV. Cấu hình: [manybody_source_context_nested_r32.yaml](../../configs/manybody_source_context_nested_r32.yaml).

### Seed ngẫu nhiên

2028; các offset seed inner/outer/final được runner lưu trong metadata.

### Metric theo protocol

Package evaluator train-only với `ignore_flagged=true`; báo cáo mean event F1 ở tIoU 0,1/0,3/0,5, median start error và false alarms/hour tại theta 0,3. Metric package-local; protocol challenge đã xác nhận xem [đặc tả challenge](../challenge_spec.md).

### Kết quả

Gói dữ liệu có 13 bundle, 467 source-path hash, 5,848544 giờ và đủ 359 sự kiện (186 valid, 173 flagged). 359/359 event sample slices được xác nhận bit-exact sau crossfade; 12 clip non-fall không còn pure interior bị loại (3,52 s), không mất event hay source group. Valid fall trước 30 s giảm từ 186/186 trong R31 repack xuống 1/186. Kiểm tra provenance đạt exact coverage và không có source overlap ở 5 outer/15 inner split.

Nested OOF hoàn tất, mean event F1 **0,630491**; F1@0,1/0,3/0,5 = **0,677003/0,651163/0,563307**. Tại theta 0,3: TP/FP/FN **126/75/60**, precision/recall **0,6269/0,6774**, median start/end error **0,17/0,25 s**, false alarms/hour **12,8237**. Outer-fold mean lần lượt **0,7679/0,5417/0,6170/0,6667/0,5763**; kết quả vẫn dao động đáng kể.

Inner selection mode khóa readout 15 lá/L2 20 và decoder theta 0,65, boundary weight 0,15, QFI weight 0,04. Trên cùng các outer folds, ablation 52 signal features + QFI decoder đạt mean **0,649616** (F1@0,1/0,3/0,5 = 0,700767/0,670077/0,578005); ablation 168 quantum features + QFI decoder đạt mean **0,123810** (0,142857/0,123810/0,104762). Các ablation giữ cấu hình đã chọn cho hệ đầy đủ, không tune riêng; chúng không chứng minh quantum advantage hoặc mức trần của quantum model. Với lựa chọn hiện tại, quantum feature block không cải thiện score so với signal-only.

Model đã khóa được áp lại lên 13 train stream gốc: mean F1 **0,664898**. Đây là in-sample context diagnostic, không phải OOF hay test. Không dùng validation R32; mục tiêu validation ≥0,82 chưa đạt và chưa được kiểm chứng độc lập.

### Phân tích lỗi

Train-only reset/context audit là động cơ của R32, không dùng validation để chọn. Gói mới giảm confounding điểm reset; tuy vậy nested OOF vẫn thấp và false alarm rate cao. OOF class audit tại tIoU 0,3: recall `Chute01/02/03/04` lần lượt **0,810/0,731/0,784/0,341**. Lớp `Chute04` chiếm 27/60 FN; OOF core mean/max bên trong event **0,416/0,701**, thấp hơn các lớp kia (core mean 0,624–0,659; max 0,875–0,897). Trong 75 FP, 18 nằm chủ yếu ở `Faire_menage`, 14 ở `Marcher`; 9 FP khác giao valid-fall interval nhưng không đạt IoU 0,3. Composition chỉ được dùng sau khi OOF đã khóa để kiểm tra lỗi train, không dùng trong inference/selection. Signal-only nhỉnh hơn full readout, còn quantum-only yếu trong hyperparameters hiện tại.

### Artifact và lệnh tái lập

- Builder: `scripts/build_source_disjoint_context_package.py`.
- Package: `data/processed/manybody_context_cv_package_r32/`; dữ liệu nguồn giữ nguyên.
- Runner: `scripts/run_manybody_context_nested.py --config configs/manybody_source_context_nested_r32.yaml`.
- Ablation: `scripts/run_manybody_context_ablations.py --config configs/manybody_source_context_nested_r32.yaml`.
- Report, lock, model, provenance và cache cục bộ: `ABC testing/r32/` (Git-ignored).
- Không chạy validation R32; không upload, commit hoặc push.

### Bước tiếp theo

R33 multiscale đã hoàn tất train-only nested CV; kết quả và quyết định ghi ở mục kế tiếp. Không mở vòng chọn theo validation cũ. Cần holdout/test mới với participant mapping rõ để xác nhận mục tiêu 0,82.


## Experiment — 2026-10-21 — manybody-source-context-multiscale-r33

### Câu hỏi / giả thuyết

R32 chỉ dùng causal temporal summary 0,4 s dù fall hợp lệ có median duration khoảng 1,8 s. Thêm cửa sổ causal 1,2 s có cải thiện nested source-group OOF không? Cấu hình và quyết định được ghi trước khi xem pooled R32 result; chỉ train data được dùng.

### Phiên bản dữ liệu và split

Dùng lại nguyên gói R32 gồm 13 continuous context bundle, 467 source-path hash, 359 event và 5,848544 giờ. Giữ nguyên 5 outer × 3 inner split, seed 2028, source union disjoint. Không dùng validation/test/nhãn validation hoặc composition khi inference. Không chứng minh participant/session disjoint; metric trên synthetic continuous bundle vẫn là package-local. Xem [đặc tả challenge](../challenge_spec.md).

### Tiền xử lý và cửa sổ hóa

Giữ nguyên R32 causal complex encoder, QRC, tensor 3 radar và 220 feature đầu vào. Temporal summary dùng history causal 0,4 s và 1,2 s; không dùng frame tương lai. Gói feature cache train R32 được copy độc lập, không dùng hardlink/symlink.

### Mô hình và cấu hình

Giữ HistGB ba soft heads, duration prior theo fit scope và exact DP. So sánh nội bộ ba readout candidate: R32 compact7, R32 compact15 và compact15 với temporal windows `[0.4, 1.2]` s. Cùng 24k hàng train, 80 iteration, max 15 leaf, L2 20 cho candidate multiscale; chiều readout tăng từ 660 lên **1.100 cột**. Giữ 18 decoder candidate như R32. Cấu hình: [manybody_source_context_multiscale_r33.yaml](../../configs/manybody_source_context_multiscale_r33.yaml).

### Seed ngẫu nhiên

2028, cùng outer/inner seed R32 để so sánh trên cùng các nhóm.

### Metric theo protocol

Package evaluator train-only với `ignore_flagged=true`; mean event F1 tại tIoU 0,1/0,3/0,5. Không dùng nhãn validation; trạng thái protocol chính thức xem [đặc tả challenge](../challenge_spec.md).

### Kết quả

Nested OOF mean event F1 **0,615257**; F1@0,1/0,3/0,5 = **0,661692/0,621891/0,562189**, thấp hơn R32 lần lượt **0,0153/0,0293/0,0011**. Tại theta 0,3: TP/FP/FN **125/91/61**, precision/recall **0,5787/0,6720**, median start/end error **0,16/0,22 s**, mean tIoU **0,7226**, false alarms/hour **15,5594**. Outer-fold mean: **0,6667/0,5409/0,6055/0,6563/0,6667**. Inner CV chọn multiscale ở cả 5 outer folds, nhưng decoder index thay đổi **11/2/10/10/6**. Final lock chọn compact15 multiscale, theta **0,55**, boundary weight **0,15**, QFI weight **0,04**.

Model đã khóa được áp lên 13 train stream gốc: mean in-sample **0,679901**, so với **0,664898** của R32. Cùng lúc, source-group OOF giảm từ 0,630491 xuống 0,615257 và FP tăng 75→91 tại theta 0,3. Chẩn đoán original-train dùng các nguồn mẫu đã có trong fit, nên không phải generalization evidence. Không chạy validation R33; mục tiêu ≥0,82 chưa đạt/chưa được xác nhận.

### Phân tích lỗi

Mở rộng history làm tăng 440 cột nhưng giữ nguyên 24k hàng fit và chỉ 13 bundle độc lập. In-sample score tăng còn outer OOF giảm; TP gần như không đổi, FP tăng 16. OOF class recall `Chute01/02/03/04` tại tIoU 0,3 lần lượt **0,857/0,712/0,804/0,268**; `Chute04` chiếm 30/61 FN. Trong 91 FP, 23 nằm chủ yếu ở `Faire_menage`, 16 ở `Marcher`; 14 FP giao valid-fall interval nhưng không đạt IoU 0,3. OOF core mean/max bên trong `Chute04` là **0,423/0,719**. Mẫu này phù hợp với việc thêm scale làm tăng khớp in-sample nhưng không cải thiện phân biệt theo group; không xác định được nguyên nhân vật lý của `Chute04` chỉ từ metric này.

### Artifact và lệnh tái lập

- Runner: `scripts/run_manybody_context_multiscale.py --source-config configs/manybody_source_context_nested_r32.yaml --config configs/manybody_source_context_multiscale_r33.yaml`.
- Report, model lock, OOF predictions, cache và log: `ABC testing/r33/` (local, Git-ignored).
- Không chạy ablation riêng cho R33; không chấm lại validation; không upload, commit hoặc push.

### Bước tiếp theo

Không promote R33. Dừng thử thêm quantum feature/decoder trên cùng 13 bundle và không dùng validation đã tiêu thụ để săn điểm. Cần holdout mới cùng metadata participant/session và protocol evaluator được xác nhận trước khi tuyên bố generalization hoặc đạt mục tiêu 0,82.
