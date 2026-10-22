# Scripts

Đặt tại đây các lệnh có thể chạy từ terminal, chẳng hạn kiểm tra dữ liệu, huấn luyện baseline, đánh giá hoặc sinh dự đoán. Mỗi script nên có trợ giúp tham số, dùng đường dẫn tương đối/cấu hình và không phụ thuộc vào thư mục riêng trên máy một thành viên.

## Chọn cấu hình train-only và dự đoán holdout

Lựa chọn cấu hình ứng viên R17 chỉ đọc 13 stream train, dùng 5-fold GroupKFold và package evaluator. Lệnh lưu OOF artifacts trong `data/processed/` và sinh [cấu hình ứng viên](../configs/quantum_histgb_final_candidate_r17.yaml):

```bash
./.venv/bin/python -u scripts/select_final_candidate.py --config configs/quantum_histgb_final_selection_r17.yaml
```

Kết quả chọn cấu hình không phải điểm validation/test độc lập. Năm stream validation đã được dùng cho baseline R2; R17 chỉ có một lượt diagnostic theo yêu cầu, không được dùng để chỉnh mô hình. Mặc định script vẫn chặn validation. Khi có thư mục test/holdout mới được phép dùng, dự đoán bằng:

```bash
./.venv/bin/python -u scripts/predict_holdout.py --config configs/quantum_histgb_final_candidate_r17.yaml --streams-dir /path/to/test/streams --split-name test
```

Script fit readout từ train, không đọc nhãn holdout hoặc `streams_composition.csv`, từ chối split validation cũ, rồi dùng evaluator package để kiểm tra schema CSV và giới hạn thời gian dựa trên độ dài các stream đầu vào. Việc kiểm tra này không đọc nhãn; dự đoán cùng metadata được ghi vào `outputs/submissions/` theo cấu hình.

R23 dùng readout stream-balanced (exponent 1,0), chọn feature/decoder bằng OOF train-only và tạo [cấu hình riêng](../configs/quantum_histgb_final_candidate_stream_balance_r23.yaml). R26 và R28 cũng có diagnostic validation theo yêu cầu. Mỗi lượt fit từ train; nhãn validation chỉ được evaluator dùng sau khi CSV đã khóa. Vì validation đã được xem qua nhiều lượt, các điểm này không độc lập và không dùng để tinh chỉnh/chọn mô hình. Không chạy thêm diagnostic validation.

Lưu ý: OOF trước đây GroupKFold theo `stream_id` đã qua partition audit nhưng train-only composition audit phát hiện source-time ADL overlap giữa các stream. Chạy [audit_stream_source_overlap.py](audit_stream_source_overlap.py) cùng [audit_nested_oof.py](audit_nested_oof.py); chưa coi R16–R29 là estimate OOF độc lập cho tới khi có protocol source-disjoint/purged hợp lệ.

## Many-body quantum R31

Theo yêu cầu nâng cấp, R31 dùng [cấu hình riêng](../configs/manybody_source_nested_r31.yaml), chọn bằng nested source-group CV trên train-only repack. Mỗi group là một hash **đường dẫn** source; chưa chứng minh khác participant hoặc không có duplicate giữa hai đường dẫn. Artifact được giữ cục bộ trong `ABC testing/r31/`.

```bash
./.venv/bin/python -u scripts/run_manybody_nested.py --config configs/manybody_source_nested_r31.yaml --stage train
```

Lệnh fit 5 outer × 3 inner folds; chọn readout/decoder trong inner scope, fit duration prior từ fit labels, rồi fit final readout trên toàn bộ source-group train. `pipeline_lock.json` lưu code/config/model SHA-256 trước khi mở validation. Lựa chọn cuối là mode của các lựa chọn inner, không chọn bằng outer score. Trạng thái đang chạy nằm trong `progress.json`.

Yêu cầu R31 cho phép một lượt validation diagnostic mới sau khi khóa pipeline:

```bash
./.venv/bin/python -u scripts/run_manybody_nested.py --config configs/manybody_source_nested_r31.yaml --stage predict-validation
./.venv/bin/python -u scripts/run_manybody_nested.py --config configs/manybody_source_nested_r31.yaml --stage evaluate-validation
```

Inference chỉ nhận CIR/timestamp, không đọc nhãn hoặc composition. CSV và model phải khớp SHA-256 khi chấm; script chặn chấm lại hoặc fit lại cùng experiment sau khi prediction validation đã khóa. CSV xuất ở `submissions/`; đây chưa phải submission test chính thức. Validation đã được xem trước đây nên kết quả vẫn chỉ là diagnostic và không dùng để mở vòng chọn cấu hình mới.

Lõi là mô phỏng ba spin 1/2 bằng complex128: causal analytic encoding, tensor product, XXZ unitary, CPTP memory, partial trace/negativity và hình học Bures/QFI finite difference. Pha analytic không được khẳng định là measured RF Doppler phase; entanglement thuộc encoding được mô phỏng. Readout đối xứng radar; coherence vẫn phụ thuộc computational basis. Decoder là classical exact semi-Markov DP, dùng toàn stream để backtrack, nên chưa phải bộ phát cảnh báo online với latency được đảm bảo.

## R32 — Sửa context của source-group CV

R31 source repack reset sau mỗi file ngã làm tất cả positive events ở đầu stream. Cùng model R31 hoạt động rất kém ngay trên continuous train gốc; đây là context shift cần sửa trước khi bàn generalization. Builder R32 tạo 13 bundle liên tục trộn ADL/ngã, mỗi nguồn thuộc đúng một bundle:

```bash
./.venv/bin/python scripts/build_source_disjoint_context_package.py --source-root data/processed/quantum_source_group_cv_package_20261020 --output-dir data/processed/manybody_context_cv_package_r32 --artifact-dir 'ABC testing/r32/build' --seed 2028 --bundles 13
./.venv/bin/python -u scripts/run_manybody_context_nested.py --config configs/manybody_source_context_nested_r32.yaml
```

Wrapper kiểm tra toàn bộ source sets trong từng outer/inner split, giữ code/config/model lock, và chỉ cung cấp train-only run. Không chấm lại validation đã tiêu thụ. Sau train, kiểm tra trên 13 train streams gốc chỉ là in-sample context diagnostic, không phải estimate generalization. Toàn bộ tiến trình R32 được giữ cục bộ tại `ABC testing/r32/` và bị Git bỏ qua.
