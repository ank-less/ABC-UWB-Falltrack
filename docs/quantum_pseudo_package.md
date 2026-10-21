# Gói baseline pseudo-quantum

Đây là gói baseline R17 đã chọn từ các thí nghiệm train-only. Tên “pseudo-quantum” chỉ các đặc trưng trạng thái lượng tử được mô phỏng bằng NumPy từ CIR cổ điển; pipeline không chạy trên QPU và không khẳng định có lợi thế lượng tử.

## Thành phần

- `src/abc_uwb_falltrack/quantum_features.py`: lấy sai khác CIR, gom 368 range bins thành 16 bin năng lượng và amplitude-encode thành trạng thái 4-qubit thực riêng cho từng radar. R17 còn bật causal static contrast (`tau=8 s`, `gamma=2.0`), ghép chuyển động/nền thành trạng thái hybrid 5-qubit và thêm normalized rho temporal similarity. Readout nhận cả đặc trưng classical lẫn các đại lượng ma trận mật độ, entropy, fidelity và Pauli; reservoir tùy chọn không bật.
- `src/abc_uwb_falltrack/oof.py`: đọc nhãn train, mask sự kiện bị đánh cờ, fit readout và giải mã chuỗi.
- `scripts/predict_holdout.py`: fit lại readout trên train rồi dự đoán thư mục stream mới; chặn split validation đã dùng và không đọc nhãn holdout hay `streams_composition.csv`.
- `configs/quantum_histgb_final_candidate_r17.yaml`: cấu hình readout HistGradientBoosting cùng decoder Ising/DP đã khóa.

Pipeline giả định package dữ liệu cục bộ có train streams, `streams_events.csv` và evaluator ở đường dẫn `data_root` trong config. Dữ liệu bị loại khỏi Git và phải được cấp riêng theo giấy phép của challenge.

## Cài đặt và chạy

```bash
python -m venv .venv
./.venv/bin/python -m pip install -e '.[analysis]'
./.venv/bin/python -u scripts/predict_holdout.py \
  --config configs/quantum_histgb_final_candidate_r17.yaml \
  --streams-dir /path/to/test/streams \
  --split-name test
```

Lệnh fit readout lại trên train; không cần checkpoint đã huấn luyện. Dự đoán và metadata được ghi vào thư mục `output_dir` trong config. Không dùng lệnh này với validation đã xem để chọn mô hình.

## Kết quả và giới hạn

R17 ghi nhận OOF mean event F1 **0.8520**, với F1@tIoU 0.1/0.3/0.5 là **0.8930/0.8663/0.7968**. Lượt diagnostic trên validation ghi nhận mean **0.6957** và F1@0.1/0.3/0.5 **0.8116/0.7536/0.5217**. Validation đã được dùng trong các lượt trước nên không còn là holdout độc lập.

Audit sau đó phát hiện khoảng **74.9–78.3%** overlap theo source-time ADL giữa các fold của OOF GroupKFold theo `stream_id`. Vì thế OOF 0.8520 có thể lạc quan và không được xem là ước lượng generalization độc lập. Metric là evaluator của package dữ liệu cục bộ; protocol công khai hiện chưa xác nhận đây là công thức xếp hạng chính thức. Xem [đặc tả challenge](challenge_spec.md) để biết trạng thái protocol.
