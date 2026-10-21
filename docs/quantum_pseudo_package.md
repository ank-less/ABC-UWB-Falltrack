# Gói baseline pseudo-quantum

Đây là gói baseline R17 đã chọn từ các thí nghiệm train-only. Tên “pseudo-quantum” chỉ các đặc trưng trạng thái lượng tử được mô phỏng bằng NumPy từ CIR cổ điển; pipeline không chạy trên QPU và không khẳng định có lợi thế lượng tử.

## Thành phần

- `src/abc_uwb_falltrack/quantum_features.py`: lấy sai khác CIR theo thời gian, tạo trạng thái biên độ 4-qubit riêng cho mỗi radar và tính các đặc trưng mật độ/trạng thái. Cấu hình R17 bật causal static contrast (`tau=8 s`, `gamma=2.0`) và normalized rho similarity; không bật reservoir tùy chọn.
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

Audit sau đó phát hiện khoảng **74.9–78.3%** overlap theo source-time ADL giữa các fold của OOF GroupKFold theo `stream_id`. Vì thế OOF 0.8520 có thể lạc quan và không được xem là ước lượng generalization độc lập. Metric là evaluator của package dữ liệu cục bộ; protocol công khai hiện chưa xác nhận đây là công thức xếp hạng chính thức. Xem [đặc tả challenge](challenge_spec.md) và [experiment log của workspace](knowledge/Experiment%20Log.md) để biết nguồn và bối cảnh.
