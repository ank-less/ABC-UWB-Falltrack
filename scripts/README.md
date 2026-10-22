# Scripts

Các lệnh chạy từ thư mục gốc của repo. Mọi script đều có `--help`; tham số đọc từ `configs/baseline.yaml` và có thể sửa tạm bằng `--override section.key=value`.

| Script | Việc |
|---|---|
| `fetch_data.py` | Tải các file cần thiết từ zip dữ liệu chính thức (range request, kiểm tra CRC) |
| `inspect_data.py` | Báo cáo kiểm tra dữ liệu: schema HDF5, timestamp, NaN/Inf, nhãn, composition |
| `build_manifest.py` | Manifest theo luồng và bảng segment (không chứa CIR) |
| `run_baseline.py` | Pipeline baseline; `--stage inspect|manifest|train|evaluate|all`, `--smoke` |
| `evaluate_predictions.py` | Chấm một file submission bằng evaluator chính thức |
| `make_eda_report.py` | Hình và số liệu cho báo cáo EDA (NV2) |
| `make_results_report.py` | Hình và số liệu cho báo cáo kết quả (NV3), đọc từ thư mục run |

Tài liệu đầy đủ: [`docs/baseline_pipeline.md`](../docs/baseline_pipeline.md).
