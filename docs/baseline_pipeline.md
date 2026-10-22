# Baseline pipeline — phát hiện ngã liên tục trên luồng UWB

Tài liệu kỹ thuật của pipeline baseline: cách chạy, cách pipeline chống rò rỉ dữ liệu, và những gì còn là giả định.
Kết quả và phân tích nằm trong [`reports/baseline_results_report.md`](../reports/baseline_results_report.md). Phần EDA nằm trong [`reports/eda_preprocessing_report.md`](../reports/eda_preprocessing_report.md).

## 1. Cách chạy

Yêu cầu Python 3.10 trở lên. Từ thư mục gốc của repo:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[analysis,test]"
python scripts/fetch_data.py --subset all     # khoảng 4,6 GB; --subset smoke khoảng 1,1 GB
```

| Việc | Lệnh |
|---|---|
| Kiểm tra dữ liệu (đọc từng khúc, quét NaN/Inf) | `python scripts/inspect_data.py --config configs/baseline.yaml` |
| Manifest và bảng segment | `python scripts/build_manifest.py --config configs/baseline.yaml` |
| Smoke run (2 luồng mỗi split, mỗi luồng 300 s) | `python scripts/run_baseline.py --config configs/baseline.yaml --smoke` |
| Chạy đầy đủ (train rồi đánh giá val) | `python scripts/run_baseline.py --config configs/baseline.yaml` |
| Chỉ train | `python scripts/run_baseline.py --config configs/baseline.yaml --stage train` |
| Đánh giá một run đã train | `python scripts/run_baseline.py --stage evaluate --run-dir outputs/experiments/<run>` |
| Chấm một file submission | `python scripts/evaluate_predictions.py --config configs/baseline.yaml --predictions <csv> --split val` |
| Đổi một tham số mà không sửa file | `--override windowing.duration_seconds=2.0` (dùng được nhiều lần) |
| Test tự động | `python -m pytest -q` |
| Hình và số liệu cho báo cáo | `python scripts/make_eda_report.py`, `python scripts/make_results_report.py --run-dir <run>` |
| Chẩn đoán lệch ngưỡng giữa model fold và model cuối | `python scripts/diagnose_fold_ensemble.py --run-dir <run>` |

Mọi đường dẫn trong config là đường dẫn tương đối, tính từ thư mục chạy lệnh (gốc repo). Các lệnh chạy được trên Windows PowerShell, Linux và Colab. Notebook Colab: [`notebooks/02_baseline_colab.ipynb`](../notebooks/02_baseline_colab.ipynb).

Chạy đầy đủ mất khoảng 4 phút trên laptop: khoảng 1 phút đọc 6,7 GB CIR lần đầu, sau đó dùng cache trong `outputs/cache/`.

## 2. Schema đã quan sát

Đã kiểm tra trên toàn bộ 18 file ngày 2026-10-22. Kết quả chi tiết nằm trong `outputs/inspection/<run>/inspection.json`.

- `cir`: `(3, T, 368)`, float32, giá trị thực, chunk `(3, 5000, 368)`, không nén. Trục 0 là radar, trục 1 là thời gian, trục 2 là range bin.
- `t`: float64, tính bằng giây, `t[0] = 0`, bước 0,02 s (50 Hz), tăng đơn điệu.
- `radar_serials = [29171, 29444, 30722]`, giống nhau ở mọi file.
- 0 giá trị NaN/Inf. Train dài 6,18 h, val dài 2,05 h. Mỗi luồng khoảng 29 phút (riêng `train_12` dài 19 phút, `val_04` dài 5 phút).
- Khoảng 25–27% số frame giống hệt frame trước: radar gốc chạy khoảng 37–38 Hz, frame được nhân bản khi căn về lưới 50 Hz. Xem báo cáo EDA.

## 3. Kiến trúc

```text
src/abc_uwb_falltrack/
  config.py          đọc YAML, --override, báo lỗi khi thiếu giá trị bắt buộc
  data_io.py         metadata HDF5, đọc CIR theo khúc, đọc events/composition
  manifest.py        manifest theo luồng, bảng segment kèm subject_group
  validation.py      kiểm tra schema/timestamp/nhãn/split, LeakageError
  splitting.py       split chính thức, fold theo người, đánh dấu cửa sổ chạm fold nào
  preprocessing.py   trừ nền EMA nhân quả, chuỗi năng lượng theo frame, cache
  windowing.py       cửa sổ trong một luồng, gán nhãn theo chồng lấn thời gian
  features.py        thống kê theo cửa sổ của từng chuỗi năng lượng
  modeling.py        Dummy / Logistic / HistGradientBoosting trong sklearn Pipeline
  postprocessing.py  score cửa sổ -> sự kiện (làm mượt, ngưỡng, gộp, độ dài tối thiểu)
  metrics.py         metric cửa sổ; gọi evaluator chính thức cho metric sự kiện
  experiment.py      điều phối, thư mục run, chẩn đoán, artifact
  inspection.py      báo cáo kiểm tra dữ liệu
  eda.py             phân tích cho báo cáo NV2
  synthetic.py       dữ liệu tổng hợp cho test
scripts/             CLI (mỗi script đều có --help)
tests/               44 test (pytest), chủ yếu trên dữ liệu tổng hợp
```

## 4. Split và các hàng rào chống rò rỉ

**Split chính thức** (`split.strategy: official`): train dùng để học và chọn tham số. Val không được dùng ở bất kỳ bước fit hay bước chọn nào. Tuy vậy, mỗi run vẫn chấm val cho cả ba mô hình, và nhóm đã xem điểm val của nhiều run (baseline và các ablation). Vì vậy val là tập **development**, không còn là holdout sạch. Điểm cuối cùng chỉ đáng tin trên test ẩn.

**Cross-validation bên trong train theo người.** Các luồng được ghép từ những bản ghi nguồn dùng chung: một file ADL nguồn có mặt trong 2–13 luồng train, mỗi người ngã có mặt trong 4–12 luồng. Vì vậy chia fold theo `stream_id` sẽ rò rỉ. Nhóm của mỗi segment được suy ra từ đường dẫn nguồn trong `streams_composition.csv`:

- `falls/Dataset d/Test_Pd_k/...` → `fall_person_k`: gom cùng người ở cả ba vị trí thảm, đây là cách gom thô hơn và an toàn hơn;
- `activities/<Activity>_n.json` → `adl_subject_n`.

Giả thuyết "k và n là mã người" khớp với dữ liệu: các mã này không giao nhau giữa train và val chính thức (train có người ngã 1, 2, 3, 5, 6, 7 và ADL 4, 5, 6, 9, 10, 12; val có người ngã 8, 9 và ADL 0, 11). Có 12 nhóm train, chia vào 3 fold. Mỗi corpus được xáo trộn riêng với seed 2027, nên fold nào cũng có cả ngã lẫn ADL.

**Các kiểm tra tự động.** Pipeline dừng với `LeakageError` nếu gặp một trong các trường hợp sau:

1. một `stream_id` có mặt ở hai split chính thức;
2. một `subject_group` cấp dữ liệu cho luồng của hai split chính thức;
3. một nhóm nằm ở hai fold;
4. model của fold k được train bằng một cửa sổ có chạm tới segment của fold k. Đây là bước **purge** phía train: cửa sổ nằm vắt qua chỗ nối giữa hai người bị loại khỏi tập train của cả hai fold;
5. một cửa sổ held-out của fold k chạm tới segment của fold khác.
6. tập fit cuối cùng có cửa sổ không thuộc train;
7. đặc trưng của val không khớp schema đặc trưng đã train.

**Cửa sổ vắt qua hai fold không có score OOF.** Một cửa sổ chạm segment của hai fold j và k không có mô hình fold nào "sạch" để chấm nó: model của fold k đã học các cửa sổ chỉ nằm trong fold j, và một số cửa sổ đó dùng chung frame với nó. Vì vậy cửa sổ này để trống (NaN) trong score OOF, và hậu xử lý coi nó là cửa sổ bị thiếu. Trên train có 6 822 cửa sổ như vậy (6,1%). Chúng vẫn được dùng để train model của fold mà chúng không chạm tới. Phiên bản trước có lỗi rò rỉ ở đây: nó gán cửa sổ này cho fold chiếm đa số và chấm như cửa sổ held-out bình thường.

**Các bước có fit.** Imputer và scaler nằm trong sklearn `Pipeline`, nên chỉ fit trên các dòng của tập train tương ứng. Trừ nền EMA là bộ lọc nhân quả, không fit và không trộn các luồng. Threshold và hậu xử lý được chọn trên score **out-of-fold của train**.

**Thứ tự các bước:** split → fold → (cho từng luồng) tiền xử lý, cửa sổ, đặc trưng → CV → chọn ngưỡng → fit cuối → chấm val.

## 5. Cửa sổ và nhãn

- Cửa sổ dài `1,0 s` (50 frame), bước `0,2 s` (10 frame), nằm trọn trong một luồng; không có padding.
- Cửa sổ bị loại nếu tỉ lệ frame hữu hạn thấp hơn `minimum_valid_fraction` (1,0).
- `coverage = overlap / min(độ dài cửa sổ, độ dài cú ngã)`.
  - Cửa sổ là **dương** nếu `coverage ≥ 0,5`.
  - Cửa sổ là **âm** nếu không chạm cú ngã nào. Hai khoảng chỉ chạm nhau ở đầu mút không tính là chồng lấn.
  - Mọi trường hợp còn lại là **mơ hồ** và bị loại khỏi tập huấn luyện. Khi chấm điểm thì vẫn tính score cho mọi cửa sổ.
- Cú ngã bị gắn cờ (`quality = low` hoặc `needs_review`) mặc định vẫn là dương (`flagged_event_policy: positive`).
- Trên train có 3 328 cửa sổ dương, 106 170 cửa sổ âm và 1 718 cửa sổ mơ hồ, tức tỉ lệ dương là 3,0%.

Các test trong `tests/test_windowing.py` bao phủ trường hợp chạm đầu mút, chồng lấn một phần, cú ngã ngắn nằm trong cửa sổ, và chính sách cho cú ngã bị gắn cờ.

## 6. Đặc trưng

Mỗi frame được tóm tắt theo từng radar `r`:

- `bg_energy[t, r] = Σ_bin (x - EMA_τ(x))²` với `τ = 3 s`: năng lượng phần dư sau khi trừ nền tĩnh;
- `diff_energy[t, r] = Σ_bin (x_t - x_{t-1})²`: năng lượng thay đổi giữa hai frame liên tiếp;
- tùy chọn `range_centroid`: trọng tâm range của chuyển động. **Mặc định tắt**, vì nó chủ yếu mã hóa vị trí thảm.

Hai chuỗi năng lượng được lấy `log10`. Với mỗi cửa sổ và mỗi radar, pipeline tính 12 thống kê: mean, std, min, max, median, q25, q75, IQR, RMS, slope, max |Δ|, max − median. Cộng thêm max và mean qua ba radar, và độ tương phản nhân quả "mean trong cửa sổ − mean của 5 s trước đó". Tổng cộng 86 đặc trưng; tensor CIR không bao giờ bị flatten.

## 7. Mô hình

| Model | Cấu hình |
|---|---|
| `dummy` | `DummyClassifier(strategy="prior")`, làm mốc thấp nhất |
| `logistic` | `SimpleImputer` → `StandardScaler` → `LogisticRegression(class_weight="balanced", max_iter=5000)`, thử C ∈ {0,1; 1} |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, lr=0,1, max_leaf_nodes=31, early_stopping=False)` |

Không oversampling (chỉ dùng class weight). Seed là 2027. Nếu một tập train chỉ có một lớp, pipeline dùng `ConstantScorer` và ghi cảnh báo.

## 8. Ngưỡng và hậu xử lý

Cửa sổ không có score (NaN) bị bỏ trước. Score còn lại được làm mượt (trung bình trượt có tâm, độ rộng k) trên từng đoạn `w_index` liên tục, nên không bắc qua cửa sổ bị thiếu. Sau đó score được so với ngưỡng và gom thành run các cửa sổ liên tiếp. Run ngắn hơn `minimum_positive_windows` bị bỏ. Khoảng của sự kiện là `[tâm cửa sổ đầu − stride/2, tâm cửa sổ cuối + stride/2]`. Các sự kiện cách nhau ≤ `merge_gap` được gộp; sự kiện ngắn hơn `minimum_event_duration` bị bỏ; có thêm cooldown, rồi cắt cho nằm gọn trong luồng.

- `score` của sự kiện là score làm mượt cao nhất trong sự kiện.
- `t_alarm` là thời điểm kết thúc cửa sổ đầu tiên vượt ngưỡng.

Toàn bộ lưới tham số (ngưỡng × làm mượt × số cửa sổ tối thiểu × gộp × độ dài tối thiểu × cooldown) được chấm bằng **f1_mean chính thức trên dự đoán out-of-fold của train**. Kết quả lưu ở `postprocess_grid_results.csv` và `threshold_sensitivity.csv`.

Hai điểm cần lưu ý:

- Điểm OOF tốt nhất vừa là tiêu chí chọn trong 144 tổ hợp, vừa là con số được báo cáo, nên nó hơi lạc quan. Muốn có ước lượng không lệch thì cần nested CV (chọn hậu xử lý ở vòng trong, chấm ở vòng ngoài).
- Ngưỡng được chọn trên score của các model fold (mỗi model học khoảng 2/3 số người), rồi áp lên model cuối học toàn bộ train. `scripts/diagnose_fold_ensemble.py --run-dir <run>` đo độ lệch giữa hai phân bố score này trên val. Script đọc nhãn val, nên kết quả chỉ để chẩn đoán, không dùng để chọn.

## 9. Metric

- **Mức sự kiện:** gọi thẳng `evaluate.py` của gói (`evaluate`, `score_events`) với `ignore_flagged = true`. Nếu không có evaluator, ví dụ khi chạy test tổng hợp, pipeline dùng một bản cài lại theo đúng quy tắc trong README và đánh dấu `"source": "provisional"`. Test `test_provisional_matches_official` kiểm tra hai bản cho kết quả giống nhau.
- **Mức cửa sổ:** precision, recall, F1 ở ngưỡng đã chọn, PR-AUC, ROC-AUC, confusion matrix, balanced accuracy. Nếu tập chỉ có một lớp thì trả về `None` kèm cảnh báo.
- **Theo nhóm:** f1_mean theo từng luồng val (chính thức), recall theo `dataset` và `fall_type`. Phần chia nhóm do nhóm tự đặt nên được đánh dấu provisional.

## 10. Đọc chẩn đoán overfit/underfit

`final_metrics.json → models.<m>.diagnostics` liệt kê các bằng chứng sau:

| Bằng chứng | Gợi ý |
|---|---|
| PR-AUC in-sample và OOF đều thấp | Underfit, hoặc đặc trưng/nhãn yếu: kiểm tra lại nhãn, trục tín hiệu và cách cân bằng lớp trước khi tăng độ phức tạp |
| In-sample ≫ OOF | Model nhớ người trong tập train (overfit hoặc variance cao) |
| Std của PR-AUC giữa các fold lớn | Không ổn định giữa các người |
| Val ≫ OOF, hoặc điểm quá cao (> 0,95) | Kiểm tra rò rỉ trước khi kết luận |
| Val ≪ OOF | Domain shift giữa các người |

Thêm vào đó:

- `learning_curve.csv`: PR-AUC trên phần held-out khi chỉ dùng 50% hoặc 100% số người trong train;
- `train_summary.json → junction_diagnostic`: tỉ lệ báo động giả nằm gần chỗ nối giữa các đoạn ghép.

## 11. Artifact của một run

`outputs/experiments/<YYYYmmdd-HHMMSS>_<name>/` (không bao giờ ghi đè run cũ):

- **Cấu hình và môi trường:** `resolved_config.yaml`, `environment.json` (Python, thư viện, git commit/branch; `dirty` chỉ xét file đã được Git theo dõi, kèm danh sách file đã sửa và file chưa theo dõi), `runtime.json`, `warnings.json`, `run.log`. Run dùng cho báo cáo phải có `dirty: false`, để truy ngược được đúng commit.
- **Dữ liệu và split:** `data_validation.json`, `manifest_snapshot.csv`, `segments_snapshot.csv`, `split_assignments.csv` (luồng → split, nhóm → split/fold, seed, khóa group), `feature_schema.json`.
- **Model và tham số đã chọn:** `models/<model>.joblib`; `selection.json` (tham số model, ngưỡng, hậu xử lý, dữ liệu dùng để chọn).
- **Bảng metric:** `fold_metrics.csv`, `postprocess_grid_results.csv`, `threshold_sensitivity.csv`, `pr_curve_oof_<model>.csv`, `learning_curve.csv`.
- **Dự đoán:** `window_predictions_train.csv.gz` (score OOF và in-sample), `window_predictions_val.csv.gz`; `event_predictions_train_oof_<model>.csv`, `event_predictions_val_<model>.csv`, `submission_val.csv` (model được chọn, đúng định dạng nộp).
- **Tổng hợp:** `train_summary.json`, `final_metrics.json`.

Run không lưu CIR thô hay bản sao dữ liệu. `outputs/` (trừ các README) bị `.gitignore` loại khỏi Git.

## 12. Giả định chưa xác nhận và giới hạn

- `fall_person_k` và `adl_subject_n` là mã người: suy ra từ việc chúng tách rời giữa train và val, chưa được ban tổ chức xác nhận.
- Dùng `streams_composition.csv` để chia fold: chỉ dùng khi train, không dùng làm đặc trưng.
- Các tham số cửa sổ, nhãn, đặc trưng và lưới hậu xử lý là lựa chọn của nhóm, không thuộc protocol.
- Chỉ có 3 fold với 12 nhóm, nên phương sai giữa các fold còn lớn.
- Phân tích theo `dataset` và `fall_type` dùng bản ghép provisional.
- Còn các đường tắt của bộ dữ liệu: vị trí thảm, phần dư EMA sau chỗ nối, tỉ lệ frame lặp. Xem hai báo cáo.
