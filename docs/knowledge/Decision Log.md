---
type: project-decisions
project: ABC2027 UWB-FallTrack
updated: 2026-10-20
tags:
  - abc2027
  - uwb
---

# Nhật ký quyết định

## 2026-10-08 — Tập trung vào UWB-FallTrack

**Quyết định:** Định hướng dự án vào bài toán phát hiện ngã liên tục từ luồng radar UWB.

**Lý do:** Đây là hướng challenge nhóm chọn cho project ABC 2027.

**Hệ quả:** Xác nhận protocol và dữ liệu trước khi chốt biểu diễn tín hiệu, split, mô hình hoặc cách đánh giá. Thay mọi giả định tạm bằng yêu cầu có nguồn từ ban tổ chức.

## 2026-10-19 — Đưa ghi chú Obsidian vào repository

**Quyết định:** Dùng thư mục gốc repository làm Obsidian vault; lưu ghi chú dùng chung trong `docs/knowledge/`.

**Lý do:** Ghi chú, code và tài liệu chuẩn cần cùng một nguồn có thể xem xét và quản lý bằng Git.

**Hệ quả:** Cấu hình giao diện Obsidian là cục bộ; nội dung Markdown được quản lý chung. Xem quy tắc trong [AGENTS.md](../../AGENTS.md) và [GUIDELINES.md](../../GUIDELINES.md).

## 2026-10-20 — Khảo sát nguồn và gói dữ liệu trước khi dựng pipeline

**Quyết định:** Chia vòng khảo sát ban đầu thành ba nhánh đọc độc lập: kiểm kê code trong repo và package; đọc tài liệu/giấy phép package cùng metadata tệp; tra cứu trang challenge ABC/UQAC chính thức. Không giao vai trò riêng để review hoặc chấm kết quả của thành viên khác.

**Lý do:** Mission được cung cấp kèm theo chứa chi tiết protocol chưa có trong tài liệu chuẩn của repo. Cần phân biệt tuyên bố của package với điều ban tổ chức đã công bố trước khi chọn metric hay split.

**Kết quả:** Nguồn challenge xác nhận bài toán liên tục và định vị ranh giới sự kiện, nhưng chưa xác nhận ngưỡng/cách xếp hạng của metric. README package mô tả chi tiết hơn và ghi LICENSE CC BY-NC 4.0; các chi tiết đó chỉ được dùng như protocol thực nghiệm cục bộ. Xem [đặc tả challenge](../challenge_spec.md).

## 2026-10-20 — Giữ validation ngoài quá trình chọn mô hình

**Quyết định:** Nếu dựng prototype theo package cục bộ, huấn luyện/chọn cấu hình trên OOF của các stream train, nhóm theo `stream_id`. Dùng `evaluate.py` đi kèm package làm metric thực nghiệm tham khảo với `--ignore-flagged`; không gọi metric mean event F1 0.70 là mục tiêu xếp hạng chính thức. Chỉ chấm validation sau khi đã khóa cấu hình/mô hình; không dùng kết quả đó để chọn lại đặc trưng, ngưỡng hay tham số.

**Lý do:** Trang challenge chính thức chưa công bố công thức đầy đủ của metric. Việc lặp lại xem điểm validation rồi thay mô hình sẽ phá tính độc lập của holdout.

**Hệ quả:** Kết quả được ghi nhãn là package-local, không đại diện cho điểm challenge chính thức. Không dùng `streams_composition.csv` trong inference/predict; nếu cần phân tích provenance hoặc lỗi, chỉ dùng trong bước audit train được ghi riêng.

## 2026-10-20 — Đóng băng một lần đánh giá holdout package-local

**Quyết định:** Chọn HistGradientBoosting + quantum-inspired density features theo OOF train (mean F1 package-local 0.7322) và chạy một lần final holdout. Lưu cấu hình, dự đoán và report riêng; không dùng điểm validation để chỉnh mô hình.

**Kết quả:** Holdout package-local đạt mean event F1 0.6306; mục tiêu nội bộ 0.70 chưa đạt. Metric này chưa được xác nhận là ranking chính thức, xem [đặc tả challenge](../challenge_spec.md) và [[Experiment Log]].

**Hệ quả:** Năm stream validation đã được dùng làm holdout cuối cho cấu hình này. Không dùng lại chúng để lựa chọn mô hình; cần một holdout mới hoặc đánh giá test chính thức để xác nhận phiên bản cải tiến mà không rò rỉ.

## 2026-10-20 — Thử profile tĩnh causal như nhánh state lượng tử bổ sung

**Quyết định:** Thử một nhánh contrast CIR so với EMA causal tau=8 s, reset nền theo từng stream và ghép motion/static amplitude vào state 5-qubit. Đánh giá bằng nested GroupKFold chỉ trên train; không dùng composition trong feature extraction hoặc inference.

**Lý do:** Nhánh motion-only bỏ qua thay đổi profile còn tồn tại khi chuyển động tức thời suy yếu sau landing. Đây là giả thuyết cần kiểm tra vì profile tĩnh cũng có thể tạo shortcut theo môi trường/segment.

**Kết quả:** Nested train OOF với HistGradientBoosting base đạt mean event F1 0.8000, F1@0.1/0.3/0.5 = 0.8623/0.8260/0.7117; audit train có 13 FP ở ADL. Nhánh readout regularization lồng đạt 0.7844, với ít FP hơn (33 so với 40) và start error thấp hơn (0.18 s so với 0.20 s). Full-train OOF candidate đạt 0.8087 nhưng điểm này có selection optimism. Chi tiết ở [[Experiment Log]].

**Hệ quả:** Lưu cấu hình ứng viên tại [quantum_histgb_static_final_candidate.yaml](../../configs/quantum_histgb_static_final_candidate.yaml). Không dùng lại validation đã tiêu thụ và chưa tạo submission; cần holdout mới hoặc evaluator test chính thức để xác nhận mốc 0.70.

## 2026-10-20 — Chọn weighting và boundary shift bên trong nested OOF

**Quyết định:** Thêm lựa chọn gamma static/motion và dịch chuyển start/end vào inner stream-grouped OOF, rồi chấm outer train folds. Giữ alarm time theo thời điểm decoder phát hiện trước khi dịch biên.

**Kết quả:** Nested train OOF đạt mean F1 0.8228, F1@0.1/0.3/0.5 = 0.8737/0.8421/0.7526; tại 0.3 có 34 FP/26 FN. Kết quả tăng so với r7 nhưng không xác nhận mục tiêu validation; năm stream validation trước đó đã tiêu thụ.

**Hệ quả:** Giữ đây là kết quả development-only, chưa khóa thành submission. Sai số train còn cho thấy phần cuối event dài thường muộn; bước kế tiếp sẽ so sánh exit penalty bất đối xứng và positive weight=4 chỉ trên inner folds trước khi outer scoring.

## 2026-10-20 — Giữ decoder đối xứng làm ứng viên train-only

**Quyết định:** Không thay r9 bằng thử nghiệm positive weight=4 và asymmetric exit penalty r10.

**Lý do:** r10 có 32 FP/28 FN tại tIoU 0.3 và false alarms/hour thấp hơn nhẹ, nhưng nested mean F1 giảm từ 0.8228 xuống 0.8174, F1@0.5 từ 0.7526 xuống 0.7340, còn median absolute start error tăng từ 0.17 s lên 0.18 s. Exit penalty thấp hơn entry không được chọn ổn định.

**Hệ quả:** r9 là ứng viên train-only tốt nhất hiện tại, không phải submission đã xác nhận. Mục tiêu validation ≥0.70 chưa được xác nhận cho candidate mới; validation cũ không được tái sử dụng.

## 2026-10-20 — Thử duration profile từ inner OOF

**Quyết định:** Dùng cutoff 1.8 s và end offsets có điều kiện làm ứng viên tiếp theo trong inner OOF, với profile uniform làm anchor.

**Kết quả:** Nested train OOF r11 đạt mean F1 0.8342 và F1@0.5 0.7801, cao hơn r9 (0.8228 và 0.7526). False alarms/hour tăng nhẹ từ 5.50 lên 5.66; không có đủ bằng chứng để xem đây là cải thiện ngoài train.

**Hệ quả:** r11 là train-only candidate tốt nhất hiện tại; chỉ được xác nhận trên holdout mới hoặc official evaluator. Validation đã dùng trước đó cho mô hình cũ và không được mở lại.


## 2026-10-20 — Tính cutoff duration từ inner OOF quantile

**Quyết định:** Thử profile biên adaptive q25/q75, với cutoff dự đoán tính từ inner OOF trong từng outer fold; mọi thành phần được khóa trước khi chấm outer fold.

**Kết quả:** R12 đạt nested train OOF mean F1 0.8355 và F1@0.1/0.3/0.5 = 0.8825/0.8460/0.7781. So với r11, mean F1 tăng 0.0013; F1@0.5 giảm 0.0020; median absolute start error tăng từ 0.16 lên 0.18 s; false alarms/hour giữ 5.66. Audit ở theta 0.5 cho thấy end error median giảm từ +0.20 xuống +0.12 s và duration ratio giảm từ 1.139 xuống 1.070.

**Hệ quả:** R12 có mean OOF cao nhất đến nay nhưng mức chênh nhỏ và tie-breakers hỗn hợp; giữ là baseline số học cho vòng tiếp theo, chưa xem là cải thiện tổng quát hóa hay validation score. Không dùng lại validation đã tiêu thụ và chưa tạo submission. Vòng kế tiếp sẽ chỉ thử fidelity lag1−lag3 trên train nested OOF.


## 2026-10-20 — Không giữ delta fidelity lag1−lag3

**Đội ngũ tư vấn chu kỳ:** Dynamics feature advisor và nested OOF/readout advisor đề xuất một paired ablation train-only; không tạo vai trò evaluator riêng. Tôi triển khai và chạy nested GroupKFold.

**Quyết định:** Không bật ba cột delta fidelity trong ứng viên train-only kế tiếp. Mã cho phép bật/tắt để tái lập ablation.

**Lý do:** R13 đạt mean F1 0.8316 và F1@0.5 0.7684, giảm tương ứng 0.0039 và 0.0096 so với r12. Delta được inner OOF chọn ở 2/5 outer folds; cả hai fold đều có outer mean F1 thấp hơn r12. F1@0.3 tăng 0.0014 và false alarms/hour giảm 0.32, nhưng chưa bù được giảm ở metric chính và tIoU 0.5.

**Hệ quả:** R12 tiếp tục là ứng viên train-only có mean OOF cao nhất (0.8355), chưa phải validation result. Thử nghiệm tiếp theo chỉ dùng train GroupKFold; validation cũ không được mở lại.


## 2026-10-20 — Chọn end trim adaptive q25/q75 làm ứng viên train-only

**Đội ngũ tư vấn chu kỳ:** Dynamics feature advisor khuyến nghị không thêm feature sau ablation r13; nested OOF/readout advisor đề xuất một decoder-only profile với end trim thêm 0.1 s. Không có evaluator riêng.

**Quyết định:** Giữ r14 làm ứng viên train-only đứng đầu hiện tại; thêm adaptive q25/q75 profile với short/middle/long end offsets −0.1/−0.3/−0.4 s.

**Lý do:** R14 đạt mean F1 0.8390 và F1@0.5 0.7885 so với 0.8355/0.7781 ở r12. F1@0.3, median absolute start error và false alarms/hour không đổi. Profile được chọn ở 2/5 outer folds; các fold đó cải thiện F1@0.5, ba fold còn lại khớp r12. Audit cho thấy signed end error giảm nhẹ về 0 và duration ratio tiến gần 1.0.

**Hệ quả:** R14 là development-only train OOF, chưa xác nhận mục tiêu validation ≥0.70. Không dùng lại validation đã tiêu thụ và chưa tạo submission; cần holdout mới hoặc official evaluator để xác nhận.

## 2026-10-20 — Gia cố split guard và chờ test mới

**Đội ngũ tư vấn chu kỳ:** OOF split-safety advisor rà soát ranh giới fit/held-out và predictor; official protocol researcher kiểm tra trang challenge công khai. Không tạo vai trò evaluator riêng.

**Quyết định:** Bổ sung runtime assertion cho namespace train, phân hoạch outer/inner folds, scope dữ liệu dùng khi fit/decoder selection và độ phủ OOF. Chặn `predict_holdout.py` khi trỏ tới thư mục validation đã tiêu thụ hoặc đặt split name là `val`/`validation`; yêu cầu đường dẫn holdout/test tường minh. Cập nhật audit để kiểm tra lại fold report và coverage logits trước khi đọc nhãn/composition.

**Lý do:** Review source phát hiện `make_group_folds` chưa xác nhận partition, nested decoder nhận map events toàn train dù chỉ chấm `outer_train`, và lệnh prediction mặc định vào validation với guard overwrite theo từng output folder. Những điểm này cần fail-fast để dễ chứng minh split isolation.

**Kết quả nguồn:** Trang ABC 2027 ghi lịch gửi test cho người tham gia vào 2026-10-20, nhưng không có liên kết test công khai; gói cục bộ chỉ có train/validation và receipt qua kênh riêng chưa được xác nhận. Không tìm thấy public evaluator/results endpoint. Chi tiết và nguồn ở [đặc tả challenge](../challenge_spec.md).

**Hệ quả:** Không mở lại validation và chưa tạo dự đoán submission. R14 vẫn là kết quả train nested OOF tốt nhất (mean F1 0.8390), chưa xác nhận mốc mục tiêu trên một holdout mới. Cần có test package/materials mới hoặc holdout mới được cấp phép để đánh giá cấu hình đã khóa.

## 2026-10-20 — Không giữ end trim tăng thêm ở R15

**Đội ngũ tư vấn chu kỳ:** Dynamics feature advisor và nested OOF/readout advisor xem audit R14 và đề xuất một decoder-only ablation; không có evaluator riêng.

**Quyết định:** Không giữ `adaptive_q25_q75_end_minus_0_2`, profile trim end thêm 0.1 s trên cả ba duration bin.

**Lý do:** Inner OOF không chọn profile mới ở bất kỳ outer fold nào. R15 chạy qua các assertion partition/scope mới và khớp R14 ở cả 5 outer fold: mean F1 0.8390, F1@0.1/0.3/0.5 = 0.8825/0.8460/0.7885; theta 0.3 vẫn TP/FP/FN 162/35/24, start error 0.18 s và FA/h 5.66. Audit train cũng giữ nguyên median signed end error +0.10 s và median absolute end error 0.20 s.

**Hệ quả:** R14 vẫn là train-only candidate đứng đầu. Không tiếp tục tinh chỉnh end offset; mọi OOF score mới tiếp tục là development evidence do các vòng lặp đã dùng các train folds để định hướng giả thuyết. Xem [[Experiment Log]] cho artifact và lệnh tái lập R15.

## 2026-10-20 — Thử normalized density-matrix temporal similarity R16

**Đội ngũ tư vấn chu kỳ:** Dynamics feature advisor đề xuất normalized Hilbert–Schmidt similarity với lag một bước và reset state theo stream; nested OOF/readout advisor cảnh báo feature này có thể trùng thông tin với temporal overlap/purity hiện có và khuyến nghị dừng chọn mô hình trên cùng các fold. Không giao vai trò evaluator riêng.

**Quyết định:** Ghi R16 như một ablation train-only có kiểm soát, không xem kết quả là đánh giá độc lập. Giữ R14 làm anchor và R16 làm ứng viên có mean nested OOF cao nhất đến hiện tại.

**Kết quả:** R16 đạt mean F1 0.8412 và F1@0.1/0.3/0.5 = 0.8796/0.8482/0.7958, so với R14 0.8390 và 0.8825/0.8460/0.7885. Feature được inner selection chọn ở 2/5 folds. Audit train-only ghi TP/FP/FN 162/34/24 tại theta 0.3 và tất cả assertion namespace, partition, scope, coverage đều đạt; validation không được đọc và composition chỉ được dùng cho audit train.

**Lý do:** Cải thiện mean (+0.0022) và F1@0.5 (+0.0073) nhỏ; F1@0.1 giảm 0.0029, còn tie-breaker start error không đổi. Do nhiều vòng lặp đã dùng cùng 13 stream train để hình thành giả thuyết, điểm nested OOF hiện có selection optimism ở cấp độ experiment.

**Hệ quả:** Không mở lại 5 stream validation đã dùng làm final holdout sau baseline R2 và không tiếp tục tinh chỉnh theo các outer fold này. Mốc validation nội bộ ≥0.70 vẫn chưa được xác nhận cho R14/R16; cần test package mới hoặc holdout độc lập chưa dùng. Xem [[Experiment Log]] và [đặc tả challenge](../challenge_spec.md).


## 2026-10-20 — Chốt cấu hình ứng viên triển khai R17 trên train OOF

**Quyết định:** Sinh cấu hình dự đoán holdout R17 từ lượt 5-fold GroupKFold OOF trên 13 stream train. Chọn `gamma=2.0` cùng normalized rho similarity và decoder `adaptive_q25_q75_end_minus_0_1` vì cấu hình này có ranking score train OOF cao nhất trong bốn biến thể đã định trước.

**Kết quả train-only:** Mean event F1 0.8520; F1@0.1/0.3/0.5 = 0.8930/0.8663/0.7968. Tại theta 0.3: TP/FP/FN 162/26/24, median absolute start error 0.16 s và false alarms/hour 4.21. Cả `validation_used` và `composition_file_used` đều false.

**Giới hạn:** Đây là lựa chọn feature/decoder trên các OOF predictions của cùng 13 stream sau nhiều vòng phát triển; điểm có selection optimism và không thay thế validation/test độc lập. Metric của package evaluator chưa được xác nhận là công thức xếp hạng cuối cùng; xem [đặc tả challenge](../challenge_spec.md).

**Hệ quả:** [quantum_histgb_final_candidate_r17.yaml](../../configs/quantum_histgb_final_candidate_r17.yaml) trở thành cấu hình mặc định cho `predict_holdout.py`. Chưa tạo submission vì package cục bộ không có test stream; không tái sử dụng validation đã tiêu thụ. Tiếp theo cần holdout mới hợp lệ hoặc official evaluator.


## 2026-10-20 — Chấm một lượt validation diagnostic cho R17

**Quyết định:** Theo yêu cầu của người dùng, chạy R17 với cấu hình đã khóa trên năm stream validation, chỉ để có một kết quả chẩn đoán. Không thay đổi feature, decoder, ngưỡng hoặc mô hình dựa trên kết quả này; không chạy thêm validation.

**Lý do:** Cần đối chiếu kết quả OOF với các stream validation sẵn có trong lúc test challenge chưa được nhận. Validation đã được dùng cho baseline R2 nên không còn là holdout hoàn toàn độc lập.

**Kết quả:** Package evaluator cục bộ với `--ignore-flagged` cho mean event F1 **0,6957**; F1@0,1/0,3/0,5/0,7 = **0,8116/0,7536/0,5217/0,2754**. Tại theta 0,3: TP/FP/FN **52/19/15**, precision **0,7324**, recall **0,7761**, false alarms/hour **9,27**, median absolute start/end error **0,52/0,41 s**. So với R2 trên cùng validation/evaluator (mean F1 0,6306; F1@0,3 0,6757; FA/h 15,13), R17 tốt hơn về F1 và báo động giả nhưng vẫn có khoảng cách lớn giữa OOF và validation, nhất là ở tIoU 0,5. Metric package chưa được xác nhận là xếp hạng chính thức.

**Hệ quả:** Xem mean F1 0,6957 là chẩn đoán hậu kiểm, không phải kiểm định độc lập hoặc kết quả chính thức. Không chỉnh R17 từ điểm này; cần test mới hoặc official evaluator để xác nhận.


## 2026-10-20 — Cập nhật mục tiêu phát triển nội bộ lên F1 0,75

**Quyết định:** Theo yêu cầu mới của người dùng, nâng mục tiêu phát triển nội bộ lên F1 ít nhất 0,75 và giữ F1 giữa các mốc tIoU gần nhau. Cách diễn giải tạm trong vòng phát triển là F1@0.1/0.3/0.5 đều ≥0.75 và max−min ≤0.10; đây không phải protocol chính thức và sẽ được nêu rõ khi báo cáo.

**Bằng chứng hiện có:** Nested train-only R16 đạt 0.8796/0.8482/0.7958 (trung bình 0.8412, độ vênh 0.0838), đáp ứng cách diễn giải tạm trên outer folds. R17 đạt 0.6957 trên validation diagnostic đã được xem trước đó; kết quả này không đạt mục tiêu mới và không được dùng để chọn lại cấu hình.

**Hệ quả:** Tiếp tục thử nghiệm chỉ trên nested GroupKFold theo `stream_id` trong train. Không đọc lại nhãn validation để chọn mô hình/ngưỡng; không tuyên bố đã đạt mục tiêu generalization cho tới khi có holdout/test mới hoặc official evaluator.


## 2026-10-20 — Giữ quantum reservoir R18 ở chế độ opt-in

**Quyết định:** Thêm reset-channel quantum reservoir causal như feature opt-in; không thay cấu hình final đã khóa. Giữ bốn feature variant R16 và ghép thêm bốn variant reservoir để so sánh trên cùng nested GroupKFold folds.

**Kết quả train-only:** Mean nested OOF F1 tăng nhẹ từ 0,8412 (R16) lên 0,8447 (R18). F1@0,1/0,3 tăng 0,0105/0,0052; F1@0,5 giảm 0,0052. Median absolute start error giảm 0,02 s và FA/h giảm khoảng 0,16. Inner selection chọn reservoir ở 4/5 folds nhưng fold-wise delta mean F1 trái chiều; cùng train streams đã dùng qua nhiều vòng phát triển nên kết quả còn selection optimism.

**Lý do:** Reservoir có tín hiệu triển vọng nhưng mức tăng aggregate nhỏ và trade-off ở tIoU 0,5 chưa đủ cơ sở để thay candidate đã khóa. Việc triển khai inference path chỉ đảm bảo có thể bật feature bằng config; không xem đó là quyết định phát hành mô hình.

**Kiểm soát leakage:** R18 report ghi `validation_used=false` và `composition_file_used=false`; assertion kiểm tra 13 train IDs, phân hoạch outer/inner folds, coverage logits và prediction IDs train-only đều đạt. Không đọc validation stream/labels hay `streams_composition.csv`.

**Hệ quả:** Giữ `include_quantum_reservoir` mặc định false để tương thích pipeline hiện tại; `predict_holdout.py` truyền các tham số reservoir và feature names khi bật config. Không sinh final candidate hoặc submission từ R18. Xem [[Experiment Log]] cho config, lệnh, paired metrics và artifact.


## 2026-10-20 — Không promote causal semi-Markov R17

**Quyết định:** Giữ decoder Ising/DP hiện tại; không thêm causal semi-Markov vào đường inference.

**Kết quả:** Nested train-only OOF đạt F1@0.1/0.3/0.5 = 0,7532/0,7013/0,5714, mean 0,6753. R16 đạt 0,8796/0,8482/0,7958, mean 0,8412 trên cùng các stream và split. Semi-Markov kém rõ rệt ở cả ba ngưỡng, nhất là tIoU 0,5.

**Kiểm soát leakage:** Report xác nhận chỉ dùng 13 train streams; outer/inner folds và duration-prior fitting nằm trong scope train tương ứng. Validation và `streams_composition.csv` không được dùng.

**Hệ quả:** Không thay `predict_holdout.py`; giữ kết quả làm ablation âm tính và ghi chi tiết trong [[Experiment Log]]. Tiếp tục tìm cải thiện bằng train-only nested OOF hoặc chờ holdout độc lập mới.


## 2026-10-20 — Không promote boundary trim R22

**Quyết định:** Không bật các boundary trim profile R22 trong đường inference.

**Kết quả:** R22 thử hai adaptive profile trim start/end 0,1 s hoặc 0,2 s, chọn trong inner OOF. Tất cả năm outer fold chọn cùng readout, decoder và duration profile như R21; nested OOF metric trùng hoàn toàn với R21 (mean 0,8400; F1@0,1/0,3/0,5 = 0,8847/0,8633/0,7721; spread 0,1126).

**Kiểm soát leakage:** Report xác nhận outer/inner folds phân hoạch đúng các train ID, predictions chỉ gồm train streams, `validation_used=false`, `composition_file_used=false`.

**Hệ quả:** Offset trim cố định/adaptive không giúp trong nested selection. Tiếp tục train-only OOF với readout onset/offset riêng; không mở lại validation.


## 2026-10-20 — Chấm chẩn đoán một candidate stream-balanced trên validation theo yêu cầu

**Quyết định:** Chọn candidate bằng 5-fold OOF train-only, fit lại readout trên cả 13 stream train, dự đoán 5 stream validation và chấm đúng một lượt bằng package evaluator `--ignore-flagged`. Không dùng nhãn validation để fit, chọn feature/decoder/ngưỡng hay sửa mô hình.

**Kết quả:** R23 đạt validation mean event F1 **0,6857**, F1@0,1/0,3/0,5 = **0,8143/0,7143/0,5286**, spread **0,2857**. Kết quả thấp hơn mục tiêu nội bộ và các ngưỡng chênh nhiều; tại theta 0,3 có TP/FP/FN 50/23/17 và median absolute start/end error 0,41/0,41 s.

**Kiểm soát và giới hạn:** Metadata ghi model fit trên 13 train streams; nhãn validation chưa được đọc khi dự đoán, composition không được dùng. Evaluator chỉ đọc nhãn sau khi prediction CSV đã khóa. Tuy vậy validation đã từng được dùng cho R2 và R17, nên R23 chỉ là diagnostic, không độc lập. Metric là package-local, chưa được xác nhận là ranking chính thức.

**Hệ quả:** Không promote candidate R23 và không tái sử dụng validation cho lựa chọn tiếp theo. Tiếp tục tìm nguyên nhân/generalization bằng nested OOF train-only; cần test/holdout mới để xác nhận.


## 2026-10-20 — Không promote boundary readouts R24

**Quyết định:** Không bật onset/offset heads R24 trong candidate inference; cả năm outer fold đều chọn boundary radius 0 s. Thử nghiệm tiếp theo sẽ chọn hệ số co duration trong inner OOF.

**Kết quả train-only:** Nested OOF đạt mean F1 0,8436; F1@0,1/0,3/0,5 = 0,8847/0,8686/0,7775; spread 0,1072. Pooled F1 cả ba ngưỡng vượt 0,75, nhưng spread vượt ngưỡng tạm 0,10 và outer-fold minimum tại tIoU 0,5 là 0,7143. Vì vậy chưa xác nhận độ ổn định cần thiết.

**Kiểm soát leakage:** Report và train-only audit đều xác nhận `validation_used=false`; boundary heads fit đúng inner-train/outer-train scope và scores chỉ trên fold held-out tương ứng. `streams_composition.csv` chỉ được dùng trong error audit train, không dùng cho fit/selection.

**Lý do:** Bán kính 0 được chọn ở mọi outer fold, nên hai head không cải thiện inner OOF. Audit cho thấy median start sớm 0,08 s, end muộn khoảng 0,08–0,10 s, duration ratio median 1,1165; đây là động lực để thử co interval theo tỷ lệ trong inner OOF.

**Hệ quả:** Giữ Ising/DP và event readout hiện tại. Chưa đạt mục tiêu; validation đã tiêu thụ nên không chấm lại. Xem [[Experiment Log]] cho cấu hình, audit và artifact R24.


## 2026-10-20 — Không promote duration contraction R25

**Quyết định:** Không áp co đối xứng duration trong đường inference; cả năm outer fold chọn scale 1,00.

**Kết quả:** R25 nested OOF trùng R24: mean F1 0,8436; F1@0,1/0,3/0,5 = 0,8847/0,8686/0,7775; spread 0,1072; min outer-fold F1@0,5 = 0,7143.

**Kiểm soát leakage:** Hệ số chỉ được chọn trên inner OOF; outer fold đánh giá sau khi khóa feature/readout/decoder/scale. Train-only audit hoàn tất; validation không được đọc.

**Lý do:** Median duration ratio hơi dài nhưng co chung không giúp nested selection. Hướng tiếp theo là so sánh quantum reservoir dưới cùng objective ưu tiên F1 thấp nhất/spread và stream weighting.

**Hệ quả:** Giữ boundary radius 0 và interval scale 1. Chưa đạt mục tiêu ổn định; không dùng validation đã tiêu thụ để tinh chỉnh.


## 2026-10-20 — Không promote quantum-reservoir selection R26

**Quyết định:** Chưa promote tổ hợp reservoir/stream-weight R26 vì mục tiêu là cân bằng F1 giữa tIoU, không tối đa mean đơn lẻ.

**Kết quả:** Nested OOF mean 0,8493; F1@0,1/0,3/0,5 = 0,8989/0,8723/0,7766; spread 0,1223. Outer-fold minimum F1@0,5 là 0,7200. Reservoir được chọn ở 3/5 fold.

**Kiểm soát leakage:** Report ghi `validation_used=false`, `composition_file_used=false`; 5×4 GroupKFold trên train và prediction/logit chỉ cho train IDs. Train composition chỉ đọc trong audit lỗi.

**Lý do:** Reservoir tăng nhẹ mean và F1 ở tIoU 0,1/0,3, nhưng không tăng tIoU 0,5 và làm spread rộng hơn R25. Việc chọn reservoir không ổn định giữa outer folds.

**Hệ quả:** Trong R27, chuyển sang objective có ngưỡng mục tiêu rõ: candidate chỉ vào nhóm ưu tiên nếu F1 cả ba tIoU ≥0,75 và spread ≤0,10; nếu không đạt thì tối đa hóa min F1 rồi giảm spread. Lựa chọn chỉ trong inner OOF.


## 2026-10-20 — Chấm diagnostic R26 consensus trên validation theo yêu cầu

**Quyết định:** Dùng cấu hình đồng thuận tạo từ lựa chọn inner OOF của R26, fit readout trên toàn bộ 13 stream train, dự đoán 5 stream validation rồi chấm một lượt bằng package evaluator `--ignore-flagged`. Không dùng nhãn validation khi fit/chọn, không đọc composition, không điều chỉnh mô hình sau khi xem điểm.

**Kết quả:** Mean event F1 **0,7121**; F1@0,1/0,3/0,5 = **0,8333/0,7576/0,5455**, spread **0,2879**. Ở theta 0,3 có TP/FP/FN **50/15/17**, false alarms/hour **7,32**, median absolute start/end error **0,43/0,36 s**. Không đạt cách diễn giải mục tiêu tạm F1 ≥0,75 tại cả ba ngưỡng và spread ≤0,10.

**Kiểm soát và giới hạn:** Metadata ghi đủ 13 train streams, `prediction_labels_read=false`, `composition_file_used_for_prediction=false`; CSV được khóa trước khi evaluator đọc nhãn. Validation đã được xem ở các lượt R2/R17/R23 nên kết quả này chỉ là diagnostic, không phải holdout độc lập và không dùng để chọn lại cấu hình. Metric package-local chưa được xác nhận là xếp hạng chính thức.

**Hệ quả:** Cấu hình không được promote. Tiếp tục nested OOF train-only; chờ test/holdout mới để đo generalization độc lập.


## 2026-10-20 — R27 target-floor nested OOF

**Quyết định:** Giữ objective target-floor trong inner OOF cho vòng phát triển mới; chưa promote cấu hình vì pooled spread còn nhỉnh hơn 0,10 và outer-fold minimum F1@0,5 chỉ 0,70.

**Kết quả:** Nested OOF pooled mean F1 0,8169; F1@0,1/0,3/0,5 = 0,8563/0,8394/0,7549; spread 0,1014. Audit scope xác nhận phân hoạch, coverage và isolation đúng; `validation_used=false`, `composition_file_used=false`.

**Lý do:** Target-floor ưu tiên đã cải thiện độ cân bằng pooled so với R26 (spread 0,1223 xuống 0,1014), nhưng chưa đạt trần và hiệu quả giữa outer groups còn dao động. Error audit cho thấy median start sớm 0,10 s, end muộn 0,04 s.

**Hệ quả:** R28 kiểm tra profile offset start muộn hơn/end sớm hơn chỉ bên trong inner OOF. Không dùng nhãn validation đã tiêu thụ để chỉnh mô hình.


## 2026-10-20 — Không promote boundary shifts R28

**Quyết định:** Không promote start/end profile shifts R28 vì pooled F1@0,1/0,3/0,5 và spread trùng R27.

**Kết quả:** 0,8563/0,8394/0,7549, spread 0,1014; outer-fold minimum F1@0,5 là 0,70. Signed start/end median cải thiện nhẹ, nhưng không đổi event matches.

**Kiểm soát leakage:** Nested scope audit đạt; `validation_used=false`, `composition_file_used=false`.

**Hệ quả:** Dựa vào OOF misses không chồng lấp, R29 mở rộng positive weight và giảm ngưỡng logit candidate; vẫn chỉ chọn trong inner OOF.


## 2026-10-20 — Chấm diagnostic consensus R28 trên validation theo yêu cầu

**Quyết định:** Dùng consensus median/mode từ lựa chọn inner OOF của R28, fit readout trên cả 13 train streams, dự đoán 5 validation streams rồi chấm một lượt bằng package evaluator `--ignore-flagged`. Không dùng nhãn validation để fit/chọn và không đọc `streams_composition.csv`.

**Kết quả:** Mean event F1 **0,7218**; F1@0,1/0,3/0,5 = **0,8271/0,7820/0,5564**, spread **0,2707**. Tại theta 0,3 có TP/FP/FN **52/14/15**, precision/recall **0,7879/0,7761**, false alarms/hour **6,83**, median absolute start/end error **0,31/0,38 s**. Mean vượt mục tiêu package-local 0,70 nhưng chưa đạt yêu cầu nội bộ ba ngưỡng ≥0,75 và spread ≤0,10.

**Kiểm soát và giới hạn:** Metadata xác nhận fit trên 13 train streams, `prediction_labels_read=false`, `composition_file_used_for_prediction=false`; CSV được khóa trước khi evaluator đọc nhãn. Validation đã được dùng ở R2/R17/R23/R26 nên kết quả không độc lập và không dùng để tinh chỉnh. Audit sau đó phát hiện candidate consensus lấy từ stream-level OOF có source overlap ADL; score validation vẫn là package-local diagnostic, còn quy trình chọn candidate có thể lạc quan.

**Hệ quả:** Không promote cấu hình và không chấm lại validation. Sau lượt diagnostic, R29 được tiếp tục bằng nested OOF train-only; kết quả và quyết định promote được ghi ở mục R29 bên dưới.


## 2026-10-20 — Không promote readout sensitivity R29

**Quyết định:** Không promote R29; tiếp tục lựa chọn chỉ bằng train-only nested OOF. Không dùng metric validation R28 trong lựa chọn này.

**Kết quả:** Pooled nested OOF F1@0,1/0,3/0,5 = **0,8635/0,8468/0,7521**, mean **0,8208**, spread **0,1114**. Outer-fold minimum F1@0,5 là **0,70**. Positive weight 8 được chọn ở 1/5 fold; feature/readout selection còn thay đổi giữa các fold.

**Kiểm soát leakage:** Nested report đạt partition/coverage theo stream ID; fit/selection ghi `validation_used=false`, `composition_file_used=false`, prediction chỉ chứa 13 train IDs. Tuy nhiên source-overlap audit cho thấy **74,9–78,4%** thời lượng source ADL ở outer held-out có mặt trong source intervals của fit streams; không có interval trùng hoàn toàn nhưng các khoảng bị chồng lấn một phần. Error audit train-only dùng composition để phân loại false alarms. Audit tại theta 0,3 ghi TP/FP/FN **152/21/34** và median absolute start/end error **0,19/0,18 s**.

**Lý do:** Trước khi xét source overlap, R29 so với R27 có pooled F1@0,1/0,3 tăng nhẹ, F1@0,5 giảm nhẹ và spread xấu hơn **0,0100**. Source-level overlap làm các OOF scores không còn bằng chứng leak-free; tiêu chí nội bộ cũng chưa đạt theo score ghi nhận.

**Hệ quả:** Ghi R29 làm ablation readout, không promote. Không tiếp tục chọn mô hình từ cùng stream-level folds; trước hết cần protocol source-disjoint/purged hợp lệ hoặc holdout mới. Không dùng validation thêm để chọn model/ngưỡng.


## 2026-10-20 — Source overlap làm mất tính độc lập của stream-level OOF

**Phát hiện:** Generator lưu `source_t0/source_t1` rồi dùng chính hai mốc này để cắt CIR vào stream. Train-only composition audit cho thấy outer held-out folds tái sử dụng **74,9–78,4%** thời lượng source ADL đã có trong stream fit. Không thấy interval trùng hoàn toàn; các source intervals chồng lấn một phần. Đồ thị overlap nối cả 13 train streams thành một component.

**Ảnh hưởng:** Kiểm tra `GroupKFold(stream_id)` chỉ chứng minh một stream không xuất hiện ở cả hai phía; không ngăn các CIR samples gốc lặp qua stream tổng hợp khác. Vì vậy OOF R16–R29 không còn được coi là đánh giá độc lập hoặc bằng chứng zero leakage; điểm có thể lạc quan. Fit/predict train→validation không đọc nhãn validation, nhưng R28 candidate selection dựa trên OOF bị ảnh hưởng bởi caveat này.

**Quyết định:** Dừng promote dựa trên các stream-level OOF hiện có. Không tái chấm hoặc điều chỉnh bằng validation đã tiêu thụ. Chỉ tiếp tục model selection sau khi có source-disjoint/purged evaluation protocol phù hợp với các giới hạn dữ liệu, hoặc khi nhận holdout/test độc lập mới.

## 2026-10-20 — Không promote VQC baseline R30

**Quyết định:** Giữ VQC R30 làm baseline tham khảo, chưa promote.

**Lý do:** Source-group OOF không dùng validation/test nhưng mean event F1 chỉ **0,1668**, với **185,22** false alarms/hour tại tIoU 0,3. Ngưỡng 0,5 và decoder đơn giản là cấu hình baseline cố định; kết quả không chứng minh VQC không phù hợp sau hiệu chỉnh, nhưng hiện không đáp ứng mục tiêu.

**Hệ quả:** So sánh tiếp theo cần dùng cùng source-group folds và cùng đánh giá cho readout cổ điển; mọi hiệu chỉnh ngưỡng/decoder phải ở train folds.

## 2026-10-20 — Khởi động nâng cấp many-body R31 theo yêu cầu

**Quyết định:** Mục tiêu mới là mean event F1 validation ≥0,82. Triển khai mô phỏng many-body trên máy cục bộ, causal analytic encoding, tensor ba radar, Hamiltonian đối xứng, partial trace, readout đa đầu được điều chuẩn và exact semi-Markov DP. Lưu tiến trình/cache/model trong `ABC testing/r31/`, mã dùng lại trong `src/`, cấu hình trong `configs/`, CSV xuất trong `submissions/`; artifact bị Git bỏ qua.

**Protocol lựa chọn:** Không tiếp tục chọn bằng các fold 13 stream bị trùng source. Dùng train-only repack gồm 467 hash đường dẫn source, 5 outer × 3 inner folds. Mọi scaler, readout, duration prior và lựa chọn decoder chỉ dùng fit/inner scopes. Chọn final config bằng mode của các lựa chọn inner, không bằng outer score. Đây là phép đo nội bộ trên pseudo-stream và chưa chứng minh subject-disjoint. Xem [đặc tả challenge](../challenge_spec.md) cho trạng thái protocol.

**Validation:** Chỉ một lượt diagnostic sau khi khóa code/config/model/prediction bằng SHA-256, theo yêu cầu mới. Validation đã tiêu thụ không trở lại thành holdout độc lập. Không chạy vòng lặp lựa chọn theo điểm validation; không dùng test hoặc composition khi inference.

**Giải thích toán học:** Pha analytic của CIR thực không được khẳng định là carrier Doppler phase đo bằng I/Q. Rối lượng tử là rối trong hệ mã hóa được mô phỏng. Dùng CPTP injection channel cho memory; tổng hai statevector rồi chuẩn hóa không phải channel quantum tổng quát. Entropy subsystem của mixed state là mixedness; dùng pure-state Schmidt và mixed-state negativity để nói về entanglement. Coherence phụ thuộc basis, không bất biến dưới mọi unitary; phân biệt nó với spectrum và fidelity. Không có bảo đảm các phép toán này tự làm F1 tăng.


## 2026-10-20 — Không promote many-body R31

**Kết quả:** Source-group nested OOF mean F1 0,600522; signal-readout ablation 0,601667 gần như ngang; quantum-readout-only ablation 0,003546 với cấu hình được chọn cho full pipeline. Chưa có lợi ích tăng F1 của quantum observables trong readout.

**Validation:** Pipeline/config/model SHA-256 khóa trước prediction; CSV khóa trước evaluator. Lượt diagnostic duy nhất cho mean F1 0,0, TP/FP/FN 0/25/67 tại tIoU0,3. Không đạt mục tiêu0,82 và kém R28 diagnostic0,7218; không promote hoặc thay cấu hình cũ.

**Hệ quả:** Giữ nguyên mọi lock/artifact R31 để tái lập. Không tune từ validation và không chấm lại cùng validation để săn điểm. Kiểm tra timestamp/schema và context mismatch chỉ trên train: source-group repack thay đổi điểm reset nền/reservoir của clip ngã so với continuous streams. Đây mới là giả thuyết, cần diagnostic train gốc trước khi kết luận. Entanglement/partial trace đúng toán học không tự bảo đảm giá trị dự đoán hoặc generalization.


## 2026-10-20 — Sửa reset/context confounding bằng R32 train-only

**Bằng chứng:** R31 đạt 0,6005 source-repack OOF nhưng chỉ 0,0391 trên 13 train stream gốc khi áp model/ngưỡng đã khóa. Cả 186 valid fall của repack xảy ra trong 30 s đầu, so với 5 ở train gốc; feature shift lớn, còn mapping 359 nhãn và raw CIR của 26 clip đã được kiểm tra.

**Quyết định:** Tạo 13 continuous context bundle từ train clips, mỗi source file group thuộc đúng một bundle; nested CV 5×3 theo bundle với source union disjoint. Giữ lõi quantum và R31 artifact bất biến. Không chấm lại validation.

**Giới hạn:** Participant/session mapping còn thiếu; synthetic context không thay official stream. Validation R31 đã tiêu thụ và không dùng score đó để chọn R32.


## 2026-10-21 — Không promote many-body R32

**Kết quả:** R32 train-only source-disjoint nested OOF đạt mean event F1 **0,6305** (F1@0,1/0,3/0,5 = **0,6770/0,6512/0,5633**). Ở theta 0,3 có **12,82 false alarms/hour**. Original train in-sample diagnostic đạt 0,6649 nhưng không phải estimate generalization.

**Ablation:** Signal-only readout + QFI decoder đạt mean **0,6496**, cao hơn full pipeline 0,6305; quantum-only readout + QFI decoder đạt **0,1238**. Hyperparameters giữ nguyên cấu hình chính, nên đây là phép đo có điều kiện, không phải giới hạn tổng quát của quantum models.

**Quyết định:** Không promote R32. Mục tiêu validation ≥0,82 chưa đạt và chưa có validation R32. Cho phép hoàn tất duy nhất thí nghiệm R33 multiscale đã khai báo trước, dựa trên train-only nested CV. Không mở vòng chọn dựa trên validation đã tiêu thụ.

**Giới hạn:** OOF source-path-disjoint nhưng chưa participant/session-disjoint; evaluator package-local và dữ liệu là synthetic continuous bundles. Cần holdout/test mới, độc lập và có mapping người tham gia rõ để kiểm tra generalization.


## 2026-10-21 — Không promote multiscale R33

**Kết quả:** R33 train-only nested OOF đạt mean event F1 **0,6153**, thấp hơn R32 **0,6305**. F1@0,1/0,3/0,5 giảm từ **0,6770/0,6512/0,5633** xuống **0,6617/0,6219/0,5622**. Ở theta 0,3, false alarms tăng từ **75** lên **91** (12,82→15,56/giờ), trong khi TP 126→125 và FN 60→61.

**Phân tích:** Cửa sổ 1,2 s được inner CV chọn ở cả 5 outer fold, nhưng OOF held-out không cải thiện. Original-train in-sample tăng 0,6649→0,6799 cùng lúc OOF giảm; đây không phải bằng chứng tổng quát hóa. Readout tăng từ 660 lên 1.100 cột trên cùng 24k mẫu và 13 bundle, phù hợp với giả thuyết tăng khớp train mà không tăng phân biệt theo group.

**Quyết định:** Dừng R33 và không promote. Không chạy validation vì đã được dùng nhiều lần; không mở thêm lượt tìm đặc trưng/ngưỡng trên cùng dữ liệu train nhằm đuổi mục tiêu validation 0,82.

**Điều kiện tiếp tục:** Cần holdout/test mới và participant/session mapping để đánh giá độc lập. Lõi mô phỏng có cơ học lượng tử hợp lệ về số học nhưng được kích thích bởi CIR cổ điển; chưa có bằng chứng phase I/Q đo thực, QPU hay quantum advantage. R32 ablation cho thấy signal-only readout nhỉnh hơn full, quantum-only rất yếu.
