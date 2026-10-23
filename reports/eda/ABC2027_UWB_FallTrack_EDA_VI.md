# ABC2027 UWB-FallTrack: cấu trúc và khám phá dữ liệu (EDA)

## Tóm tắt nhanh

Bài toán yêu cầu hệ thống theo dõi radar liên tục, tìm từng cú ngã và ước lượng lúc cú ngã bắt đầu, kết thúc. Người làm không được nhận sẵn các đoạn tín hiệu đã cắt quanh cú ngã. Mỗi luồng dữ liệu trộn cú ngã với những hoạt động thường ngày như đi bộ, ăn uống và dọn dẹp.

Các số liệu chính từ những tệp hiện có:

- Có **18 luồng dữ liệu**: 13 luồng train dài tổng cộng **6,18 giờ**, 5 luồng validation dài **2,05 giờ**. Các bảng CSV có **478 cú ngã được gắn nhãn** và **2.346 đoạn nguồn** được ghép thành luồng.
- Cú ngã thường ngắn: một nửa dài từ **1,26 đến 2,20 giây**; thời lượng ở giữa là **1,74 giây**. Khoảng dài nhất là 5 giây. Vì vậy, mô hình cần ước lượng thời điểm bắt đầu và kết thúc, thay vì chỉ trả lời “có ngã trong đoạn này”.
- Có **225/478 nhãn** bị gắn cờ chất lượng thấp hoặc cần xem lại. Khi chấm giống cách dùng trên test, những nhãn này được bỏ qua. Khi đó còn **253 sự kiện được tính điểm**: 186 train và 67 validation.
- Tệp dự đoán mẫu đạt F1 **0,358 / 0,338 / 0,249 / 0,060** tại các mức yêu cầu trùng khớp thời gian 0,1 / 0,3 / 0,5 / 0,7. Kết quả giảm rõ khi yêu cầu dự đoán phải khớp sát hơn với biên cú ngã.
- Hướng nên thử trước: dùng một cách đơn giản để tìm nhiều khoảng có thể là ngã; dùng mô hình nhỏ đọc tín hiệu của ba radar theo thời gian để định vị biên; sau đó dùng bộ lọc thứ hai để loại các khoảng sinh hoạt thường ngày dễ nhầm.
- Bản giải nén hiện có `starter_extra.h5` nhưng chưa có các luồng HDF5 liên tục trong `streams/train/` và `streams/val/`. ZIP dữ liệu liệt kê 18 luồng này cùng các segment còn lại, nhưng chưa được giải nén đầy đủ vào workspace. Vì vậy chưa thể kiểm tra tín hiệu của các luồng train/validation trực tiếp từ bản giải nén hiện tại.

## 1. Bài toán và dữ liệu

### Bài toán thực sự cần giải

Ba radar siêu rộng băng (UWB) gắn trên tường ghi lại tín hiệu phản xạ trong một căn hộ thí nghiệm. Mỗi phép đo có một dãy 368 con số cho biết tín hiệu phản xạ từ các khoảng cách khác nhau. Dãy này gọi là **đáp ứng xung kênh (CIR)**. Có ba radar; chúng được đồng bộ và ghi khoảng **50 phép đo mỗi giây**, tức cách nhau khoảng 0,02 giây.

Có thể hình dung mỗi radar liên tục ghi một bản đồ phản xạ theo khoảng cách. Khi người di chuyển, bản đồ đó thay đổi. Hệ thống phải nhận ra thay đổi nào là cú ngã, rồi đánh dấu thời điểm bắt đầu và kết thúc.

Theo README, một cú ngã bắt đầu khi chuyển động ngã khởi phát và kết thúc khi người nằm ổn định trên sàn. Thời gian nằm yên sau đó không tính vào cú ngã. Người gắn nhãn xác định biên dựa trên lúc dấu hiệu phản xạ tĩnh thay đổi rõ nhất, ở radar thể hiện thay đổi mạnh nhất. Biên có độ không chắc khoảng **±0,5 đến 1 giây**, một phần vì đồng hồ radar chỉ ghi thời gian theo giây nguyên. Bộ chấm còn báo có bao nhiêu sự kiện khớp trong phạm vi ±0,5 giây và ±1 giây; các phạm vi này không thay đổi cách tính tIoU. README cho biết phần chồng 0,20 giây khi nối đoạn không làm thay đổi biên nhãn.

**F1** là điểm tổng hợp giữa hai loại sai lầm: báo ngã khi không có ngã và bỏ sót cú ngã thật. **tIoU** đo mức độ khoảng thời gian dự đoán chồng lên khoảng đã được gắn nhãn. Mức 0,1 chấp nhận dự đoán lỏng hơn mức 0,7. Vì cú ngã ngắn, dự đoán lệch một chút có thể làm giảm mạnh độ trùng khớp.

### Các tệp hiện có

| Tệp | Nội dung |
|---|---|
| streams_composition.csv | 2.346 dòng, 10 cột. Mỗi dòng mô tả một đoạn nguồn được ghép vào một luồng dữ liệu. Đây là thông tin về cách tạo luồng, không chứa tín hiệu radar. |
| streams_events.csv | 478 dòng, 12 cột. Mỗi dòng mô tả một cú ngã đã gắn nhãn: thời gian bắt đầu/kết thúc, loại cú ngã và cờ chất lượng. |
| sample_submission.csv | 140 dự đoán mẫu trên 5 luồng validation. Theo README, đây là kết quả của cách dò dựa trên năng lượng tín hiệu. |
| README.md | Quy tắc bài toán, cách chấm, định dạng tệp, các phương pháp mốc và giới hạn dữ liệu. |
| pyproject.toml, requirements.txt | Cấu hình môi trường Python và thư viện cần cài. |
| LICENSE | Giấy phép CC BY-NC 4.0: cần ghi nguồn và không được dùng thương mại theo giấy phép này. |

Gói dữ liệu đầy đủ được trang challenge mô tả có dung lượng khoảng **8,2 GB**. Tệp tín hiệu trong gói thường là HDF5, một dạng tệp chứa nhiều dữ liệu số. README mô tả mảng CIR có kích thước **(3, T, 368)**, kiểu số thực 32-bit (float32): ba radar, T thời điểm ghi, 368 giá trị theo khoảng cách. Tệp còn có mảng thời gian tính bằng giây và mã nhận diện radar. Các tệp này không có trong phần dữ liệu hiện tại nên chưa thể phân tích tín hiệu thô.

### Cách đọc hai bảng CSV chính

Trong streams_composition.csv, mỗi dòng là một **đoạn được lấy từ một bản ghi khác**, sau đó ghép vào luồng:

- stream_id cho biết đoạn thuộc luồng nào, chẳng hạn train_00 hay val_04.
- segment_id là mã của đoạn.
- kind cho biết đoạn thuộc hoạt động thường ngày (adl), cú ngã (fall) hay nhóm khác (other).
- path cho biết đoạn đến từ tệp nguồn nào. Đây là thông tin để lần theo nguồn gốc, không phải tín hiệu được phép đưa cho mô hình.
- label ghi tên hoạt động hoặc loại cú ngã.
- dataset ghi bộ dữ liệu nguồn của cú ngã. Ô này để trống với hoạt động thường ngày là đúng thiết kế.
- t_start và t_end là vị trí của đoạn trong luồng đã ghép. Chúng **không phải** biên chính xác của cú ngã.
- source_t0 và source_t1 là đoạn thời gian được lấy từ tệp nguồn.

Trong streams_events.csv, t_start và t_end là biên cú ngã trong luồng. fall_type có các giá trị Chute01 đến Chute04. quality và needs_review cho biết nhãn có chất lượng thấp hay cần xem lại không. source_path giúp nối sự kiện với đoạn nguồn tương ứng.

Trong tệp nộp bài, mỗi dự đoán gồm stream_id, t_start, t_end và score. Điểm score càng cao thì mô hình càng tin dự đoán đó. t_alarm là cột tùy chọn, ghi thời điểm hệ thống phát báo động đầu tiên. Hai biên phải nằm trong luồng và t_start phải nhỏ hơn t_end.

## 2. EDA: những gì dữ liệu cho thấy

### Quy mô và thời lượng

| Phần dữ liệu | Số luồng | Thời lượng tổng | Số đoạn được ghép | Số cú ngã gắn nhãn |
|---|---:|---:|---:|---:|
| Train | 13 | 6,182 giờ | 1.758 | 359 |
| Validation | 5 | 2,049 giờ | 588 | 119 |
| **Tổng** | **18** | **8,231 giờ** | **2.346** | **478** |

Hầu hết luồng dài khoảng 29,3 đến 29,5 phút. Có hai luồng ngắn hơn nhiều: train_12 dài 18,93 phút, còn val_04 dài 5,19 phút. Vì vậy, khi so sánh báo động nhầm giữa các luồng, nên dùng **số báo động nhầm mỗi giờ**, không chỉ đếm tổng số.

### Những đoạn nào được ghép vào luồng?

| Loại đoạn | Số đoạn | Tỷ lệ theo số đoạn |
|---|---:|---:|
| Hoạt động thường ngày | 1.844 | 78,6% |
| Cú ngã | 478 | 20,4% |
| Khác hoặc phòng trống | 24 | 1,0% |

Tỷ lệ trong bảng tính theo **số đoạn**, không phải theo tổng thời gian. Có 15 hoạt động thường ngày. Trong các luồng, mỗi hoạt động xuất hiện trong 8 tệp nguồn khác nhau: 6 tệp thuộc train và 2 tệp thuộc validation. Tổng cộng có 120 tệp nguồn hoạt động thường ngày được dùng trong bảng thành phần. Một số hoạt động bị cắt thành nhiều đoạn: Faire_pates có 340 dòng, còn Boire và Laver_mains có 48 dòng mỗi loại. Vì vậy, 2.346 dòng không phải 2.346 phép đo độc lập; nhiều dòng đến từ cùng một bản ghi và gần nhau về thời gian.

Các đoạn được nối có phần chồng lên nhau khoảng **0,20 giây** để trộn tín hiệu và làm chỗ nối bớt đột ngột. Một dòng fall trong bảng thành phần có thể chứa cả phần trước và sau cú ngã. Do đó, phải dùng bảng sự kiện để lấy biên cú ngã, không dùng toàn bộ khoảng của đoạn.

### Số lượng và thời lượng cú ngã

| Loại cú ngã hoặc bộ dữ liệu nguồn | Số sự kiện |
|---|---:|
| Chute01 | 120 |
| Chute02 | 120 |
| Chute03 | 119 |
| Chute04 | 119 |
| Bộ dữ liệu 1 / 2 / 3 | 160 / 159 / 159 |

Một nửa số cú ngã kéo dài từ **1,26 đến 2,20 giây**. Thời lượng trung bình là **1,89 giây**, thời lượng ở giữa là **1,74 giây**, ngắn nhất **0,48 giây**, dài nhất **5,00 giây**. Có 44 cú ngã ngắn hơn 1 giây và 48 cú ngã dài từ 3 đến 5 giây. Chute04 dài hơn một chút, với trung vị 2,00 giây; các loại còn lại có trung vị khoảng 1,58 đến 1,68 giây.

Cộng thời lượng của cả 478 khoảng được gắn nhãn lại là khoảng **903,5 giây**, tức **3,05%** tổng thời lượng luồng. Tính ra có khoảng **58,1 cú ngã được gắn nhãn mỗi giờ**. Con số này phản ánh cách tạo dữ liệu bằng cách ghép nhiều đoạn có ngã, không phải tần suất ngã trong đời sống thường ngày.

### Chất lượng nhãn

| Tình trạng nhãn | Số cú ngã | Tỷ lệ |
|---|---:|---:|
| quality=low | 141 | 29,5% |
| needs_review=true | 126 | 26,4% |
| Có cả hai cờ | 42 | 8,8% |
| Có ít nhất một trong hai cờ | **225** | **47,1%** |
| Không có hai cờ trên | **253** | **52,9%** |

Toàn bộ 141 nhãn chất lượng thấp đến từ bộ dữ liệu 3. Có **141/159** cú ngã của bộ này bị gắn cờ chất lượng thấp. Bộ dữ liệu 1 và 2 không có nhãn quality=low, nhưng vẫn có một số nhãn cần xem lại.

Khi chấm giống cách dùng trên test, tham số --ignore-flagged biến các nhãn chất lượng thấp hoặc cần xem lại thành vùng không chấm. Khi bật tham số này, validation còn **67** sự kiện được tính điểm thay vì 119; train còn **186** thay vì 359. Cột ignore đang là false ở cả 478 dòng, nhưng khi bật tham số trên thì quy tắc cờ chất lượng vẫn được áp dụng.

Không nên coi các cú ngã bị gắn cờ là ví dụ “không ngã”. Có thể vẫn dùng chúng như ví dụ có ngã, nhưng giảm ảnh hưởng của chúng khi dạy mô hình xác định biên, hoặc không dùng biên của chúng để tính lỗi. Khi so sánh mô hình, cần dùng đúng quy tắc chấm của challenge.

### Nguy cơ mô hình học nhầm bối cảnh

README nêu một rủi ro quan trọng: chỉ nhìn vào nửa giây ngay trước cú ngã, mô hình có thể phân biệt rất dễ đoạn đó với hoạt động thường ngày. Điểm AUC trong phép thử là **1,000** so với hoạt động thường ngày và **0,839** so với đoạn khác/phòng trống. AUC là thước đo khả năng xếp một ví dụ vào đúng nhóm; 1,0 nghĩa là phân biệt hoàn hảo trong phép thử đó.

Điều này có thể xảy ra vì các cú ngã được mô phỏng trên thảm ở những vị trí cố định, còn hoạt động thường ngày được ghi ở nơi khác trong căn hộ. Khi đó, mô hình có thể học dấu hiệu của vị trí hoặc cách ghi hình thay vì học chuyển động của cú ngã.

Hai cách chỉnh nền đã được thử trong README nhưng không giải quyết hết vấn đề. Lọc nền theo trung bình trượt có trọng số với khoảng thời gian 2 giây và 8 giây vẫn cho AUC **0,923** và **0,960** khi phân biệt phần trước cú ngã với hoạt động thường ngày; phép lọc này vẫn giữ lại cấu trúc có thể khôi phục từ tín hiệu đã lọc. Trừ giá trị trung vị theo từng đoạn cũng để lại AUC **0,964** so với hoạt động thường ngày và **0,935** so với đoạn khác, vì cách này có thể làm dấu hiệu phản xạ khi người nằm trên sàn xuất hiện trở lại ở phần trước cú ngã. Vì vậy, không nên cho rằng chỉ cần trừ nền là đã loại bỏ được bối cảnh.

### Kiểm tra tính nhất quán và khả năng rò rỉ dữ liệu

- Không có ô trống trong các cột của bảng sự kiện. Cả 478 mã sự kiện đều riêng biệt và mọi cú ngã đều có thời điểm kết thúc sau thời điểm bắt đầu.
- Cả 478 sự kiện đều nối được đến một đoạn fall tương ứng trong bảng thành phần; mỗi sự kiện nằm bên trong khoảng của đoạn đó.
- Không có đường dẫn tệp nguồn trùng chính xác giữa train và validation. Mỗi nhãn hoạt động thường ngày có 6 tệp nguồn ở train và 2 ở validation.
- README nói người tham gia được tách riêng bên trong từng bộ dữ liệu, nhưng không có mã người chung để kiểm tra người trong bộ ngã có trùng với người trong bộ hoạt động thường ngày hay không. Vì vậy không thể kết luận chắc chắn rằng hai bộ nguồn tách biệt hoàn toàn theo người.
- Ô dataset trống ở các dòng hoạt động thường ngày là bình thường.
- README cấm dùng bản ghi UWB nguồn để huấn luyện hoặc chạy mô hình. Chỉ được dùng các luồng được phép của challenge; đường dẫn nguồn trong CSV chỉ giúp hiểu dữ liệu được ghép ra sao.

## 3. Kết quả của các cách làm mốc

### Dự đoán mẫu

Tệp dự đoán mẫu có 140 khoảng: 34 trên val_00, 38 trên val_01, 36 trên val_02, 26 trên val_03 và 6 trên val_04. Thời lượng dự đoán ở giữa là **0,99 giây**. Các khoảng đều hợp lệ và nằm trong thời lượng luồng. t_alarm trùng t_start ở cả 140 dòng. Theo quy tắc chấm, khi có t_alarm thì độ trễ là t_alarm trừ thời điểm bắt đầu đã gắn nhãn; nếu không có cột này, bộ chấm dùng t_start. Vì vậy, độ trễ của mẫu phụ thuộc vào việc t_start lệch bao nhiêu so với thời điểm cú ngã được gắn nhãn.

Để ghép dự đoán với nhãn, bộ chấm xếp dự đoán theo score rồi ghép từng dự đoán với tối đa một cú ngã được tính điểm. Một dự đoán được tính là đúng nếu mức chồng thời gian đạt ngưỡng yêu cầu. Sau khi ghép xong, dự đoán chưa ghép có thể được bỏ qua nếu chạm vùng nhãn bị bỏ qua đã nới thêm 2 giây. Dự đoán đã ghép đúng với một cú ngã được tính điểm vẫn là đúng, kể cả khi nó cũng chạm vùng bỏ qua.

| Mức chồng thời gian yêu cầu | Dự đoán đúng | Báo nhầm | Bỏ sót | F1 |
|---:|---:|---:|---:|---:|
| 0,1 | 36 | 98 | 31 | 0,358 |
| 0,3 | 34 | 100 | 33 | 0,338 |
| 0,5 | 25 | 109 | 42 | 0,249 |
| 0,7 | 6 | 128 | 61 | 0,060 |

Điểm xếp hạng là trung bình F1 ở ba ngưỡng 0,1, 0,3 và 0,5; kết quả của mẫu là **0,315**. Ngưỡng 0,3 là ngưỡng chính. Nếu hai đội bằng điểm, bộ chấm ưu tiên đội có sai lệch tuyệt đối trung vị ở thời điểm bắt đầu thấp hơn tại ngưỡng 0,3; nếu vẫn bằng, ưu tiên đội có ít báo động nhầm hơn mỗi giờ. Bộ chấm cũng báo mức chồng thời gian trung bình, sai lệch thời điểm bắt đầu/kết thúc, F1 ở ngưỡng 0,7 và độ trễ. Ở ngưỡng 0,3, sai lệch tuyệt đối trung vị của thời điểm bắt đầu khoảng **0,32 giây**. Điểm giảm mạnh khi yêu cầu khoảng thời gian phải trùng khớp sát hơn. Điều này cho thấy cần cải thiện vị trí hai biên, không chỉ nhận ra có cú ngã.

### Hai cách làm mốc trong README

Cách dựa trên năng lượng tín hiệu ước lượng nền bằng bộ lọc mũ có thời gian phản ứng 3 giây (tín hiệu cũ dần ít ảnh hưởng hơn), rồi đánh dấu những lúc phần tín hiệu còn lại vượt mức phân vị 98 của chính luồng đó. Nó nối các khoảng cách nhau dưới 0,8 giây và bỏ các khoảng ngắn hơn 0,6 giây. Với cách thiết lập đó, F1 là **0,358 / 0,338 / 0,249 / 0,060**, trung bình ba ngưỡng đầu là **0,315**. Tệp dự đoán mẫu chính là kết quả này.

Cách thứ hai chia tín hiệu thành các cửa sổ **200 khung hình (4 giây)**, mỗi lần dịch đi **50 khung hình (1 giây)**, trừ mức trung bình của cửa sổ để làm nổi chuyển động, rồi dùng mô hình hồi quy logistic có bù cho chênh lệch giữa số cú ngã và số hoạt động thường ngày. Ngưỡng điểm được chọn là 0,85. F1 là **0,593 / 0,167 / 0,014 / 0,000**, trung bình ba ngưỡng đầu là **0,258**. Cửa sổ 4 giây dài hơn nhiều cú ngã, vốn thường dài 1,74 giây. Cách này có thể nhận ra cú ngã ở ngưỡng lỏng, nhưng cửa sổ dài khiến thời điểm bắt đầu và kết thúc thiếu chính xác.

README lưu ý các ngưỡng của hai cách làm mốc đã được chọn dựa trên validation. Vì vậy, các điểm này giúp so sánh phương pháp trong challenge, nhưng không phải dự báo điểm trên test.

## 4. Tổ hợp mô hình nên thử

### Một quy trình gồm ba bước

**Bước 1: tìm rộng các khoảng có thể là ngã.**  
Dùng mức năng lượng hoặc mức thay đổi của tín hiệu radar để tìm các khoảng nghi vấn. Có thể giữ lại nhiều khoảng thừa để giảm khả năng bỏ sót. Nếu các khoảng nghi vấn gần nhau, nối chúng theo quy tắc thời gian rõ ràng. Bước này cần đặc biệt chú ý các cú ngã có tín hiệu yếu trong bộ dữ liệu 3.

**Bước 2: xem tín hiệu của ba radar theo thời gian để tìm biên.**  
Giữ tín hiệu của từng radar riêng ở đầu vào. Có thể đưa vào cả tín hiệu gốc, mức thay đổi giữa hai thời điểm liên tiếp, và phần chênh lệch giữa tín hiệu hiện tại với mức nền ngắn hạn hoặc dài hạn. Năng lượng, mức phân tán theo khoảng cách, vùng khoảng cách có phản xạ mạnh và tốc độ thay đổi cũng có thể giúp ích. Không nên chỉ dựa vào một cách trừ nền duy nhất.

Mô hình chính có thể gồm hai phần nhỏ. Phần đầu nhận ra dạng phản xạ theo khoảng cách ở từng radar; phần sau đọc xem các dạng đó biến đổi thế nào theo thời gian. Trong tài liệu chuyên môn, hai phần này thường được gọi là mạng tích chập (CNN) và mạng tích chập theo thời gian (TCN). Có thể kết hợp thông tin từ ba radar bằng cách cho mô hình tự điều chỉnh mức tin cậy của từng radar. Đầu ra nên được tạo liên tục, khoảng 50 lần mỗi giây: khả năng đang có cú ngã, khả năng bắt đầu và khả năng kết thúc. Như vậy mô hình dùng được ngữ cảnh vài giây nhưng vẫn tìm được biên chính xác hơn cửa sổ 4 giây cố định.

**Bước 3: lọc bớt các khoảng báo nhầm.**  
Với từng khoảng nghi vấn, tính các thông tin như thời lượng, mức thay đổi của tín hiệu, số radar cùng ghi nhận chuyển động, độ dốc và đỉnh tín hiệu, cùng điểm của mô hình chính. Một mô hình cây quyết định gọn như Random Forest hoặc HistGradientBoosting có thể dùng các thông tin này để loại bớt hoạt động thường ngày dễ nhầm. Cách này tận dụng mô hình học máy cổ điển mà không cần thêm một mạng lớn, phù hợp với số sự kiện sạch còn ít.

Sau đó, điều chỉnh để điểm tin cậy từ các phần khác nhau có thể so sánh được. Chọn ngưỡng báo, khoảng cách để nối hai đoạn, thời lượng tối thiểu và cách xác định biên dựa trên F1 sự kiện ở các ngưỡng tIoU 0,1, 0,3 và 0,5. Giữ riêng t_alarm để đo thời điểm hệ thống báo động; không dùng nó thay cho biên cú ngã.

### Các cách làm để đối chiếu

- Giữ cách dò dựa trên năng lượng làm mốc để biết mô hình mới có thật sự tốt hơn không. Cũng có thể dùng nó để tìm những khoảng nghi vấn ban đầu.
- Thử mô hình nhỏ dựa trên các đặc trưng tính sẵn, như hồi quy logistic, Random Forest hoặc HistGradientBoosting. Đừng chỉ dùng một cửa sổ cố định 4 giây.
- Có thể so sánh với CNN-LSTM, tức mô hình kết hợp nhận dạng dạng tín hiệu với khả năng đọc chuỗi thời gian. Các nghiên cứu UWB đã dùng cách này và cũng đã so sánh học sâu với các mô hình học máy cổ điển.
- Sau khi có mốc nhỏ, có thể thử mô hình CNN/TCN nhiều radar hoặc U-Net để tìm các khoảng theo cả thời gian và khoảng cách. Nghiên cứu trước còn thử dùng lại mô hình xử lý ảnh đã học trước (transfer learning) và mô hình Vision Transformer. Những hướng đó đáng làm đối chứng, nhưng không bảo đảm tốt hơn cho việc định vị liên tục trên bộ dữ liệu này.
- Chưa nên bắt đầu bằng Transformer hoặc mô hình xử lý ảnh lớn. Chỉ có 186 cú ngã train không bị gắn cờ và 6,18 giờ tín hiệu train. Mô hình lớn dễ học thuộc các ví dụ ít ỏi mà không hoạt động tốt trên dữ liệu mới. Nếu thử dùng mô hình ảnh đã học trước, cần kiểm tra nó có hiểu được dạng tín hiệu radar hay không.

### Cách huấn luyện và đánh giá

- Chia dữ liệu theo người tham gia, tệp nguồn hoặc giữ nguyên từng luồng. Không chia ngẫu nhiên các thời điểm hay các cửa sổ chồng lấn, vì khi đó gần như cùng một đoạn tín hiệu có thể xuất hiện cả trong train lẫn validation.
- Giữ 5 luồng validation làm lần đánh giá cuối. Chỉ có 67 sự kiện được tính điểm trong chế độ giống test, nên không nên thử quá nhiều bộ thông số rồi chọn kết quả tốt nhất trên cùng validation. Nếu cần đánh giá chéo bên trong train, hãy chia theo người hoặc tệp nguồn.
- Vì số thời điểm không ngã nhiều hơn rất nhiều so với số thời điểm ngã, cần bù cho chênh lệch đó khi dạy mô hình. Với biên thời gian có sai số ±0,5–1 giây, có thể xem một vùng nhỏ quanh biên là chưa chắc chắn thay vì bắt mô hình dự đoán đúng từng khung hình. Nên giảm ảnh hưởng của nhãn bị gắn cờ khi dạy mô hình xác định biên.
- Báo cáo F1 ở cả bốn ngưỡng, số báo động nhầm mỗi giờ, sai lệch thời điểm bắt đầu/kết thúc và độ trễ báo động. Độ chính xác từng khung hình hoặc AUC chỉ nên dùng để kiểm tra phụ. Có thể lấy mẫu lặp theo luồng hoặc người tham gia để thấy điểm số dao động nhiều đến đâu.
- Kiểm tra kỹ các đoạn dễ bị nhầm: hoạt động thường ngày có động tác mạnh, đoạn nối giữa hai bản ghi, và các chuyển động giống ngã. Xem mô hình có báo trước thời điểm ngã hay đang dựa vào vị trí cố định trong căn hộ không. Có thể lần lượt bỏ từng nhóm đặc trưng để kiểm tra nhóm nào thật sự giúp ích.

## 5. Kết luận

Các CSV hiện có cho biết khá rõ cách luồng dữ liệu được tạo, cách chia train/validation, số lượng và thời lượng cú ngã, cờ chất lượng nhãn, và kết quả của hai cách làm mốc. Hai khó khăn lớn nhất là cú ngã thường ngắn nên biên dễ lệch, và gần một nửa số nhãn bị gắn cờ nên số sự kiện được chấm ít hơn số sự kiện nhìn thấy trong CSV.

Hướng thử hợp lý là tìm rộng các khoảng nghi vấn trước, dùng mô hình nhỏ đọc tín hiệu ba radar để định vị chi tiết, rồi lọc bớt báo nhầm. Việc đánh giá phải giữ nguyên luồng và tập trung vào F1 sự kiện, độ khớp thời gian, số báo động nhầm mỗi giờ và độ trễ. Khi có các tệp HDF5, bước tiếp theo là xem trực tiếp tín hiệu của từng radar để kiểm tra nhiễu, khoảng cách và chỗ mất dữ liệu.

Các luồng được tạo bằng cách ghép nhiều đoạn và có mật độ cú ngã cao hơn môi trường thực. Vì vậy, điểm challenge không tự nó cho biết hệ thống sẽ báo nhầm bao nhiêu lần trong một căn hộ thật.

## Nguồn

- [ABC2027 UWB-FallTrack — The Challenge](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/the-challenge)
- [ABC2027 UWB-FallTrack — Dataset and Tutorial](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/dataset-and-tutorial)
- Maitre, Bouchard & Gaboury (2021), [Fall Detection With UWB Radars and CNN-LSTM Architecture](https://doi.org/10.1109/JBHI.2020.3027967)
- Imbeault-Nepton et al. (2023), [Fall Detection from UWB Radars: A Comparative Analysis of Deep Learning and Classical Machine Learning Techniques](https://doi.org/10.1145/3582515.3609535)
- Abudalfa & Bouchard (2025), [Transfer Learning using Computer Vision Models for Fall Detection from UWB Radars](https://doi.org/10.1016/j.trpro.2025.03.039)
- Abudalfa & Bouchard (2025), [A Hybrid Vision Transformer and Residual Neural Network Model for Fall Detection Using UWB Radars](https://doi.org/10.1007/s10489-024-06156-9)

