# ABC UWB Falltrack

**Mục đích:** workspace của nhóm cho thử thách UWB-FallTrack tại ABC 2027, tập trung vào phát hiện ngã liên tục từ luồng radar UWB.

**Tình trạng hiện tại:** gói dữ liệu train/validation có trên máy khởi tạo workspace nhưng không được đưa lên GitHub; xem [`data/README.md`](data/README.md). Notebook EDA và báo cáo tương ứng vẫn được giữ ngoài các commit ban đầu để hoàn thiện sau. Protocol chính thức cần được xác nhận theo [`docs/challenge_spec.md`](docs/challenge_spec.md).

## 1. Bối cảnh và bài toán cần giải

UWB-FallTrack đặt ra bài toán **Continuous Fall Detection in UWB Radar Streams**: xây dựng một hệ thống có thể theo dõi luồng dữ liệu radar UWB theo thời gian và nhận ra khi nào một sự kiện ngã xảy ra. Một thông báo tổng quan về ABC 2027 mô tả thử thách theo hướng này và cho biết challenge do Université du Québec à Chicoutimi (UQAC) tổ chức; thông báo đó là điểm xuất phát để nhóm hiểu mục tiêu, còn quy định và tài liệu chính thức của ban tổ chức sẽ quyết định cách làm và cách nộp cuối cùng. [Thông báo tổng quan ABC 2027](https://www.informatika.unimal.ac.id/berita/abc-2027-challenges-peluang-kompetisi-computer-vision-dan-artificial-intelligence-tingkat-internasional) · [Trang ABC Research](https://abc-research.github.io/)

Khác với việc chỉ phân loại một đoạn tín hiệu đã được cắt sẵn thành “ngã” hoặc “không ngã”, chữ **liên tục** gợi ý rằng mô hình cần hoạt động trên một chuỗi dữ liệu kéo dài: tiếp nhận các mẫu theo thứ tự thời gian, nhận biết dấu hiệu của một sự kiện, rồi đưa ra cảnh báo đúng lúc. Trong thực tế, một cú ngã có thể diễn ra nhanh nhưng các hoạt động bình thường như ngồi xuống, nằm xuống, cúi người hoặc bước hụt cũng có thể tạo ra chuyển động đáng chú ý. Vì vậy, hệ thống cần tìm được dấu hiệu của ngã mà không biến những hoạt động thường ngày thành quá nhiều cảnh báo sai.

Radar UWB quan sát chuyển động thông qua tín hiệu vô tuyến phản xạ. Dữ liệu vì thế khác đáng kể so với ảnh hoặc video. README của gói dữ liệu hiện có mô tả tín hiệu CIR từ ba radar và nhãn theo khoảng thời gian, nhưng các chi tiết đó mới là thông tin của bản phát hành cục bộ; nhóm chưa xác nhận chúng là protocol chính thức của challenge. README này không tự suy ra metric, split chính thức hay định dạng submission.

### Cách nhóm sẽ đóng khung bài toán

Ở mức khái niệm, đầu vào là một luồng radar UWB có thứ tự thời gian. Hệ thống sẽ phân tích các đoạn tín hiệu theo thời gian và đưa ra dự đoán liên quan đến sự kiện ngã. Nếu định dạng cuộc thi yêu cầu dự đoán thời điểm bắt đầu/kết thúc hoặc xuất một tệp submission cụ thể, pipeline sẽ chuyển dự đoán nội bộ sang đúng định dạng đó.

Ta cần cân bằng ba mục tiêu:

1. **Không bỏ sót ngã:** bỏ lỡ sự kiện có thể làm mất tác dụng của hệ thống cảnh báo.
2. **Hạn chế báo động giả:** các hoạt động thường ngày không nên liên tục kích hoạt cảnh báo.
3. **Phát hiện đủ sớm và ổn định:** mô hình cần phản hồi kịp thời trên luồng, đồng thời không dao động giữa các nhãn khi tín hiệu nhiễu hoặc nằm gần ranh giới giữa hai hành động.

Đây là cách diễn giải phục vụ lập kế hoạch, chưa phải đặc tả đầu ra đã được xác nhận. **Nhóm cần cập nhật cách đóng khung này sau khi đối chiếu tài liệu cuộc thi.** Bản phát hành cục bộ có README riêng mô tả schema và nhãn; các thông tin đó được ghi nhận là mô tả của gói, chưa thay thế protocol chính thức.

## 2. Câu hỏi cần xác nhận theo protocol chính thức

Trước khi chốt kiến trúc mô hình, cả nhóm cần ghi câu trả lời có dẫn nguồn vào [`docs/challenge_spec.md`](docs/challenge_spec.md):

- Dữ liệu đầu vào là tín hiệu thô, channel impulse response (CIR), range profile, spectrogram hay một dạng đặc trưng đã xử lý sẵn?
- Một bản ghi đại diện cho một đoạn hành động hay là luồng liên tục có nhiều sự kiện? Tần số lấy mẫu, đơn vị và cấu trúc tệp là gì?
- Nhãn cho biết lớp của cả bản ghi, nhãn theo từng thời điểm, hay khoảng thời gian bắt đầu/kết thúc sự kiện?
- Có metadata về người tham gia, phiên thu thập, vị trí cảm biến hoặc môi trường không? Những thông tin này có thể ảnh hưởng đến cách chia train/validation để tránh đánh giá quá lạc quan.
- Bộ chia tập và metric nào được ban tổ chức yêu cầu? Có metric phụ hoặc giới hạn thời gian suy luận nào không?
- Tệp dự đoán cần nộp có những cột, nhãn, quy ước thời gian và định dạng nào?
- Có giới hạn về thư viện, tài nguyên tính toán, kích thước mô hình hoặc thời gian chạy không?

## 3. Workflow đề xuất

Workflow đi từ việc hiểu đúng bài toán tới một pipeline có thể chạy lại và tạo kết quả nộp. Mỗi giai đoạn nên có sản phẩm cụ thể để các thành viên khác kiểm tra, thay vì chỉ trao đổi kết quả miệng.

### Giai đoạn 0 — Chốt đặc tả

Đọc trang challenge, hướng dẫn tải dữ liệu, baseline và quy định nộp bài. Ghi lại cấu trúc dữ liệu, nhãn, split, metric, định dạng dự đoán và deadline chính thức trong `docs/challenge_spec.md`. Nếu tài liệu chưa nói rõ điều gì, đánh dấu là câu hỏi mở và hỏi ban tổ chức qua kênh phù hợp; không tự biến suy đoán thành luật của cuộc thi.

**Sản phẩm:** một bản đặc tả ngắn đã được cả nhóm đồng thuận và danh sách điểm còn chưa rõ.

### Giai đoạn 1 — Kiểm kê và hiểu dữ liệu

Với mỗi bản phát hành dữ liệu được phép sử dụng, lập danh mục tệp và kích thước; đọc schema; kiểm tra nhãn thiếu, bản ghi hỏng, thứ tự thời gian, phân bố lớp và độ dài bản ghi. Vẽ một số tín hiệu đại diện cho ngã và hoạt động thường ngày. Nếu có mã người tham gia hoặc phiên thu thập, dùng thông tin đó để hiểu nguy cơ rò rỉ giữa các tập. Giữ dữ liệu gốc nguyên trạng trong `data/raw/`; mọi bước làm sạch và chuyển đổi phải có thể tái tạo.

**Sản phẩm:** notebook khám phá có chú thích, báo cáo chất lượng dữ liệu và đề xuất tiền xử lý.

### Giai đoạn 2 — Dựng baseline từ đầu đến cuối

Tạo một pipeline đơn giản chạy được từ dữ liệu đầu vào đến dự đoán theo định dạng cần đánh giá. Bắt đầu với baseline mà nhóm có thể giải thích và kiểm tra: ví dụ đặc trưng thống kê/tín hiệu kết hợp với mô hình học máy cổ điển, hoặc baseline do ban tổ chức phát hành. Chỉ chọn cách biểu diễn sau khi đã xem định dạng dữ liệu.

Baseline có vai trò làm mốc: nó cho biết pipeline đọc đúng dữ liệu, metric được tính đúng và mô hình phức tạp hơn có thực sự cải thiện hay không.

**Sản phẩm:** lệnh chạy baseline, cấu hình thí nghiệm, dự đoán validation và kết quả metric được lưu lại.

### Giai đoạn 3 — Cải thiện mô hình có kiểm soát

Thử từng thay đổi có giả thuyết rõ ràng: cách chia cửa sổ, chuẩn hóa, đặc trưng thời gian/tần số, cân bằng lớp, kiến trúc mô hình hoặc cách hợp nhất dự đoán liên tiếp thành sự kiện. Mỗi lần chạy lưu phiên bản dữ liệu/split, seed, cấu hình và kết quả. Không dùng tập test để lựa chọn mô hình hay ngưỡng dự đoán.

**Sản phẩm:** bảng thí nghiệm để so sánh baseline và các biến thể, kèm giải thích thay đổi nào có ích và trong điều kiện nào.

### Giai đoạn 4 — Đánh giá và phân tích lỗi

Dùng metric chính thức làm tiêu chí quyết định. Nếu phù hợp với hướng dẫn, theo dõi thêm precision, recall, F1, số báo động giả theo thời gian và độ trễ phát hiện. Kiểm tra các trường hợp ngã bị bỏ sót và hoạt động thường ngày bị nhận nhầm; phân tích theo người, phiên hoặc kịch bản chỉ khi metadata và protocol cho phép.

**Sản phẩm:** báo cáo validation, biểu đồ lỗi, kết quả ablation cần thiết và quyết định mô hình cuối cùng.

### Giai đoạn 5 — Đóng gói suy luận và nộp bài

Đóng băng pipeline đã chọn, chạy lại từ môi trường sạch, kiểm tra định dạng đầu ra bằng validator chính thức nếu có, rồi tạo đúng tệp cần nộp. Ghi rõ commit, cấu hình, trọng số và lệnh sinh kết quả để thành viên khác có thể tái tạo. Không sửa mô hình sau khi xem nhãn test hoặc kết quả ẩn.

**Sản phẩm:** mã suy luận, tệp kết quả đúng định dạng, hướng dẫn tái lập và ghi chú phiên bản được dùng.

### Giai đoạn 6 — Viết báo cáo/paper và hoàn tất bàn giao

Mô tả câu hỏi nghiên cứu, dữ liệu và protocol, baseline, lựa chọn phương pháp, kết quả, phân tích lỗi và giới hạn. Mọi bảng/biểu đồ trong báo cáo phải truy ngược được về một thí nghiệm đã lưu. Đối chiếu yêu cầu paper và submission với hướng dẫn chính thức thay vì dựa vào cấu trúc dự kiến trong README.

**Sản phẩm:** bản thảo và phụ lục đủ thông tin để nhóm giải thích cũng như tái lập kết quả.

Phần mô tả công việc cụ thể của từng thành viên và cách phối hợp được lưu riêng trong [GUIDELINES.md](GUIDELINES.md).

## 4. Quy ước làm việc

1. **Một thí nghiệm phải tái lập được.** Ghi split, phiên bản tiền xử lý, cấu hình, seed, metric và đường dẫn artifact.
2. **Dữ liệu gốc không bị ghi đè.** Lưu dữ liệu tải về trong `data/raw/`; dữ liệu chuyển đổi có thể tái tạo đặt tại `data/interim/` hoặc `data/processed/`.
3. **Tách khám phá khỏi pipeline.** Notebook dùng để xem xét và thử giả thuyết; logic được dùng lại nên chuyển thành module trong `src/` hoặc script trong `scripts/`.
4. **Tôn trọng split chính thức.** Chỉ dùng train/validation theo protocol để phát triển; không điều chỉnh mô hình theo nhãn test.
5. **Ghi lại kết luận, không chỉ số điểm.** Với mỗi thay đổi quan trọng, giải thích vì sao thử, metric nào đổi và nhóm lỗi nào tốt lên hoặc xấu đi.
6. **Tích hợp thường xuyên.** Mỗi phần việc nên có phạm vi nhỏ, cấu hình và hướng dẫn chạy; tránh để code chỉ chạy được trên máy của một thành viên.

## 5. Cấu trúc thư mục

```text
ABC/
├── README.md                    # Bài toán, workflow và hướng dẫn khởi động
├── GUIDELINES.md                # Vai trò và công việc cụ thể của 5 thành viên
├── .editorconfig                # Quy ước định dạng tệp
├── .gitignore                   # Bỏ qua môi trường, dữ liệu cục bộ và artifact sinh ra
├── AGENTS.md                    # Quy ước cho các thay đổi trong repository
├── pyproject.toml               # Metadata và nhóm phụ thuộc Python cơ bản
├── configs/                     # Cấu hình dữ liệu, train và đánh giá
├── data/
│   ├── README.md                 # Cách quản lý dữ liệu cục bộ; không commit dữ liệu radar
│   ├── raw/                      # Dữ liệu tải về nguyên trạng, chỉ lưu cục bộ
│   ├── interim/                  # Dữ liệu qua bước xử lý trung gian, chỉ lưu cục bộ
│   ├── processed/                # Dữ liệu sẵn sàng cho pipeline, chỉ lưu cục bộ
│   └── external/                 # Dữ liệu tham chiếu được phép, chỉ lưu cục bộ
├── docs/
│   ├── challenge_spec.md         # Protocol chính thức và các điểm còn cần xác nhận
│   └── project_guide.md          # Quy ước chung cho workspace
├── notebooks/                   # Khám phá dữ liệu và thử nghiệm ban đầu
├── outputs/
│   ├── figures/                  # Biểu đồ sinh ra trong quá trình phân tích
│   ├── models/                   # Trọng số và checkpoint
│   └── submissions/              # Tệp dự đoán xuất ra để nộp
├── reports/
│   ├── experiments/              # Báo cáo thí nghiệm và phân tích lỗi
│   └── paper/                    # Bản thảo, bảng và hình cho paper
├── scripts/                     # Lệnh tiện ích chạy pipeline
├── src/
│   └── abc_uwb_falltrack/        # Mã nguồn Python tái sử dụng
└── tests/                       # Kiểm tra tự động cho các module khi được xây dựng
```

Git chỉ theo dõi `data/README.md`; dữ liệu phát hành, nhãn và dữ liệu trung gian bị bỏ qua trong `.gitignore`. Gói hiện có trên máy khởi tạo workspace khoảng 19 GB, vượt xa quy mô nên lưu trong Git thông thường. Mỗi thành viên cần dùng nguồn dữ liệu được cấp phép; kiểm tra giấy phép và quy định challenge trước khi chia sẻ dữ liệu qua bất kỳ dịch vụ nào.

## 6. Bắt đầu môi trường Python

Yêu cầu ban đầu là Python 3.10 trở lên. Có thể tạo môi trường ảo và cài nhóm thư viện phân tích dữ liệu như sau:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis]"
```

Trên Windows PowerShell, bật môi trường bằng `.\.venv\Scripts\Activate.ps1`. Khi biết định dạng dữ liệu và phương pháp đã chọn, nhóm sẽ bổ sung thư viện chuyên biệt cần thiết vào `pyproject.toml` để các thành viên dùng cùng cấu hình.

## 7. Điểm cần cập nhật trước khi làm thí nghiệm chính thức

- Xác minh các chi tiết còn mở trong [`docs/challenge_spec.md`](docs/challenge_spec.md) từ tài liệu challenge chính thức.
- Cập nhật phiên bản và cách lấy dữ liệu theo quyền truy cập được cấp; không đưa gói dữ liệu vào GitHub theo mặc định.
- Chốt ngày nội bộ dựa trên deadline challenge hiện hành.
- Bổ sung kết quả baseline và hướng dẫn chạy sau khi pipeline đầu tiên hoạt động.
