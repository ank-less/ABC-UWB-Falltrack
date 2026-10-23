import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple

# Cấu hình hằng số theo đặc tả bài toán ABC2027 UWB-FallTrack
FPS = 50  # 50 khung hình/giây (cách nhau 0.02s)
NUM_RADARS = 3
NUM_RANGE_BINS = 368
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 0. HÀM TẠO DỮ LIỆU GIẢ LẬP (DUMMY DATA GENERATOR)
# =====================================================================
def generate_dummy_stream(duration_sec: int = 120) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Tạo luồng dữ liệu CIR giả lập có kích thước (3, T, 368) kiểu float32.
    Chèn sẵn một số đoạn có biên độ cao để mô phỏng cú ngã và hoạt động mạnh.
    """
    np.random.seed(42)
    total_frames = duration_sec * FPS
    
    # Nhiễu nền bình thường (Gaussian noise)
    cir_data = np.random.normal(0, 0.1, size=(NUM_RADARS, total_frames, NUM_RANGE_BINS)).astype(np.float32)
    
    # Chèn 4 sự kiện giả lập (bắt đầu_giây, kết_thúc_giây, cường_độ)
    # 3 sự kiện ngã thật (dài ~1.5 - 2.2s) và 1 sự kiện nhiễu ngắn
    simulated_events = [
        (15.0, 16.8, 1.8),  # Cú ngã 1
        (45.2, 47.0, 2.1),  # Cú ngã 2
        (78.0, 78.4, 1.5),  # Nhiễu ngắn (0.4s)
        (95.5, 97.6, 2.0),  # Cú ngã 3
    ]
    
    gt_intervals = []
    for start_s, end_s, intensity in simulated_events:
        s_frame = int(start_s * FPS)
        e_frame = int(end_s * FPS)
        # Tạo tín hiệu phản xạ mạnh ở vùng khoảng cách giữa (bins 100 -> 250)
        cir_data[:, s_frame:e_frame, 100:250] += np.random.normal(
            0, intensity, size=(NUM_RADARS, e_frame - s_frame, 150)
        ).astype(np.float32)
        if (end_s - start_s) >= 1.0:
            gt_intervals.append((s_frame, e_frame))
            
    return cir_data, gt_intervals


# =====================================================================
# 1. BƯỚC 1: BỘ LỌC NĂNG LƯỢNG (ENERGY-BASED HEURISTIC)
# =====================================================================
def find_candidate_windows(
    cir_array: np.ndarray,
    window_sec: float = 1.0,
    percentile_thresh: float = 95.0,
    max_gap_sec: float = 0.8,
    min_duration_sec: float = 0.5
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Tìm rộng các đoạn nghi vấn dựa trên năng lượng tín hiệu phản xạ.
    Trả về danh sách các tuple (start_frame, end_frame) và mảng năng lượng.
    """
    # 1. Tính năng lượng tổng hợp theo chiều khoảng cách (axis=2) -> shape: (3, T)
    energy_per_radar = np.sum(np.square(cir_array), axis=2)
    
    # Tổng hợp năng lượng từ 3 radar -> shape: (T,)
    total_energy = np.sum(energy_per_radar, axis=0)
    
    # 2. Bộ lọc trung bình trượt (Moving Average) với cửa sổ 1 giây (50 frames)
    win_len = max(1, int(window_sec * FPS))
    kernel = np.ones(win_len) / win_len
    smoothed_energy = np.convolve(total_energy, kernel, mode="same")
    
    # 3. Tìm các khung hình vượt phân vị thứ 95
    threshold = np.percentile(smoothed_energy, percentile_thresh)
    active_mask = smoothed_energy > threshold
    
    # Xác định điểm bắt đầu và kết thúc của các đoạn vượt ngưỡng
    padded_mask = np.pad(active_mask.astype(int), (1, 1), mode="constant")
    diff = np.diff(padded_mask)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return [], smoothed_energy
        
    # 4. Gộp các đoạn cách nhau dưới 0.8 giây (40 frames)
    max_gap_frames = int(max_gap_sec * FPS)
    min_dur_frames = int(min_duration_sec * FPS)
    
    merged_windows = []
    curr_start, curr_end = starts[0], ends[0]
    
    for next_start, next_end in zip(starts[1:], ends[1:]):
        if next_start - curr_end <= max_gap_frames:
            curr_end = next_end  # Gộp đoạn
        else:
            if (curr_end - curr_start) >= min_dur_frames:
                merged_windows.append((int(curr_start), int(curr_end)))
            curr_start, curr_end = next_start, next_end
            
    if (curr_end - curr_start) >= min_dur_frames:
        merged_windows.append((int(curr_start), int(curr_end)))
        
    return merged_windows, smoothed_energy


# =====================================================================
# 2. BƯỚC 2: MẠNG HỌC SÂU CNN + LSTM (ĐỊNH VỊ BIÊN CHUỖI THỜI GIAN)
# =====================================================================
class FallDetectorNet(nn.Module):
    """
    Kiến trúc CNN + BiLSTM đọc tín hiệu 3 radar:
    - Input: (Batch, 3, seq_len, 368)
    - Output: (Batch, seq_len) chứa xác suất ngã cho từng khung hình.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # CNN 1D quét qua 368 range bins (xem 3 radar như 3 kênh đầu vào)
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(in_channels=NUM_RADARS, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # Đưa chiều khoảng cách về cố định 8 bins -> 32 * 8 = 256 features
        )
        
        # LSTM hai chiều đọc diễn biến theo thời gian (seq_len)
        self.temporal_lstm = nn.LSTM(
            input_size=32 * 8,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Lớp phân loại cho từng khung hình
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, 3, seq_len, 368)
        batch_size, radars, seq_len, bins = x.shape
        
        # Gộp Batch và seq_len để đưa qua Conv1d: (Batch * seq_len, 3, 368)
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch_size * seq_len, radars, bins)
        
        # Trích xuất đặc trưng khoảng cách: (Batch * seq_len, 32, 8)
        cnn_feat = self.spatial_cnn(x_reshaped)
        cnn_feat = cnn_feat.view(batch_size, seq_len, -1)  # (Batch, seq_len, 256)
        
        # Đưa qua LSTM: (Batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.temporal_lstm(cnn_feat)
        
        # Dự đoán xác suất cho từng frame: (Batch, seq_len)
        probs = self.classifier(lstm_out).squeeze(-1)
        return probs


def train_cnn_lstm_dummy(model: nn.Module, epochs: int = 10) -> None:
    """
    Hàm huấn luyện minh họa cho mạng CNN+LSTM sử dụng Binary Cross Entropy (BCE).
    Tạo một tập batch nhỏ gồm các cửa sổ 100 frames (2 giây).
    """
    model.to(DEVICE)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # Tạo 16 mẫu giả lập, mỗi mẫu dài 100 khung hình (2 giây)
    seq_len = 100
    dummy_x = torch.randn(16, NUM_RADARS, seq_len, NUM_RANGE_BINS, device=DEVICE)
    dummy_y = torch.zeros(16, seq_len, device=DEVICE)
    
    # Giả sử nửa đầu batch có cú ngã ở khung hình 25 -> 75
    dummy_x[:8, :, 25:75, 100:250] += 1.5
    dummy_y[:8, 25:75] = 1.0
    
    print("--- Huấn luyện thử nghiệm mạng CNN + LSTM (Bước 2) ---")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        preds = model(dummy_x)
        loss = criterion(preds, dummy_y)
        loss.backward()
        optimizer.step()
        
        if epoch in (1, 5, epochs):
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - BCE Loss: {loss.item():.4f}")


# =====================================================================
# 3. BƯỚC 3: RANDOM FOREST (TRÍCH XUẤT ĐẶC TRƯNG & LỌC NHIỄU)
# =====================================================================
def extract_features(
    cir_segment: np.ndarray,
    frame_probs: np.ndarray
) -> np.ndarray:
    """
    Trích xuất vector đặc trưng từ một đoạn nghi vấn và xác suất của CNN+LSTM:
    1. Thời lượng đoạn (giây)
    2. Năng lượng tối đa
    3. Năng lượng trung bình
    4. Xác suất cao nhất từ CNN+LSTM
    5. Xác suất trung bình từ CNN+LSTM
    """
    seq_len = cir_segment.shape[1]
    duration_sec = seq_len / float(FPS)
    
    # Năng lượng tại mỗi khung hình của đoạn
    segment_energy = np.sum(np.square(cir_segment), axis=(0, 2))
    max_energy = float(np.max(segment_energy))
    mean_energy = float(np.mean(segment_energy))
    
    max_prob = float(np.max(frame_probs))
    mean_prob = float(np.mean(frame_probs))
    
    return np.array([duration_sec, max_energy, mean_energy, max_prob, mean_prob], dtype=np.float32)


def train_random_forest_dummy() -> RandomForestClassifier:
    """
    Khởi tạo và huấn luyện mô hình Random Forest trên tập đặc trưng giả lập.
    """
    np.random.seed(42)
    # Tạo 40 mẫu đặc trưng giả lập: [duration, max_e, mean_e, max_p, mean_p]
    # Nhãn 0: Sinh hoạt thường ngày (ADL), Nhãn 1: Cú ngã (Fall)
    adl_features = np.column_stack([
        np.random.uniform(0.5, 4.0, 20),     # Thời lượng
        np.random.uniform(10.0, 80.0, 20),   # Max energy
        np.random.uniform(5.0, 40.0, 20),    # Mean energy
        np.random.uniform(0.1, 0.55, 20),    # Max prob thấp
        np.random.uniform(0.05, 0.35, 20),   # Mean prob thấp
    ])
    
    fall_features = np.column_stack([
        np.random.uniform(1.2, 2.5, 20),     # Thời lượng chuẩn của cú ngã (1.2 - 2.5s)
        np.random.uniform(150.0, 600.0, 20), # Max energy cao
        np.random.uniform(80.0, 300.0, 20),  # Mean energy cao
        np.random.uniform(0.65, 0.99, 20),   # Max prob cao
        np.random.uniform(0.45, 0.85, 20),   # Mean prob cao
    ])
    
    x_train = np.vstack([adl_features, fall_features])
    y_train = np.array([0] * 20 + [1] * 20)
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(x_train, y_train)
    print("\n--- Đã huấn luyện xong Random Forest (Bước 3) ---")
    return rf_model


# =====================================================================
# 4. TỔNG HỢP (INFERENCE PIPELINE)
# =====================================================================
def refine_boundaries(
    start_frame: int,
    end_frame: int,
    frame_probs: np.ndarray,
    prob_thresh: float = 0.5
) -> Tuple[int, int]:
    """
    Tinh chỉnh lại biên bắt đầu và kết thúc bên trong đoạn nghi vấn dựa vào
    xác suất từng khung hình của mạng CNN+LSTM.
    """
    active_idx = np.where(frame_probs >= prob_thresh)[0]
    if len(active_idx) >= int(0.4 * FPS):  # Ít nhất 0.4 giây
        refined_start = start_frame + int(active_idx[0])
        refined_end = start_frame + int(active_idx[-1])
        return refined_start, refined_end
    return start_frame, end_frame


def predict_stream(
    cir_array: np.ndarray,
    stream_id: str,
    dl_model: nn.Module,
    rf_model: RandomForestClassifier,
    rf_score_thresh: float = 0.4
) -> pd.DataFrame:
    """
    Chạy tuần tự 3 bước trên một luồng dữ liệu CIR (3, T, 368):
    Bước 1 (Lọc năng lượng) -> Cắt tín hiệu -> Bước 2 (CNN+LSTM) -> Bước 3 (Random Forest).
    Trả về DataFrame theo định dạng nộp bài.
    """
    dl_model.eval()
    predictions = []
    
    # Bước 1: Tìm các khoảng nghi vấn bằng năng lượng
    candidate_windows, _ = find_candidate_windows(cir_array)
    print(f"\n[Stream: {stream_id}] Bước 1 tìm thấy {len(candidate_windows)} đoạn nghi vấn.")
    
    for s_frame, e_frame in candidate_windows:
        # Cắt đoạn tín hiệu nghi vấn: shape (3, seq_len, 368)
        segment = cir_array[:, s_frame:e_frame, :]
        
        # Bước 2: Đưa qua mạng CNN+LSTM để lấy xác suất từng khung hình
        seg_tensor = torch.from_numpy(segment).unsqueeze(0).to(DEVICE)  # (1, 3, seq_len, 368)
        with torch.no_grad():
            frame_probs = dl_model(seg_tensor).squeeze(0).cpu().numpy()
            
        # Bước 3: Trích xuất đặc trưng và phân loại bằng Random Forest
        features = extract_features(segment, frame_probs).reshape(1, -1)
        fall_prob_rf = float(rf_model.predict_proba(features)[0, 1])
        
        # Kết hợp điểm tin cậy từ Random Forest và CNN+LSTM
        combined_score = 0.6 * fall_prob_rf + 0.4 * float(np.max(frame_probs))
        
        if combined_score >= rf_score_thresh:
            # Tinh chỉnh biên thời gian
            ref_start, ref_end = refine_boundaries(s_frame, e_frame, frame_probs)
            t_start = round(ref_start / float(FPS), 2)
            t_end = round(ref_end / float(FPS), 2)
            
            # Đảm bảo t_start < t_end hợp lệ theo quy tắc chấm
            if t_end > t_start:
                predictions.append({
                    "stream_id": stream_id,
                    "t_start": t_start,
                    "t_end": t_end,
                    "score": round(combined_score, 4)
                })
                
    df_sub = pd.DataFrame(predictions, columns=["stream_id", "t_start", "t_end", "score"])
    # Sắp xếp theo điểm tin cậy giảm dần giống quy tắc ghép cặp của bộ chấm
    if not df_sub.empty:
        df_sub = df_sub.sort_values(by="score", ascending=False).reset_index(drop=True)
    return df_sub


# =====================================================================
# CHẠY THỬ NGHIỆM TOÀN BỘ PIPELINE TRÊN GOOGLE COLAB
# =====================================================================
if __name__ == "__main__":
    print(f"Đang chạy trên thiết bị: {DEVICE}")
    
    # 1. Tạo dữ liệu luồng radar giả lập dài 120 giây (6000 frames)
    dummy_cir, gt_events = generate_dummy_stream(duration_sec=120)
    print(f"Kích thước mảng CIR giả lập: {dummy_cir.shape} (3 radar, T khung hình, 368 bins)")
    print(f"Các cú ngã thực tế đã chèn (giây): {[(s/FPS, e/FPS) for s, e in gt_events]}\n")
    
    # 2. Khởi tạo và huấn luyện mô hình Bước 2 (CNN + LSTM)
    cnn_lstm_net = FallDetectorNet(hidden_dim=64)
    train_cnn_lstm_dummy(cnn_lstm_net, epochs=10)
    
    # 3. Khởi tạo và huấn luyện mô hình Bước 3 (Random Forest)
    rf_classifier = train_random_forest_dummy()
    
    # 4. Chạy dự đoán trên luồng dữ liệu val_00
    submission_df = predict_stream(
        cir_array=dummy_cir,
        stream_id="val_00",
        dl_model=cnn_lstm_net,
        rf_model=rf_classifier
    )
    
    print("\n--- KẾT QUẢ ĐẦU RA (ĐỊNH DẠNG SUBMISSION) ---")
    print(submission_df.to_string(index=False))
    
    # Xuất ra file CSV
    submission_df.to_csv("sample_submission_output.csv", index=False)
    print("\nĐã lưu kết quả vào file 'sample_submission_output.csv'.")
