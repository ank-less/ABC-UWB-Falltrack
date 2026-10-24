import os
import glob
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple

# =====================================================================
# CẤU HÌNH HẰNG SỐ & ĐƯỜNG DẪN
# =====================================================================
FPS = 50
NUM_RADARS = 3
NUM_RANGE_BINS = 368
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# THAY ĐỔI ĐƯỜNG DẪN NÀY THEO THƯ MỤC TRÊN COLAB CỦA BẠN
DATA_DIR = "/content/drive/MyDrive/UWB_FallTrack/data"
EVENTS_CSV = os.path.join(DATA_DIR, "streams_events.csv")
OUTPUT_CSV = "final_predictions.csv" # Hoàn toàn không dùng tên file sample

# =====================================================================
# 1. HÀM ĐỌC DỮ LIỆU THẬT & GÁN NHÃN
# =====================================================================
def load_cir_from_h5(file_path: str) -> np.ndarray:
    """Đọc mảng CIR có kích thước (3, T, 368) kiểu float32 từ tệp HDF5[cite: 1]."""
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        cir_key = "cir" if "cir" in keys else keys[0]
        cir_array = np.array(f[cir_key], dtype=np.float32)
    return cir_array

def check_overlap(start_f: int, end_f: int, gt_events: pd.DataFrame) -> int:
    """Kiểm tra xem đoạn nghi vấn có chồng lấn với cú ngã thực tế nào không."""
    if gt_events.empty:
        return 0
    start_s, end_s = start_f / FPS, end_f / FPS
    for _, row in gt_events.iterrows():
        # Lấy biên sự kiện từ file CSV[cite: 1]
        gt_start, gt_end = row['t_start'], row['t_end']
        # Nếu có giao cắt thời gian
        if max(start_s, gt_start) < min(end_s, gt_end):
            return 1
    return 0

# =====================================================================
# 2. BƯỚC 1: BỘ LỌC NĂNG LƯỢNG (TÌM KIẾM RỘNG)
# =====================================================================
def find_candidate_windows(
    cir_array: np.ndarray,
    window_sec: float = 1.0,
    percentile_thresh: float = 95.0,
    max_gap_sec: float = 0.8,
    min_duration_sec: float = 0.5
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    energy_per_radar = np.sum(np.square(cir_array), axis=2)
    total_energy = np.sum(energy_per_radar, axis=0)
    
    win_len = max(1, int(window_sec * FPS))
    kernel = np.ones(win_len) / win_len
    smoothed_energy = np.convolve(total_energy, kernel, mode="same")
    
    threshold = np.percentile(smoothed_energy, percentile_thresh)
    active_mask = smoothed_energy > threshold
    
    padded_mask = np.pad(active_mask.astype(int), (1, 1), mode="constant")
    diff = np.diff(padded_mask)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return [], smoothed_energy
        
    max_gap_frames = int(max_gap_sec * FPS)
    min_dur_frames = int(min_duration_sec * FPS)
    
    merged_windows = []
    curr_start, curr_end = starts[0], ends[0]
    
    for next_start, next_end in zip(starts[1:], ends[1:]):
        if next_start - curr_end <= max_gap_frames:
            curr_end = next_end
        else:
            if (curr_end - curr_start) >= min_dur_frames:
                merged_windows.append((int(curr_start), int(curr_end)))
            curr_start, curr_end = next_start, next_end
            
    if (curr_end - curr_start) >= min_dur_frames:
        merged_windows.append((int(curr_start), int(curr_end)))
        
    return merged_windows, smoothed_energy

# =====================================================================
# 3. BƯỚC 2: MẠNG CNN + LSTM
# =====================================================================
class FallDetectorNet(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(in_channels=NUM_RADARS, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.temporal_lstm = nn.LSTM(
            input_size=32 * 8,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, radars, seq_len, bins = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch_size * seq_len, radars, bins)
        cnn_feat = self.spatial_cnn(x_reshaped)
        cnn_feat = cnn_feat.view(batch_size, seq_len, -1)
        lstm_out, _ = self.temporal_lstm(cnn_feat)
        probs = self.classifier(lstm_out).squeeze(-1)
        return probs

def train_cnn_lstm(model: nn.Module, train_loader: DataLoader, epochs: int = 15):
    model.to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    print("\n--- Bắt đầu huấn luyện CNN + LSTM ---")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_x)
            
            # Tính loss trung bình trên toàn bộ chiều dài chuỗi (seq_len)
            # batch_y được expand để khớp với shape của preds
            target = batch_y.unsqueeze(1).expand_as(preds)
            loss = criterion(preds, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {total_loss/len(train_loader):.4f}")

# =====================================================================
# 4. BƯỚC 3: RANDOM FOREST LỌC NHIỄU
# =====================================================================
def extract_features(cir_segment: np.ndarray, frame_probs: np.ndarray) -> np.ndarray:
    seq_len = cir_segment.shape[1]
    duration_sec = seq_len / float(FPS)
    segment_energy = np.sum(np.square(cir_segment), axis=(0, 2))
    return np.array([
        duration_sec,
        float(np.max(segment_energy)),
        float(np.mean(segment_energy)),
        float(np.max(frame_probs)),
        float(np.mean(frame_probs))
    ], dtype=np.float32)

# =====================================================================
# 5. XÂY DỰNG TẬP DỮ LIỆU & CHẠY PIPELINE
# =====================================================================
def prepare_training_data(events_df: pd.DataFrame) -> Tuple[TensorDataset, np.ndarray, np.ndarray]:
    """Quét qua các tệp train_*.h5 để cắt đoạn nghi vấn và gán nhãn."""
    train_files = glob.glob(os.path.join(DATA_DIR, "train_*.h5"))
    if not train_files:
        raise FileNotFoundError(f"Không tìm thấy file train_*.h5 trong {DATA_DIR}")

    X_list, y_list = [], []
    rf_features_list, rf_labels = [], []
    
    print("Đang quét dữ liệu huấn luyện (có thể mất thời gian)...")
    for file_path in sorted(train_files):
        stream_id = os.path.basename(file_path).replace(".h5", "")
        stream_events = events_df[events_df["stream_id"] == stream_id]
        
        cir_array = load_cir_from_h5(file_path)
        windows, _ = find_candidate_windows(cir_array)
        
        for s_frame, e_frame in windows:
            label = check_overlap(s_frame, e_frame, stream_events)
            segment = cir_array[:, s_frame:e_frame, :]
            
            # Padding/Truncate để tất cả tensor có cùng độ dài (ví dụ: 150 frames = 3s)
            fixed_len = 150
            if segment.shape[1] < fixed_len:
                pad_width = fixed_len - segment.shape[1]
                segment_padded = np.pad(segment, ((0,0), (0, pad_width), (0,0)), mode='constant')
            else:
                segment_padded = segment[:, :fixed_len, :]
                
            X_list.append(segment_padded)
            y_list.append(label)
            
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    return dataset, X_tensor.numpy(), y_tensor.numpy()

def predict_stream_real(
    cir_array: np.ndarray,
    stream_id: str,
    dl_model: nn.Module,
    rf_model: RandomForestClassifier,
) -> List[dict]:
    dl_model.eval()
    predictions = []
    candidate_windows, _ = find_candidate_windows(cir_array)
    
    for s_frame, e_frame in candidate_windows:
        segment = cir_array[:, s_frame:e_frame, :]
        seg_tensor = torch.from_numpy(segment).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            frame_probs = dl_model(seg_tensor).squeeze(0).cpu().numpy()
            
        features = extract_features(segment, frame_probs).reshape(1, -1)
        fall_prob_rf = float(rf_model.predict_proba(features)[0, 1])
        combined_score = 0.6 * fall_prob_rf + 0.4 * float(np.max(frame_probs))
        
        if combined_score >= 0.4:
            active_idx = np.where(frame_probs >= 0.5)[0]
            if len(active_idx) > 0:
                ref_start = s_frame + int(active_idx[0])
                ref_end = s_frame + int(active_idx[-1])
            else:
                ref_start, ref_end = s_frame, e_frame
                
            t_start = round(ref_start / FPS, 2)
            t_end = round(ref_end / FPS, 2)
            
            if t_end > t_start:
                predictions.append({
                    "stream_id": stream_id,
                    "t_start": t_start,
                    "t_end": t_end,
                    "score": round(combined_score, 4)
                })
    return predictions

if __name__ == "__main__":
    if not os.path.exists(EVENTS_CSV):
        print(f"LỖI: Vui lòng đặt file streams_events.csv vào thư mục {DATA_DIR}")
    else:
        # 1. Đọc bảng sự kiện để lấy nhãn biên chính xác[cite: 1]
        events_df = pd.read_csv(EVENTS_CSV)
        
        # 2. Chuẩn bị dữ liệu huấn luyện từ các file train_*.h5
        train_dataset, X_numpy, y_numpy = prepare_training_data(events_df)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # 3. Khởi tạo & Huấn luyện CNN+LSTM
        cnn_lstm_model = FallDetectorNet(hidden_dim=64)
        train_cnn_lstm(cnn_lstm_model, train_loader, epochs=10)
        
        # 4. Huấn luyện Random Forest
        print("\n--- Đang trích xuất đặc trưng cho Random Forest ---")
        cnn_lstm_model.eval()
        rf_features = []
        with torch.no_grad():
            for i in range(len(X_numpy)):
                seg_tensor = torch.from_numpy(X_numpy[i]).unsqueeze(0).to(DEVICE)
                probs = cnn_lstm_model(seg_tensor).squeeze(0).cpu().numpy()
                rf_features.append(extract_features(X_numpy[i], probs))
                
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_model.fit(np.array(rf_features), y_numpy)
        print("Huấn luyện Random Forest thành công.")
        
        # 5. Chạy suy luận (Inference) trên tập Validation
        val_files = glob.glob(os.path.join(DATA_DIR, "val_*.h5"))
        all_predictions = []
        
        print("\n--- Đang dự đoán trên tập Validation ---")
        for val_file in sorted(val_files):
            stream_id = os.path.basename(val_file).replace(".h5", "")
            cir_val = load_cir_from_h5(val_file)
            preds = predict_stream_real(cir_val, stream_id, cnn_lstm_model, rf_model)
            all_predictions.extend(preds)
            print(f"Đã xử lý xong {stream_id}")
            
        # 6. Xuất kết quả
        final_df = pd.DataFrame(all_predictions)
        if not final_df.empty:
            final_df = final_df.sort_values(by="score", ascending=False).reset_index(drop=True)
            
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nHoàn thành! Đã lưu kết quả tại: {OUTPUT_CSV}")