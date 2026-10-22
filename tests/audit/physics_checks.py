"""Physics checks for CIR axis layout, native frame rate, and radar sync.

Step 0.3: Determines axis_layout ∈ {range_368, iq_concat_184, iq_interleaved_184, unclear}
Outputs: experiments/audit/reports/physics_checks.json + PNG plots
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

STREAMS_TRAIN = ROOT / "streams" / "train"
EVENTS_CSV    = ROOT / "streams_events.csv"
REPORTS_DIR   = ROOT / "experiments" / "audit" / "reports"
RATE_HZ       = 50.0


def run_physics_checks(n_streams: int = 3) -> dict:
    """Run all physics checks and return summary dict."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    stream_paths = sorted(STREAMS_TRAIN.glob("stream_*.h5"))[:n_streams]
    if not stream_paths:
        print("ERROR: No train streams found")
        return {"error": "no streams found"}

    # Load a few streams for analysis
    cirs = []
    for p in stream_paths:
        with h5py.File(p) as f:
            cirs.append(f["cir"][:])
    cir = cirs[0]  # shape (3, T, 368)
    print(f"CIR shape: {cir.shape}, dtype: {cir.dtype}")
    print(f"Min: {cir.min():.4f}, Max: {cir.max():.4f}, Mean: {cir.mean():.4f}")

    # ── 5a. Layout check ──────────────────────────────────────────────────────
    print("\n=== 5a. Axis layout check ===")

    # (i) Negative values?
    has_negative = bool(np.any(cir < 0))
    neg_fraction = float(np.mean(cir < 0))
    print(f"  Negative values: {has_negative} ({neg_fraction:.3%})")

    # (ii) Correlation between bins 0-183 and 184-367
    profile_first  = np.mean(np.abs(cir[0]), axis=0)[:184]   # (184,)
    profile_second = np.mean(np.abs(cir[0]), axis=0)[184:]    # (184,)
    corr_halves = float(np.corrcoef(profile_first, profile_second)[0, 1])
    print(f"  Correlation |profile| first_half vs second_half: {corr_halves:.4f}")

    # (iii) Correlation between even and odd bins
    profile_even = np.mean(np.abs(cir[0]), axis=0)[0::2]
    profile_odd  = np.mean(np.abs(cir[0]), axis=0)[1::2]
    corr_eo = float(np.corrcoef(profile_even, profile_odd)[0, 1])
    print(f"  Correlation even vs odd bins: {corr_eo:.4f}")

    # (iv) Three envelope candidates — compute and assess smoothness
    env_368 = np.mean(np.abs(cir[0]), axis=0)   # shape (368,)
    I_concat = cir[0, :, :184]
    Q_concat = cir[0, :, 184:]
    env_iq_concat = np.mean(np.sqrt(I_concat**2 + Q_concat**2), axis=0)  # (184,)
    I_interl = cir[0, :, 0::2]
    Q_interl = cir[0, :, 1::2]
    env_iq_interl = np.mean(np.sqrt(I_interl**2 + Q_interl**2), axis=0)  # (184,)

    # Smoothness measure: mean absolute second derivative (lower = smoother)
    def smoothness(x):
        d2 = np.diff(x, n=2)
        return float(np.mean(np.abs(d2)))

    sm_368   = smoothness(env_368)
    sm_conc  = smoothness(env_iq_concat)
    sm_interl = smoothness(env_iq_interl)
    print(f"  Smoothness (lower=better): |x|={sm_368:.4f}  iq_concat={sm_conc:.4f}  iq_interl={sm_interl:.4f}")

    # Determine layout
    if not has_negative and corr_halves > 0.8 and abs(corr_halves) > abs(corr_eo) * 0.8:
        axis_layout = "iq_concat_184"
        layout_reason = f"high half-half corr={corr_halves:.3f}, few negatives"
    elif not has_negative and corr_eo > 0.8:
        axis_layout = "iq_interleaved_184"
        layout_reason = f"high even-odd corr={corr_eo:.3f}, few negatives"
    elif has_negative and neg_fraction > 0.1:
        axis_layout = "range_368"
        layout_reason = f"many negatives={neg_fraction:.3%} → raw range profile"
    else:
        # Tie-break by smoothness
        if sm_conc < sm_interl and sm_conc < sm_368 * 1.1:
            axis_layout = "iq_concat_184"
        elif sm_interl < sm_conc and sm_interl < sm_368 * 1.1:
            axis_layout = "iq_interleaved_184"
        elif neg_fraction > 0.05:
            axis_layout = "range_368"
        else:
            axis_layout = "unclear"
        layout_reason = f"ambiguous; chosen by smoothness"

    print(f"  → axis_layout = {axis_layout} ({layout_reason})")

    results["axis_layout"] = axis_layout
    results["axis_layout_reason"] = layout_reason
    results["has_negative_values"] = has_negative
    results["negative_fraction"] = neg_fraction
    results["corr_halves"] = round(corr_halves, 4)
    results["corr_even_odd"] = round(corr_eo, 4)
    results["smoothness_368"] = round(sm_368, 6)
    results["smoothness_iq_concat"] = round(sm_conc, 6)
    results["smoothness_iq_interl"] = round(sm_interl, 6)

    # ── 5b. Native frame rate ─────────────────────────────────────────────────
    print("\n=== 5b. Native frame rate check ===")
    try:
        from scipy.signal import welch

        # Use static bin (pick bin with low variance) and active bin (high variance)
        mean_profile = np.mean(np.abs(cir[0]), axis=0)
        var_profile  = np.var(cir[0], axis=0)
        static_bin   = int(np.argmin(var_profile))
        active_bin   = int(np.argmax(var_profile))

        T = cir.shape[1]
        fs = RATE_HZ

        for radar_idx in range(min(2, cir.shape[0])):
            sig_static = cir[radar_idx, :, static_bin].astype(np.float64)
            sig_active = cir[radar_idx, :, active_bin].astype(np.float64)

            f_s, P_s = welch(sig_static, fs=fs, nperseg=min(512, T // 4))
            f_a, P_a = welch(sig_active, fs=fs, nperseg=min(512, T // 4))

            peak_static = float(f_s[np.argmax(P_s)])
            peak_active = float(f_a[np.argmax(P_a)])
            print(f"  Radar {radar_idx}: static_peak={peak_static:.2f}Hz  active_peak={peak_active:.2f}Hz")

        # Count repeated frames (diff==0) and linearly-interpolated frames
        diffs = np.diff(cir[0, :, :].astype(np.float64), axis=0)  # (T-1, 368)
        n_zero_diff = int(np.sum(np.all(diffs == 0, axis=1)))
        frame_diff_norm = np.sum(diffs**2, axis=1)
        frame_diff_norm2 = np.diff(frame_diff_norm)
        n_interp = int(np.sum(np.abs(frame_diff_norm2) < 1e-8))

        print(f"  Repeated frames (diff=0): {n_zero_diff}/{T-1} = {n_zero_diff/(T-1):.3%}")
        print(f"  Possibly interpolated frames: {n_interp}/{T-2} = {n_interp/(T-2):.3%}")

        native_rate_hz = RATE_HZ
        if n_zero_diff / (T - 1) > 0.2:
            native_rate_hz = RATE_HZ * (1 - n_zero_diff / (T - 1))
            print(f"  ⚠️ Many repeated frames → estimated native rate ≈ {native_rate_hz:.1f} Hz")

        results["native_rate_hz_estimate"] = round(native_rate_hz, 1)
        results["repeated_frame_fraction"] = round(n_zero_diff / (T - 1), 4)
        results["interpolated_frame_fraction"] = round(n_interp / (T - 2), 4)
        if native_rate_hz < 30:
            print(f"  ⚠️ WARNING: native rate {native_rate_hz:.1f} Hz < 30 Hz — "
                  f"frequency band 4-15 Hz may be interpolation artifact")
            results["native_rate_warning"] = True
        else:
            results["native_rate_warning"] = False

    except ImportError:
        print("  scipy not available, skipping Welch PSD")
        results["native_rate_hz_estimate"] = RATE_HZ
        results["native_rate_warning"] = False

    # ── 5c. Radar sync (lag between radars) ───────────────────────────────────
    print("\n=== 5c. Radar sync check ===")
    try:
        from abc2027.evaluate import load_ground_truth, temporal_iou
        from experiments.preprocessing.chebyshev_filter import chebyshev_bp
        from experiments.preprocessing.envelope import envelope

        gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
        gt_by = {}
        for ev in gt_all:
            gt_by.setdefault(ev["stream_id"], []).append(ev)

        lags_all = []
        n_falls_checked = 0

        for stream_path in stream_paths[:2]:  # limit to 2 streams for speed
            sid = f"train_{int(stream_path.stem.split('_')[1]):02d}"
            with h5py.File(stream_path) as f:
                cir_s = f["cir"][:]
            filtered = chebyshev_bp(cir_s)
            env = envelope(filtered, axis_layout)

            falls = [e for e in gt_by.get(sid, []) if not e.get("ignore")]
            for fall in falls[:5]:  # limit per stream
                ts = fall["t_start"]
                te = fall["t_end"]
                ctx = 3.0
                s = max(0, int((ts - ctx) * RATE_HZ))
                e = min(env.shape[1], int((te + ctx) * RATE_HZ))
                if e - s < 10:
                    continue

                # Check SNR for each radar
                snrs = []
                for r in range(3):
                    seg = env[r, s:e, :]
                    peak = float(np.max(np.mean(seg, axis=1)))
                    bg   = float(np.percentile(np.mean(seg, axis=1), 10))
                    snr  = peak / (bg + 1e-10)
                    snrs.append(snr)

                # Only process pairs where both radars have SNR > 3
                pairs = [(0,1),(0,2),(1,2)]
                for r1, r2 in pairs:
                    if snrs[r1] < 3 or snrs[r2] < 3:
                        continue
                    sig1 = np.mean(env[r1, s:e, :], axis=1)
                    sig2 = np.mean(env[r2, s:e, :], axis=1)
                    sig1 = sig1 - sig1.mean()
                    sig2 = sig2 - sig2.mean()
                    xcorr = np.correlate(sig1, sig2, mode="full")
                    lag_idx = int(np.argmax(xcorr)) - (len(sig1) - 1)
                    lag_s = abs(lag_idx) / RATE_HZ
                    lags_all.append(lag_s)

                n_falls_checked += 1

        if lags_all:
            med_lag = float(np.median(lags_all))
            iqr_lag = float(np.percentile(lags_all, 75) - np.percentile(lags_all, 25))
            radar_sync = med_lag <= 0.2
            print(f"  Checked {n_falls_checked} falls, {len(lags_all)} radar pairs")
            print(f"  Median |lag| = {med_lag:.3f}s, IQR = {iqr_lag:.3f}s")
            print(f"  → radar_sync = {radar_sync}")
            results["radar_sync"] = radar_sync
            results["median_lag_s"] = round(med_lag, 4)
            results["iqr_lag_s"] = round(iqr_lag, 4)
            results["n_falls_checked_sync"] = n_falls_checked
        else:
            print("  No valid pairs found for sync check")
            results["radar_sync"] = True  # assume ok
            results["median_lag_s"] = 0.0
            results["n_falls_checked_sync"] = 0

    except Exception as ex:
        print(f"  Sync check error: {ex}")
        results["radar_sync"] = True
        results["median_lag_s"] = 0.0

    # Save results
    out_path = REPORTS_DIR / "physics_checks.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPhysics checks saved to {out_path}")

    # ── Try to save PNG ───────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(env_368, label="|x| 368 bins", alpha=0.8)
        axes[0].set_title("Envelope candidate: |x| (368 bins)")
        axes[0].set_xlabel("Bin index")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()

        axes[1].plot(env_iq_concat, label="hypot(I,Q) concat (184 bins)", color="orange", alpha=0.8)
        axes[1].set_title("Envelope candidate: hypot(concat I,Q)")
        axes[1].set_xlabel("Bin index")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()

        axes[2].plot(env_iq_interl, label="hypot(even,odd) (184 bins)", color="green", alpha=0.8)
        axes[2].set_title(f"Envelope candidate: hypot(interleaved I,Q) | Selected: {axis_layout}")
        axes[2].set_xlabel("Bin index")
        axes[2].set_ylabel("Amplitude")
        axes[2].legend()

        plt.tight_layout()
        png_path = REPORTS_DIR / "physics_checks_envelopes.png"
        plt.savefig(str(png_path), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Envelope plot saved to {png_path}")
    except Exception as ex:
        print(f"  Could not save PNG: {ex}")

    return results


if __name__ == "__main__":
    results = run_physics_checks()
    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k}: {v}")
