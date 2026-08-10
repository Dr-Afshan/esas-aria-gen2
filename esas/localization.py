"""
Direction-of-arrival estimation using GCC-PHAT.

Implements the three equations from Section III-C of the paper:

  (1)  R12(tau) = IFFT[ X1(f) · X2*(f) / |X1(f) · X2*(f)| ]
  (2)  tau_hat  = argmax R12(tau)
  (3)  theta    = arcsin( tau_hat · c / d )

The key practical contribution is direct-path isolation: skipping the
first 50 ms after sound onset and using a 1.5 s sustained window reduces
frontal MAE from 56.3 degrees to 5.5 degrees in both real-room and studio conditions.
"""

import math
import numpy as np

SAMPLE_RATE  = 48_000
MIC_BASELINE = 0.060   # metres between channels 0 and 1
SOUND_SPEED  = 343.0   # m/s at 20°C


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray,
             sr: int = SAMPLE_RATE,
             d: float = MIC_BASELINE) -> tuple[float, float]:
    """
    Estimate TDOA and azimuth angle from two microphone signals.

    Returns
    -------
    azimuth_deg : float
        Estimated horizontal angle in degrees. Negative = left, positive = right.
    tdoa_s : float
        Time difference of arrival in seconds.
    """
    n    = 2 * int(2 ** math.ceil(math.log2(max(len(sig1), len(sig2)))))
    X1   = np.fft.rfft(sig1, n=n)
    X2   = np.fft.rfft(sig2, n=n)
    cross = X1 * np.conj(X2)
    gcc   = np.fft.irfft(cross / (np.abs(cross) + 1e-10), n=n)

    max_lag  = int(sr * d / SOUND_SPEED)
    gcc_half = np.concatenate([gcc[-max_lag:], gcc[:max_lag + 1]])
    peak     = int(np.argmax(gcc_half)) - max_lag
    tdoa_s   = peak / sr

    sin_val     = np.clip(tdoa_s * SOUND_SPEED / d, -1.0, 1.0)
    azimuth_deg = float(math.degrees(math.asin(sin_val)))

    return azimuth_deg, tdoa_s


def find_onset(audio: np.ndarray, sr: int = SAMPLE_RATE,
               frame: int = 512) -> int | None:
    """
    Find the sample index where the sound starts.
    Returns None if no onset detected above background level.
    """
    mono   = audio[0] if audio.ndim == 2 else audio
    rms    = [np.sqrt(np.mean(mono[i:i+frame]**2))
              for i in range(0, len(mono) - frame, frame)]
    rms    = np.array(rms)
    bg     = float(np.median(np.sort(rms)[:max(1, len(rms)//3)]))
    thresh = max(bg * 5, 0.005)
    for i, r in enumerate(rms):
        if r > thresh:
            return i * frame
    return None


def estimate_direction(audio: np.ndarray,
                       sr: int = SAMPLE_RATE) -> dict:
    """
    Full direction-of-arrival pipeline with direct-path isolation.

    Parameters
    ----------
    audio : np.ndarray
        Shape (n_channels, n_samples). Requires at least 2 channels.

    Returns
    -------
    dict with keys: azimuth_deg, direction, tdoa_s
    """
    if audio.ndim == 1 or audio.shape[0] < 2:
        return {'azimuth_deg': 0.0, 'direction': 'UNKNOWN', 'tdoa_s': 0.0}

    # Direct-path isolation
    onset = find_onset(audio, sr)
    if onset is not None:
        skip = int(sr * 0.05)      # skip 50 ms
        win  = int(sr * 1.5)       # use 1.5 s
        s    = onset + skip
        e    = s + win
        if e <= audio.shape[1]:
            audio = audio[:, s:e]

    azimuth, tdoa = gcc_phat(audio[0].astype(np.float32),
                             audio[1].astype(np.float32), sr)

    if   azimuth < -45: direction = 'FAR LEFT'
    elif azimuth < -15: direction = 'LEFT'
    elif azimuth <  15: direction = 'FRONT'
    elif azimuth <  45: direction = 'RIGHT'
    else:               direction = 'FAR RIGHT'

    return {'azimuth_deg': azimuth, 'direction': direction, 'tdoa_s': tdoa}
