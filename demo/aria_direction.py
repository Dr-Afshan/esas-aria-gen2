"""
Estimate sound direction from a VRS recording.

Loads a VRS file, finds the sound onset, applies direct-path isolation,
and runs GCC-PHAT on all 6 microphone pairs. Prints the estimated angle
and a visual direction bar.

Usage:
    python demo/aria_direction.py --vrs path/to/recording.vrs
    python demo/aria_direction.py --vrs ofire0_1.vrs --debug
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from esas.localization import gcc_phat, find_onset, SAMPLE_RATE


def load_vrs(path):
    from projectaria_tools.core import data_provider as dp
    provider = dp.create_vrs_data_provider(str(path))
    audio_id = provider.get_stream_id_from_label('mic')
    n        = provider.get_num_data(audio_id)
    chunks   = []
    for i in range(n):
        frame, _ = provider.get_audio_data_by_index(audio_id, i)
        try:    raw = np.array(frame.data, dtype=np.float32)
        except: raw = np.array(frame.audio_array, dtype=np.float32)
        if raw.ndim == 1 and len(raw) % 7 == 0:
            chunks.append(raw.reshape(7, -1))
    return np.concatenate(chunks, axis=1) if chunks else None


def direction_bar(angle, width=41):
    pos = int((angle + 90) / 180 * width)
    pos = max(0, min(width - 1, pos))
    bar = ['-'] * width
    bar[width // 2] = ':'
    bar[pos] = '|'
    return ''.join(bar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vrs',   required=True)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    path = Path(args.vrs)
    if not path.exists():
        print(f'File not found: {path}')
        sys.exit(1)

    print(f'\nLoading {path.name}...')
    audio = load_vrs(path)
    if audio is None:
        print('No audio data found.')
        sys.exit(1)

    print(f'Duration: {audio.shape[1]/SAMPLE_RATE:.1f}s   Channels: {audio.shape[0]}')

    # Find onset and extract direct-path window
    onset = find_onset(audio)
    if onset is None:
        print('No sound onset detected.')
        sys.exit(1)

    skip = int(SAMPLE_RATE * 0.05)
    win  = int(SAMPLE_RATE * 1.5)
    s    = onset + skip
    e    = s + win
    if e > audio.shape[1]:
        e = audio.shape[1]
        s = max(0, e - win)
    window = audio[:, s:e]

    print(f'Onset: {onset/SAMPLE_RATE:.2f}s  '
          f'Window: {s/SAMPLE_RATE:.2f}–{e/SAMPLE_RATE:.2f}s')

    # GCC-PHAT on all 6 mic pairs
    pairs  = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    angles = []
    print('\nAll mic pair estimates:')
    for c1, c2 in pairs:
        if max(c1, c2) < window.shape[0]:
            a, _ = gcc_phat(window[c1].astype(np.float32),
                            window[c2].astype(np.float32))
            angles.append(a)
            if   a < -45: label = 'FAR LEFT'
            elif a < -15: label = 'LEFT'
            elif a <  15: label = 'FRONT'
            elif a <  45: label = 'RIGHT'
            else:         label = 'FAR RIGHT'
            print(f'  Mic ({c1},{c2}): {a:+.1f}°  [{label}]')

    if not angles:
        print('No valid mic pairs.')
        sys.exit(1)

    median = float(np.median(angles))
    if   median < -45: final = 'FAR LEFT'
    elif median < -15: final = 'LEFT'
    elif median <  15: final = 'FRONT'
    elif median <  45: final = 'RIGHT'
    else:              final = 'FAR RIGHT'

    print(f'\n{"="*50}')
    print(f'  Median angle:  {median:+.1f}°')
    print(f'  Direction:     {final}')
    print(f'  [{direction_bar(median)}]')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
