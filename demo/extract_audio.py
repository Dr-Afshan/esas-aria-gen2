"""
Export audio from a VRS recording as a WAV file.

Note: The Aria Gen 2 records 7-channel spatial audio optimised for
direction-of-arrival estimation, not for casual listening. Standard
media players will not play 7-channel WAV files correctly.

Use --mono to export a single channel for playback verification.

Usage:
    python demo/extract_audio.py --vrs recording.vrs           # 7-channel WAV
    python demo/extract_audio.py --vrs recording.vrs --mono    # single channel
    python demo/extract_audio.py --vrs recording.vrs --channel 3  # specific mic
"""

import argparse
import wave
from pathlib import Path

import numpy as np


def load_audio(vrs_path):
    from projectaria_tools.core import data_provider as dp
    provider = dp.create_vrs_data_provider(str(vrs_path))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vrs',     required=True)
    ap.add_argument('--out',     default='')
    ap.add_argument('--mono',    action='store_true',
                    help='Export single channel for playback')
    ap.add_argument('--channel', type=int, default=0,
                    help='Mic channel to use for mono export (0-6)')
    args = ap.parse_args()

    vrs  = Path(args.vrs)
    print(f'Loading {vrs.name}...')
    audio = load_audio(vrs)

    if audio is None:
        print('No audio found.')
        return

    n_ch, n_samp = audio.shape
    dur = n_samp / 48_000
    print(f'Duration: {dur:.1f}s   Channels: {n_ch}')

    # Normalise
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    if args.mono:
        ch   = min(args.channel, n_ch - 1)
        data = (audio[ch] * 32767).astype(np.int16)
        suffix = f'_ch{ch}_mono'
        n_channels = 1
        samples = data.tobytes()
    else:
        # Interleaved 7-channel
        data = (audio * 32767).astype(np.int16)
        suffix = '_7ch'
        n_channels = n_ch
        samples = data.T.flatten().tobytes()

    out = Path(args.out) if args.out else vrs.with_suffix(f'{suffix}.wav')
    with wave.open(str(out), 'w') as f:
        f.setnchannels(n_channels)
        f.setsampwidth(2)
        f.setframerate(48_000)
        f.writeframes(samples)

    print(f'Saved: {out}')
    if not args.mono:
        print('Note: 7-channel audio is for GCC-PHAT processing, '
              'not for media players. Use --mono to hear it.')


if __name__ == '__main__':
    main()
