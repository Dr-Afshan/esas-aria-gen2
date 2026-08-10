"""
Sound detection using PANNs CNN14.

Loads a pretrained CNN14 checkpoint and a fine-tuned logistic regression
head to classify 2-second audio windows into HIGH / MEDIUM / LOW hazard
priority tiers.

Reference: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks
for Audio Pattern Recognition," IEEE/ACM TASLP, 2020.
"""

import pickle
import warnings
import numpy as np
from pathlib import Path


# Hazard taxonomy from paper Table I (matches ESC-50 category names)
HAZARD_HIGH = {
    'car_horn', 'chainsaw', 'crackling_fire', 'glass_breaking',
    'hand_saw', 'siren', 'fireworks',
}

HAZARD_MEDIUM = {
    'clock_alarm', 'clock_tick', 'crying_baby', 'dog',
    'door_wood_knock', 'sneezing', 'coughing', 'church_bells',
    'vacuum_cleaner', 'washing_machine', 'toilet_flush', 'cat',
}


def get_priority(label: str) -> str:
    label = label.lower()
    if any(h in label for h in HAZARD_HIGH):
        return 'HIGH'
    if any(m in label for m in HAZARD_MEDIUM):
        return 'MEDIUM'
    return 'LOW'


class Detector:
    """
    Two-stage sound detector.

    Stage 1: PANNs CNN14 extracts a 2048-dim embedding and 527-class scores.
    Stage 2: Logistic regression head maps the embedding to HIGH/MEDIUM/LOW.

    Parameters
    ----------
    checkpoint : str or Path
        Path to Cnn14_mAP=0.431.pth
    lr_head : str or Path, optional
        Path to pickled LogisticRegression (esas_finetuned.pkl).
        If not provided, falls back to raw AudioSet scores.
    device : str
        'cpu' or 'cuda'
    """

    SR = 32_000
    WINDOW = 32_000 * 2  # 2 seconds
    THRESHOLD = 0.25

    def __init__(self, checkpoint, lr_head=None, device='cpu'):
        self.device = device
        self._load_panns(checkpoint)
        self._load_head(lr_head)

    def _load_panns(self, path):
        try:
            from panns_inference import AudioTagging
            warnings.filterwarnings('ignore')
            self.at = AudioTagging(checkpoint_path=str(path), device=self.device)
        except Exception as e:
            raise RuntimeError(f'Could not load PANNs: {e}') from e

    def _load_head(self, path):
        self.head = None
        if path and Path(path).exists():
            with open(path, 'rb') as f:
                self.head = pickle.load(f)

    def predict(self, audio: np.ndarray) -> list[dict]:
        """
        Classify a 2-second audio window.

        Parameters
        ----------
        audio : np.ndarray
            Shape (samples,), float32, normalised to [-1, 1].

        Returns
        -------
        List of dicts with keys: label, priority, confidence
        """
        audio = audio.astype(np.float32)
        if len(audio) < self.WINDOW:
            audio = np.pad(audio, (0, self.WINDOW - len(audio)))
        else:
            audio = audio[:self.WINDOW]

        try:
            scores, embedding = self.at.inference(audio[None, :], None)
            scores    = scores[0]
            embedding = embedding[0]
        except Exception:
            return []

        if self.head is not None:
            proba   = self.head.predict_proba(embedding[None, :])[0]
            classes = self.head.classes_
            results = []
            for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
                if prob >= self.THRESHOLD:
                    results.append({
                        'label':      cls,
                        'priority':   cls,
                        'confidence': float(prob),
                    })
            return results

        # Fallback: top AudioSet scores
        top = np.argsort(scores)[::-1][:5]
        results = []
        for idx in top:
            conf = float(scores[idx])
            if conf < self.THRESHOLD:
                break
            results.append({
                'label':      f'class_{idx}',
                'priority':   'LOW',
                'confidence': conf,
            })
        return results
