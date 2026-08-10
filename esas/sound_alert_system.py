"""
ESAS — Sound Alert System
Combines PANNs detection and GCC-PHAT localization into a single pipeline.
"""
from .detection import ESASDetector
from .localization import GCCPHATLocalizer

class SoundAlertSystem:
    """End-to-end ESAS pipeline: detect sound, estimate direction, raise alert."""

    HAZARD_LABELS = {
        'HIGH':   {'car_horn','chainsaw','crackling_fire','glass_breaking',
                   'hand_saw','siren','fireworks'},
        'MEDIUM': {'clock_alarm','clock_tick','crying_baby','dog',
                   'door_wood_knock','sneezing','coughing','church_bells',
                   'vacuum_cleaner','washing_machine','toilet_flush','cat'},
    }

    def __init__(self, model_path=None, sr=48000):
        self.detector   = ESASDetector(model_path=model_path)
        self.localizer  = GCCPHATLocalizer(sr=sr)
        self.sr         = sr

    def get_priority(self, label):
        if label in self.HAZARD_LABELS['HIGH']:   return 'HIGH'
        if label in self.HAZARD_LABELS['MEDIUM']: return 'MEDIUM'
        return 'LOW'

    def process(self, audio_multichannel):
        """
        Process a multi-channel audio chunk.

        Parameters
        ----------
        audio_multichannel : np.ndarray, shape (7, N)
            7-channel audio at self.sr Hz.

        Returns
        -------
        dict with keys: label, priority, angle_deg, confidence
        """
        mono  = audio_multichannel[0]
        label, confidence = self.detector.predict(mono)
        priority  = self.get_priority(label)
        angle_deg = self.localizer.estimate(audio_multichannel)
        return {
            'label':      label,
            'priority':   priority,
            'angle_deg':  angle_deg,
            'confidence': confidence,
        }
