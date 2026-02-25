"""
Audio analysis for music-reactive physics.

Extracts frequency bands and beat information from WAV files
for real-time physics modulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import librosa


@dataclass
class AudioFrame:
    """Audio analysis for a single frame."""
    bass_energy: float      # 20-250 Hz
    mid_energy: float       # 250-4000 Hz
    treble_energy: float    # 4000-20000 Hz
    beat_strength: float    # Beat detection value
    onset_strength: float   # Onset detection value
    time: float            # Timestamp in seconds


class AudioAnalyzer:
    """
    Analyze WAV audio for physics modulation.

    Extracts:
    - Frequency band energy (bass, mid, treble)
    - Beat detection
    - Onset strength

    Example:
        analyzer = AudioAnalyzer("music.wav", fps=60)
        for frame_idx in range(analyzer.total_frames):
            audio = analyzer.get_frame(frame_idx)
            # Use audio.bass_energy etc. for physics modulation
    """

    # Frequency band boundaries (Hz)
    BASS_LOW = 20
    BASS_HIGH = 250
    MID_LOW = 250
    MID_HIGH = 4000
    TREBLE_LOW = 4000
    TREBLE_HIGH = 20000

    def __init__(
        self,
        audio_path: str,
        fps: float = 60.0,
        hop_length: int = 512
    ):
        """
        Load and analyze audio file.

        Args:
            audio_path: Path to WAV file
            fps: Target frame rate for analysis
            hop_length: STFT hop length
        """
        self.fps = fps
        self.hop_length = hop_length

        # Load audio
        self.y, self.sr = librosa.load(audio_path, sr=None, mono=True)
        self.duration = len(self.y) / self.sr
        self.total_frames = int(self.duration * fps)

        # Compute STFT
        self.stft = np.abs(librosa.stft(self.y, hop_length=hop_length))
        self.freqs = librosa.fft_frequencies(sr=self.sr)

        # Compute beat and onset features
        self._compute_beats()
        self._compute_onsets()

        # Pre-compute frequency band masks
        self._compute_band_masks()

        # Pre-compute band energies for all frames
        self._compute_all_energies()

    def _compute_band_masks(self):
        """Create boolean masks for frequency bands."""
        self.bass_mask = (self.freqs >= self.BASS_LOW) & (self.freqs < self.BASS_HIGH)
        self.mid_mask = (self.freqs >= self.MID_LOW) & (self.freqs < self.MID_HIGH)
        self.treble_mask = (self.freqs >= self.TREBLE_LOW) & (self.freqs < self.TREBLE_HIGH)

    def _compute_beats(self):
        """Detect beats in audio."""
        self.tempo, self.beats = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        # Convert beat frames to time
        self.beat_times = librosa.frames_to_time(
            self.beats, sr=self.sr, hop_length=self.hop_length
        )

    def _compute_onsets(self):
        """Compute onset strength envelope."""
        self.onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )

    def _compute_all_energies(self):
        """Pre-compute band energies for all STFT frames."""
        # RMS energy per band
        self.bass_energy = np.sqrt(np.mean(self.stft[self.bass_mask, :] ** 2, axis=0))
        self.mid_energy = np.sqrt(np.mean(self.stft[self.mid_mask, :] ** 2, axis=0))
        self.treble_energy = np.sqrt(np.mean(self.stft[self.treble_mask, :] ** 2, axis=0))

        # Normalize to [0, 1] range
        self.bass_energy = self._normalize(self.bass_energy)
        self.mid_energy = self._normalize(self.mid_energy)
        self.treble_energy = self._normalize(self.treble_energy)

        # Normalize onset envelope
        self.onset_env = self._normalize(self.onset_env)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        return np.zeros_like(arr)

    def _time_to_stft_frame(self, time: float) -> int:
        """Convert time in seconds to STFT frame index."""
        return int(time * self.sr / self.hop_length)

    def get_frame(self, frame_idx: int) -> AudioFrame:
        """
        Get audio analysis for a video frame.

        Args:
            frame_idx: Video frame index (0 to total_frames-1)

        Returns:
            AudioFrame with energy and beat data
        """
        time = frame_idx / self.fps
        stft_idx = min(self._time_to_stft_frame(time), len(self.bass_energy) - 1)

        # Check if this frame is near a beat
        beat_strength = 0.0
        if len(self.beat_times) > 0:
            distances = np.abs(self.beat_times - time)
            min_dist = distances.min()
            # Beat window: 50ms
            if min_dist < 0.05:
                beat_strength = 1.0 - (min_dist / 0.05)

        return AudioFrame(
            bass_energy=float(self.bass_energy[stft_idx]),
            mid_energy=float(self.mid_energy[stft_idx]),
            treble_energy=float(self.treble_energy[stft_idx]),
            beat_strength=beat_strength,
            onset_strength=float(self.onset_env[stft_idx]) if stft_idx < len(self.onset_env) else 0.0,
            time=time
        )

    def get_params_dict(self, frame_idx: int) -> dict:
        """
        Get audio parameters as dict for physics kernel.

        Args:
            frame_idx: Video frame index

        Returns:
            Dict compatible with NBodySimulation.step(audio_params=...)
        """
        frame = self.get_frame(frame_idx)
        return {
            'bass_energy': frame.bass_energy,
            'mid_energy': frame.mid_energy,
            'treble_energy': frame.treble_energy,
            'modulation_depth': 0.5 + frame.beat_strength * 0.5
        }

    def get_summary(self) -> dict:
        """Get summary statistics of the audio."""
        return {
            'duration_sec': self.duration,
            'sample_rate': self.sr,
            'total_frames': self.total_frames,
            'tempo_bpm': float(self.tempo),
            'num_beats': len(self.beats),
            'avg_bass': float(self.bass_energy.mean()),
            'avg_mid': float(self.mid_energy.mean()),
            'avg_treble': float(self.treble_energy.mean())
        }
