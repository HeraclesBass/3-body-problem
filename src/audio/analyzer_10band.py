"""
10-Band Audio Analysis for Ultra-Reactive Physics

Extracts 10 frequency bands for granular control over physics and visuals:
- Sub-bass (20-60 Hz) → Gravity strength
- Bass (60-150 Hz) → Acceleration bursts
- Low-mid (150-300 Hz) → Trail decay
- Mid (300-600 Hz) → Particle intensity
- High-mid (600-1200 Hz) → Color saturation
- Presence (1.2k-2.5k Hz) → Glow intensity
- Brilliance (2.5k-5k Hz) → Trail brightness
- Air (5k-10k Hz) → Shimmer effect
- Ultra (10k-16k Hz) → Background nebula
- Extreme (16k-20k Hz) → Sparkle highlights
"""

import numpy as np
from dataclasses import dataclass
import librosa


@dataclass
class AudioFrame10Band:
    """Audio analysis with 10 frequency bands."""
    # 10 frequency bands
    sub_bass: float      # 20-60 Hz
    bass: float          # 60-150 Hz
    low_mid: float       # 150-300 Hz
    mid: float           # 300-600 Hz
    high_mid: float      # 600-1200 Hz
    presence: float      # 1200-2500 Hz
    brilliance: float    # 2500-5000 Hz
    air: float           # 5000-10000 Hz
    ultra: float         # 10000-16000 Hz
    extreme: float       # 16000-20000 Hz

    # Beat/onset features
    beat_strength: float
    onset_strength: float
    tempo: float

    # Time
    time: float


class AudioAnalyzer10Band:
    """
    Ultra-detailed audio analysis with 10 frequency bands.

    Provides granular control for audio-reactive physics and visuals.
    """

    # 10-band frequency boundaries (Hz)
    BANDS = [
        (20, 60, 'sub_bass'),
        (60, 150, 'bass'),
        (150, 300, 'low_mid'),
        (300, 600, 'mid'),
        (600, 1200, 'high_mid'),
        (1200, 2500, 'presence'),
        (2500, 5000, 'brilliance'),
        (5000, 10000, 'air'),
        (10000, 16000, 'ultra'),
        (16000, 20000, 'extreme'),
    ]

    def __init__(
        self,
        audio_path: str,
        fps: float = 60.0,
        hop_length: int = 512
    ):
        """
        Load and analyze audio with 10-band resolution.

        Args:
            audio_path: Path to audio file
            fps: Target frame rate
            hop_length: STFT hop length (smaller = better time resolution)
        """
        self.fps = fps
        self.hop_length = hop_length

        print(f"Loading audio: {audio_path}")
        self.y, self.sr = librosa.load(audio_path, sr=None, mono=True)
        self.duration = len(self.y) / self.sr
        self.total_frames = int(self.duration * fps)

        print(f"  Duration: {self.duration:.1f}s @ {self.sr} Hz")
        print(f"  Target frames: {self.total_frames}")

        # Compute STFT
        print("Computing STFT...")
        self.stft = np.abs(librosa.stft(self.y, hop_length=hop_length))
        self.freqs = librosa.fft_frequencies(sr=self.sr)

        # Beat and onset detection
        print("Detecting beats...")
        self._compute_beats()
        self._compute_onsets()

        # 10-band energy extraction
        print("Extracting 10-band energies...")
        self._compute_all_bands()

        print("✅ Audio analysis complete!")

    def _compute_beats(self):
        """Detect beats in audio."""
        self.tempo, self.beats = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        self.beat_times = librosa.frames_to_time(
            self.beats, sr=self.sr, hop_length=self.hop_length
        )

    def _compute_onsets(self):
        """Compute onset strength envelope."""
        self.onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )

    def _compute_all_bands(self):
        """Pre-compute energy for all 10 frequency bands."""
        self.band_energies = {}

        for low, high, name in self.BANDS:
            # Create frequency mask
            mask = (self.freqs >= low) & (self.freqs < high)

            # RMS energy in this band
            energy = np.sqrt(np.mean(self.stft[mask, :] ** 2, axis=0))

            # Normalize to [0, 1]
            energy = self._normalize(energy)

            self.band_energies[name] = energy

        # Normalize onset envelope
        self.onset_env = self._normalize(self.onset_env)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        return np.zeros_like(arr)

    def _time_to_stft_frame(self, time: float) -> int:
        """Convert time to STFT frame index."""
        return int(time * self.sr / self.hop_length)

    def get_frame(self, frame_idx: int) -> AudioFrame10Band:
        """
        Get 10-band audio analysis for a video frame.

        Args:
            frame_idx: Video frame index

        Returns:
            AudioFrame10Band with all frequency bands
        """
        time = frame_idx / self.fps
        stft_idx = min(self._time_to_stft_frame(time),
                      len(self.band_energies['bass']) - 1)

        # Beat detection
        beat_strength = 0.0
        if len(self.beat_times) > 0:
            distances = np.abs(self.beat_times - time)
            min_dist = distances.min()
            if min_dist < 0.05:  # 50ms window
                beat_strength = 1.0 - (min_dist / 0.05)

        return AudioFrame10Band(
            sub_bass=float(self.band_energies['sub_bass'][stft_idx]),
            bass=float(self.band_energies['bass'][stft_idx]),
            low_mid=float(self.band_energies['low_mid'][stft_idx]),
            mid=float(self.band_energies['mid'][stft_idx]),
            high_mid=float(self.band_energies['high_mid'][stft_idx]),
            presence=float(self.band_energies['presence'][stft_idx]),
            brilliance=float(self.band_energies['brilliance'][stft_idx]),
            air=float(self.band_energies['air'][stft_idx]),
            ultra=float(self.band_energies['ultra'][stft_idx]),
            extreme=float(self.band_energies['extreme'][stft_idx]),
            beat_strength=beat_strength,
            onset_strength=float(self.onset_env[stft_idx]) if stft_idx < len(self.onset_env) else 0.0,
            tempo=float(self.tempo),
            time=time
        )

    def get_summary(self) -> dict:
        """Get 10-band audio summary statistics."""
        summary = {
            'duration_sec': self.duration,
            'sample_rate': self.sr,
            'total_frames': self.total_frames,
            'tempo_bpm': float(self.tempo),
            'num_beats': len(self.beats),
        }

        # Add average energy per band
        for name in self.band_energies:
            summary[f'avg_{name}'] = float(self.band_energies[name].mean())
            summary[f'max_{name}'] = float(self.band_energies[name].max())

        return summary
