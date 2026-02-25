"""
Harmonic Audio Analysis for V3 Audio-Reactive Systems

Extends 10-band analysis with:
- Musical note frequency detection (A4=440Hz, chromatic scale)
- Harmonic overtone detection (fundamental + 2x, 3x, 4x)
- Chord detection (major/minor triads, 7ths)
- Beat strength with attack/decay envelopes (not binary)
- Spectral flux for detecting note onsets

Used by:
- Frequency zone system (particle assignment)
- Harmonic resonance engine (particle vibration)
- Spectral spawning (note-based particle creation)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import librosa


@dataclass
class HarmonicFrame:
    """Enhanced audio frame with harmonic analysis."""
    # Time
    time: float

    # Dominant musical notes (up to 3)
    dominant_notes: List[Tuple[str, float]]  # [(note_name, frequency), ...]

    # Harmonic overtones for dominant note
    harmonics: List[float]  # [fundamental, 2x, 3x, 4x]

    # Detected chord (if any)
    chord: Optional[str]  # "Cmaj", "Dmin", "G7", etc.
    chord_confidence: float

    # Beat strength with envelope (0-1)
    beat_strength: float  # Smooth envelope, not binary
    beat_attack: bool     # True on beat onset

    # Spectral flux (change in spectrum)
    spectral_flux: float

    # Overall harmonic content (musical vs noise)
    harmonicity: float


class HarmonicAnalyzer:
    """
    Enhanced audio analyzer for musical harmony detection.

    Detects notes, chords, harmonics, and provides smooth beat envelopes.
    """

    # Musical note frequencies (A4=440Hz standard)
    # 12 notes per octave, covering 3 octaves (C3-B5)
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    A4_FREQ = 440.0

    # Chord patterns (semitone intervals from root)
    CHORD_PATTERNS = {
        'maj': [0, 4, 7],           # Major triad
        'min': [0, 3, 7],           # Minor triad
        'dim': [0, 3, 6],           # Diminished
        'aug': [0, 4, 8],           # Augmented
        'maj7': [0, 4, 7, 11],      # Major 7th
        'min7': [0, 3, 7, 10],      # Minor 7th
        '7': [0, 4, 7, 10],         # Dominant 7th
    }

    def __init__(
        self,
        audio_path: str,
        fps: float = 60.0,
        hop_length: int = 512,
        n_fft: int = 4096  # Larger FFT for better frequency resolution
    ):
        """
        Initialize harmonic analyzer.

        Args:
            audio_path: Path to audio file
            fps: Target frame rate
            hop_length: STFT hop length
            n_fft: FFT size (larger = better frequency resolution)
        """
        self.fps = fps
        self.hop_length = hop_length
        self.n_fft = n_fft

        print(f"Loading audio for harmonic analysis: {audio_path}")
        self.y, self.sr = librosa.load(audio_path, sr=None, mono=True)
        self.duration = len(self.y) / self.sr
        self.total_frames = int(self.duration * fps)

        print(f"  Duration: {self.duration:.1f}s @ {self.sr} Hz")
        print(f"  Target frames: {self.total_frames}")

        # Compute high-resolution STFT for note detection
        print("Computing high-resolution STFT...")
        self.stft = np.abs(librosa.stft(
            self.y, n_fft=n_fft, hop_length=hop_length
        ))
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)

        # Chromagram for pitch detection
        print("Computing chromagram...")
        self.chroma = librosa.feature.chroma_stft(
            y=self.y, sr=self.sr, hop_length=hop_length
        )

        # Beat tracking with strength envelope
        print("Detecting beats with envelope...")
        self._compute_beat_envelope()

        # Spectral flux for onset detection
        print("Computing spectral flux...")
        self._compute_spectral_flux()

        # Harmonicity estimation
        print("Estimating harmonicity...")
        self._compute_harmonicity()

        print("✅ Harmonic analysis complete!")

    def _compute_beat_envelope(self):
        """Compute smooth beat strength envelope (not binary)."""
        # Get beat times
        self.tempo, self.beats = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        self.beat_times = librosa.frames_to_time(
            self.beats, sr=self.sr, hop_length=self.hop_length
        )

        # Create smooth envelope with attack/decay
        n_frames = self.stft.shape[1]
        self.beat_envelope = np.zeros(n_frames)

        attack_frames = int(0.05 * self.sr / self.hop_length)  # 50ms attack
        decay_frames = int(0.3 * self.sr / self.hop_length)    # 300ms decay

        for beat_frame in self.beats:
            if beat_frame >= n_frames:
                continue

            # Attack phase (instant rise)
            self.beat_envelope[beat_frame] = 1.0

            # Decay phase (exponential)
            for i in range(1, decay_frames):
                frame_idx = beat_frame + i
                if frame_idx >= n_frames:
                    break
                decay_val = np.exp(-3 * i / decay_frames)  # Exponential decay
                self.beat_envelope[frame_idx] = max(
                    self.beat_envelope[frame_idx], decay_val
                )

    def _compute_spectral_flux(self):
        """Compute spectral flux (measure of spectral change)."""
        # Difference between consecutive frames
        diff = np.diff(self.stft, axis=1)
        # Only positive changes (energy increases)
        diff = np.maximum(diff, 0)
        # Sum across frequencies
        self.spectral_flux = np.sum(diff, axis=0)
        # Normalize
        if self.spectral_flux.max() > 0:
            self.spectral_flux = self.spectral_flux / self.spectral_flux.max()
        # Prepend zero for first frame
        self.spectral_flux = np.concatenate([[0], self.spectral_flux])

    def _compute_harmonicity(self):
        """Estimate harmonicity (how musical vs noisy the signal is)."""
        # Use harmonic-percussive separation
        harmonic, _ = librosa.effects.hpss(self.y)

        # Compute RMS energy for harmonic component per frame
        harmonic_rms = librosa.feature.rms(
            y=harmonic, hop_length=self.hop_length
        )[0]
        total_rms = librosa.feature.rms(
            y=self.y, hop_length=self.hop_length
        )[0]

        # Ratio of harmonic to total energy
        self.harmonicity = np.divide(
            harmonic_rms,
            total_rms,
            out=np.zeros_like(harmonic_rms),
            where=total_rms > 1e-6
        )

    def _freq_to_note_name(self, freq: float) -> str:
        """Convert frequency to note name (e.g., 440Hz -> A4)."""
        if freq <= 0:
            return "?"

        # Calculate semitones from A4
        semitones = 12 * np.log2(freq / self.A4_FREQ)
        # A is the 9th note (index 9) when starting from C
        note_idx = (int(round(semitones)) + 9) % 12
        octave = 4 + (int(round(semitones)) + 9) // 12

        return f"{self.NOTE_NAMES[note_idx]}{octave}"

    def _detect_dominant_notes(self, chroma_frame: np.ndarray, n_notes: int = 3) -> List[Tuple[str, float]]:
        """
        Detect dominant musical notes in a frame.

        Args:
            chroma_frame: 12-bin chromagram for this frame
            n_notes: Number of notes to return

        Returns:
            List of (note_name, strength) tuples
        """
        # Find peaks in chromagram
        note_strengths = []
        for i in range(12):
            strength = chroma_frame[i]
            if strength > 0.1:  # Threshold for detection
                # Convert chroma bin to approximate frequency (A4=440 reference)
                # Chroma bin 9 = A, so offset accordingly
                semitones_from_a = (i - 9) % 12
                freq = self.A4_FREQ * (2 ** (semitones_from_a / 12))
                note_name = self._freq_to_note_name(freq)
                note_strengths.append((note_name, strength, freq))

        # Sort by strength
        note_strengths.sort(key=lambda x: x[1], reverse=True)

        # Return top n_notes with (name, frequency)
        return [(name, freq) for name, _, freq in note_strengths[:n_notes]]

    def _detect_chord(self, chroma_frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect chord from chromagram.

        Args:
            chroma_frame: 12-bin chromagram

        Returns:
            (chord_name, confidence) or (None, 0.0)
        """
        # Normalize chroma
        if chroma_frame.max() > 0:
            chroma_norm = chroma_frame / chroma_frame.max()
        else:
            return None, 0.0

        best_chord = None
        best_confidence = 0.0

        # Try each root note
        for root in range(12):
            # Try each chord type
            for chord_type, intervals in self.CHORD_PATTERNS.items():
                # Calculate expected chroma for this chord
                expected = np.zeros(12)
                for interval in intervals:
                    expected[(root + interval) % 12] = 1.0

                # Correlation between expected and actual
                correlation = np.dot(chroma_norm, expected) / len(intervals)

                if correlation > best_confidence:
                    best_confidence = correlation
                    root_name = self.NOTE_NAMES[root]
                    best_chord = f"{root_name}{chord_type}"

        # Threshold for chord detection
        if best_confidence < 0.5:
            return None, 0.0

        return best_chord, best_confidence

    def _get_harmonics(self, fundamental_freq: float) -> List[float]:
        """Get harmonic overtones for a fundamental frequency."""
        return [
            fundamental_freq,      # 1st harmonic (fundamental)
            fundamental_freq * 2,  # 2nd harmonic
            fundamental_freq * 3,  # 3rd harmonic
            fundamental_freq * 4,  # 4th harmonic
        ]

    def _time_to_stft_frame(self, time: float) -> int:
        """Convert time to STFT frame index."""
        return int(time * self.sr / self.hop_length)

    def get_frame(self, frame_idx: int) -> HarmonicFrame:
        """
        Get harmonic analysis for a video frame.

        Args:
            frame_idx: Video frame index

        Returns:
            HarmonicFrame with note, chord, and harmonic data
        """
        time = frame_idx / self.fps
        stft_idx = min(
            self._time_to_stft_frame(time),
            self.chroma.shape[1] - 1
        )

        # Get chromagram for this frame
        chroma_frame = self.chroma[:, stft_idx]

        # Detect dominant notes
        dominant_notes = self._detect_dominant_notes(chroma_frame)

        # Get harmonics for most dominant note
        harmonics = []
        if dominant_notes:
            harmonics = self._get_harmonics(dominant_notes[0][1])

        # Detect chord
        chord, chord_conf = self._detect_chord(chroma_frame)

        # Get beat strength (smooth envelope)
        beat_strength = float(self.beat_envelope[stft_idx])

        # Check if this is a beat attack (transition from low to high)
        beat_attack = False
        if stft_idx > 0:
            prev_strength = self.beat_envelope[stft_idx - 1]
            beat_attack = (beat_strength > 0.8 and prev_strength < 0.5)

        # Get spectral flux
        spectral_flux = float(self.spectral_flux[stft_idx])

        # Get harmonicity
        harmonicity = float(self.harmonicity[stft_idx])

        return HarmonicFrame(
            time=time,
            dominant_notes=dominant_notes,
            harmonics=harmonics,
            chord=chord,
            chord_confidence=chord_conf,
            beat_strength=beat_strength,
            beat_attack=beat_attack,
            spectral_flux=spectral_flux,
            harmonicity=harmonicity
        )

    def get_summary(self) -> dict:
        """Get harmonic analysis summary."""
        return {
            'duration_sec': self.duration,
            'sample_rate': self.sr,
            'total_frames': self.total_frames,
            'tempo_bpm': float(self.tempo),
            'num_beats': len(self.beats),
            'avg_harmonicity': float(self.harmonicity.mean()),
            'avg_spectral_flux': float(self.spectral_flux.mean()),
        }
