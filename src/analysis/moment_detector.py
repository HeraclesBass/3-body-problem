"""
Moment Detection System for V3

Identifies "golden moments" in rendered video for clip extraction.
Analyzes musical and visual features to find the most engaging segments.

Detection Criteria:
- Musical drops (sudden energy increase)
- Build-ups (gradual energy rise)
- Visual complexity peaks (lots of particles, fast motion)
- Harmonic richness (musical vs. noisy)

Usage:
    detector = MomentDetector()

    # Record frame data during rendering
    detector.record_frame(
        time=t,
        harmonic_frame=frame,
        particle_count=count,
        visual_complexity=complexity
    )

    # After rendering, analyze moments
    moments = detector.detect_moments()

    # Get best clips
    clips = detector.get_clip_markers(
        duration=15.0,  # 15-second clips
        count=5         # Top 5 clips
    )
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class FrameData:
    """Data recorded for each frame."""
    time: float
    beat_strength: float
    beat_attack: bool
    spectral_flux: float
    harmonicity: float
    particle_count: int
    visual_complexity: float  # Custom metric (particle speed, spread, etc.)


@dataclass
class Moment:
    """A detected interesting moment."""
    time: float
    duration: float
    score: float
    moment_type: str  # 'drop', 'buildup', 'peak', 'harmonic'
    description: str


@dataclass
class ClipMarker:
    """A suggested clip extraction."""
    start_time: float
    end_time: float
    score: float
    moments: List[Moment]  # Moments contained in this clip


class MomentDetector:
    """
    Detects interesting moments in rendered video.

    Analyzes frame-by-frame data to identify peaks in musical
    and visual interest for optimal clip extraction.
    """

    def __init__(self):
        """Initialize moment detector."""
        self.frames: List[FrameData] = []
        self.moments: List[Moment] = []

    def record_frame(
        self,
        time: float,
        harmonic_frame,
        particle_count: int,
        visual_complexity: float
    ):
        """
        Record data for a single frame.

        Args:
            time: Frame time (seconds)
            harmonic_frame: HarmonicFrame from analyzer
            particle_count: Number of active particles
            visual_complexity: Visual complexity metric [0, 1]
        """
        frame_data = FrameData(
            time=time,
            beat_strength=harmonic_frame.beat_strength,
            beat_attack=harmonic_frame.beat_attack,
            spectral_flux=harmonic_frame.spectral_flux,
            harmonicity=harmonic_frame.harmonicity,
            particle_count=particle_count,
            visual_complexity=visual_complexity
        )
        self.frames.append(frame_data)

    def detect_moments(self) -> List[Moment]:
        """
        Analyze recorded frames and detect interesting moments.

        Returns:
            List of detected moments
        """
        if len(self.frames) < 30:
            return []

        self.moments = []

        # Convert to numpy arrays for analysis
        times = np.array([f.time for f in self.frames])
        beat_strengths = np.array([f.beat_strength for f in self.frames])
        spectral_fluxes = np.array([f.spectral_flux for f in self.frames])
        harmonicities = np.array([f.harmonicity for f in self.frames])
        visual_complexities = np.array([f.visual_complexity for f in self.frames])

        # =====================================================================
        # 1. DETECT DROPS (sudden energy spikes)
        # =====================================================================
        drops = self._detect_drops(times, beat_strengths, spectral_fluxes)
        self.moments.extend(drops)

        # =====================================================================
        # 2. DETECT BUILD-UPS (gradual energy increases)
        # =====================================================================
        buildups = self._detect_buildups(times, visual_complexities)
        self.moments.extend(buildups)

        # =====================================================================
        # 3. DETECT VISUAL COMPLEXITY PEAKS
        # =====================================================================
        peaks = self._detect_peaks(times, visual_complexities, "visual")
        self.moments.extend(peaks)

        # =====================================================================
        # 4. DETECT HARMONIC RICHNESS PEAKS
        # =====================================================================
        harmonic_peaks = self._detect_peaks(times, harmonicities, "harmonic")
        self.moments.extend(harmonic_peaks)

        # Sort by time
        self.moments.sort(key=lambda m: m.time)

        return self.moments

    def _detect_drops(
        self,
        times: np.ndarray,
        beat_strengths: np.ndarray,
        spectral_fluxes: np.ndarray
    ) -> List[Moment]:
        """Detect musical drops (sudden energy spikes)."""
        moments = []

        # Combine beat strength and spectral flux
        energy = beat_strengths * 0.5 + spectral_fluxes * 0.5

        # Find peaks with threshold
        threshold = 0.7
        for i in range(1, len(energy) - 1):
            if (energy[i] > threshold and
                energy[i] > energy[i-1] and
                energy[i] > energy[i+1]):

                # Calculate score based on peak height
                score = float(energy[i])

                moment = Moment(
                    time=times[i],
                    duration=0.5,  # 500ms around drop
                    score=score,
                    moment_type='drop',
                    description=f'Musical drop (energy={score:.2f})'
                )
                moments.append(moment)

        return moments

    def _detect_buildups(
        self,
        times: np.ndarray,
        visual_complexities: np.ndarray,
        window_size: int = 30
    ) -> List[Moment]:
        """Detect build-ups (gradual energy increases)."""
        moments = []

        for i in range(window_size, len(times) - window_size):
            # Check if complexity is increasing
            before = visual_complexities[i-window_size:i]
            after = visual_complexities[i:i+window_size]

            before_avg = before.mean()
            after_avg = after.mean()

            # Build-up if significant increase
            if after_avg > before_avg * 1.3:
                # Calculate slope (rate of increase)
                slope = (after_avg - before_avg) / window_size

                # Score based on slope
                score = float(min(slope * 10, 1.0))

                moment = Moment(
                    time=times[i],
                    duration=1.0,  # 1 second build-up
                    score=score,
                    moment_type='buildup',
                    description=f'Build-up (slope={slope:.3f})'
                )
                moments.append(moment)

                # Skip ahead to avoid overlapping detections
                i += window_size

        return moments

    def _detect_peaks(
        self,
        times: np.ndarray,
        values: np.ndarray,
        peak_type: str
    ) -> List[Moment]:
        """Detect peaks in a signal."""
        moments = []

        # Find peaks above 75th percentile
        threshold = np.percentile(values, 75)

        for i in range(1, len(values) - 1):
            if (values[i] > threshold and
                values[i] >= values[i-1] and
                values[i] >= values[i+1]):

                # Normalize score to [0, 1]
                score = float((values[i] - values.min()) / (values.max() - values.min() + 1e-6))

                moment = Moment(
                    time=times[i],
                    duration=0.3,
                    score=score,
                    moment_type=peak_type,
                    description=f'{peak_type.capitalize()} peak (score={score:.2f})'
                )
                moments.append(moment)

        return moments

    def get_clip_markers(
        self,
        duration: float = 15.0,
        count: int = 5,
        min_spacing: float = 5.0
    ) -> List[ClipMarker]:
        """
        Generate clip markers for best moments.

        Args:
            duration: Clip duration (seconds)
            count: Number of clips to generate
            min_spacing: Minimum time between clips (seconds)

        Returns:
            List of clip markers
        """
        if not self.moments:
            self.detect_moments()

        if not self.moments:
            return []

        # Score potential clip windows
        clip_candidates = []
        total_duration = self.frames[-1].time if self.frames else 0

        # Slide window across timeline
        step = duration / 2  # 50% overlap
        current_time = 0.0

        while current_time + duration <= total_duration:
            clip_end = current_time + duration

            # Find moments in this window
            window_moments = [
                m for m in self.moments
                if current_time <= m.time <= clip_end
            ]

            # Score this window
            if window_moments:
                # Average score of moments in window
                score = sum(m.score for m in window_moments) / len(window_moments)

                # Bonus for multiple moments
                score *= (1.0 + len(window_moments) * 0.1)

                clip_candidates.append(ClipMarker(
                    start_time=current_time,
                    end_time=clip_end,
                    score=score,
                    moments=window_moments
                ))

            current_time += step

        # Sort by score
        clip_candidates.sort(key=lambda c: c.score, reverse=True)

        # Select top clips with minimum spacing
        selected_clips = []
        for clip in clip_candidates:
            # Check spacing from already selected clips
            too_close = False
            for selected in selected_clips:
                if abs(clip.start_time - selected.start_time) < min_spacing:
                    too_close = True
                    break

            if not too_close:
                selected_clips.append(clip)

            if len(selected_clips) >= count:
                break

        # Sort by time
        selected_clips.sort(key=lambda c: c.start_time)

        return selected_clips

    def get_stats(self) -> dict:
        """Get detection statistics."""
        if not self.moments:
            return {}

        moment_types = {}
        for moment in self.moments:
            moment_types[moment.moment_type] = moment_types.get(moment.moment_type, 0) + 1

        return {
            'total_frames': len(self.frames),
            'total_moments': len(self.moments),
            'moments_by_type': moment_types,
            'avg_moment_score': np.mean([m.score for m in self.moments]) if self.moments else 0.0,
        }
