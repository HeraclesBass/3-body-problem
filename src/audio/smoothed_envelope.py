"""
Smoothed Audio Envelope System

Prevents jittery audio reactions by smoothing audio parameters with
attack/release curves. Creates musical, natural-feeling responses.

Design Philosophy:
- Attack fast (immediate response to onsets)
- Release slow (gentle decay, no abrupt stops)
- Musical timing (feels natural)
- Prevents visual flickering
"""

import numpy as np


class SmoothedAudioEnvelope:
    """
    Exponential smoothing for audio parameters with attack/release.

    Similar to envelope generators in synthesizers:
    - Attack: How quickly value rises to target
    - Release: How quickly value falls from target
    """

    def __init__(self, attack_time=0.05, release_time=0.2, fps=30):
        """
        Initialize envelope with timing parameters.

        Args:
            attack_time: Rise time in seconds (typical: 0.01-0.1)
            release_time: Fall time in seconds (typical: 0.1-0.5)
            fps: Frame rate for time calculations

        Timing Guidelines:
        - Very fast attack (0.01s): Instant, snappy
        - Fast attack (0.05s): Quick, responsive
        - Medium attack (0.1s): Smooth, musical
        - Slow attack (0.2s): Gentle, ambient

        - Fast release (0.1s): Tight, rhythmic
        - Medium release (0.2-0.3s): Natural, balanced
        - Slow release (0.5-1.0s): Ambient, lingering
        """
        self.value = 0.0  # Current envelope value

        # Convert times to exponential coefficients
        # Formula: 1 - exp(-1 / (time * fps))
        # This gives ~63% response per time constant
        self.attack_coeff = 1.0 - np.exp(-1.0 / (attack_time * fps))
        self.release_coeff = 1.0 - np.exp(-1.0 / (release_time * fps))

    def update(self, target):
        """
        Update envelope toward target value.

        Args:
            target: Target value (0-1 typical)

        Returns:
            Current smoothed value

        Behavior:
        - If target > current: Use attack coefficient (fast rise)
        - If target < current: Use release coefficient (slow fall)
        """
        if target > self.value:
            # Attack (rising)
            self.value += (target - self.value) * self.attack_coeff
        else:
            # Release (falling)
            self.value += (target - self.value) * self.release_coeff

        return self.value

    def reset(self, value=0.0):
        """Reset envelope to specific value."""
        self.value = value


class MultiEnvelopeController:
    """
    Manages multiple smoothed envelopes for different audio parameters.

    Convenience class for handling all audio-reactive parameters.
    """

    def __init__(self, fps=30):
        """
        Initialize with standard envelope configurations.

        Args:
            fps: Frame rate
        """
        # Beat envelope: Medium attack, long release (pulsing feel)
        self.beat_envelope = SmoothedAudioEnvelope(
            attack_time=0.05,
            release_time=0.3,
            fps=fps
        )

        # Bass envelope: Fast attack, medium release (punchy)
        self.bass_envelope = SmoothedAudioEnvelope(
            attack_time=0.03,
            release_time=0.2,
            fps=fps
        )

        # Mid envelope: Medium attack/release (balanced)
        self.mid_envelope = SmoothedAudioEnvelope(
            attack_time=0.08,
            release_time=0.25,
            fps=fps
        )

        # Treble envelope: Very fast attack, medium release (crisp)
        self.treble_envelope = SmoothedAudioEnvelope(
            attack_time=0.02,
            release_time=0.15,
            fps=fps
        )

        # Presence envelope: Fast attack, fast release (responsive)
        self.presence_envelope = SmoothedAudioEnvelope(
            attack_time=0.04,
            release_time=0.15,
            fps=fps
        )

        # Brilliance envelope: Very fast (sparkles)
        self.brilliance_envelope = SmoothedAudioEnvelope(
            attack_time=0.01,
            release_time=0.1,
            fps=fps
        )

    def update(self, audio_frame):
        """
        Update all envelopes from audio frame.

        Args:
            audio_frame: AudioFrame10Band object

        Returns:
            Dictionary of smoothed values
        """
        smoothed = {
            'beat': self.beat_envelope.update(audio_frame.beat_strength),
            'bass': self.bass_envelope.update(
                (audio_frame.sub_bass + audio_frame.bass) / 2
            ),
            'mid': self.mid_envelope.update(
                (audio_frame.mid + audio_frame.high_mid) / 2
            ),
            'treble': self.treble_envelope.update(
                (audio_frame.brilliance + audio_frame.air) / 2
            ),
            'presence': self.presence_envelope.update(audio_frame.presence),
            'brilliance': self.brilliance_envelope.update(audio_frame.brilliance),
        }

        return smoothed


class RMSEnergySmooth:
    """
    RMS energy smoother for overall audio intensity.

    Uses running average for very smooth energy measurements.
    """

    def __init__(self, window_size=10):
        """
        Initialize RMS smoother.

        Args:
            window_size: Number of frames to average (typical: 5-20)
        """
        self.window_size = window_size
        self.history = []

    def update(self, energy):
        """
        Update with new energy value.

        Args:
            energy: Current frame energy

        Returns:
            Smoothed RMS energy
        """
        self.history.append(energy)

        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # RMS calculation
        rms = np.sqrt(np.mean(np.array(self.history) ** 2))

        return rms


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == '__main__':
    """
    Example usage of envelope system.
    """
    # Create controller
    controller = MultiEnvelopeController(fps=30)

    # Simulate audio frames
    class MockAudioFrame:
        def __init__(self, beat=0, bass=0):
            self.beat_strength = beat
            self.sub_bass = bass
            self.bass = bass
            self.mid = 0.5
            self.high_mid = 0.5
            self.presence = 0.3
            self.brilliance = 0.4
            self.air = 0.3

    print("Testing envelope smoothing...")
    print("Frame | Beat Target | Beat Smooth | Bass Target | Bass Smooth")
    print("-" * 65)

    # Simulate beat pulse
    for i in range(30):
        # Beat pulse at frame 5
        beat = 1.0 if i == 5 else 0.0
        bass = 0.8 if i < 10 else 0.2

        audio = MockAudioFrame(beat=beat, bass=bass)
        smoothed = controller.update(audio)

        print(f"{i:5d} | {beat:11.3f} | {smoothed['beat']:11.3f} | "
              f"{bass:11.3f} | {smoothed['bass']:11.3f}")

    print("\nNotice:")
    print("- Beat: Fast attack (frame 5), slow release (frames 6-15)")
    print("- Bass: Fast attack (frames 0-2), slow release (frames 10-20)")
    print("- No jitter, smooth transitions!")
