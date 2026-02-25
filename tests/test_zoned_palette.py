"""
Unit tests for Zoned Palette System

Tests:
- Zone color assignment
- Color transitions with audio
- Particle color variation
- RGB output validity
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from audio.frequency_zones import FrequencyZones
from audio.analyzer_10band import AudioFrame10Band
from rendering.zoned_palette import ZonedPalette, PalettePreview


def test_palette_initialization():
    """Test palette initializes correctly."""
    zones = FrequencyZones(num_particles=50, seed=42)
    palette = ZonedPalette(zones)

    # Should have initialized arrays
    assert len(palette.zone_hue_offsets) == 10
    assert len(palette.zone_hue_targets) == 10
    assert len(palette.particle_offsets) == 50

    # Particle offsets should be in valid range
    assert np.all(palette.particle_offsets >= -palette.particle_hue_variation)
    assert np.all(palette.particle_offsets <= palette.particle_hue_variation)

    print("✅ Palette initialization works")


def test_zone_colors():
    """Test that each zone has a valid color."""
    zones = FrequencyZones(num_particles=50, seed=42)
    palette = ZonedPalette(zones)

    # Create audio frame with energy
    audio_frame = AudioFrame10Band(
        sub_bass=0.5, bass=0.5, low_mid=0.5, mid=0.5, high_mid=0.5,
        presence=0.5, brilliance=0.5, air=0.5, ultra=0.5, extreme=0.5,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    palette.update(audio_frame)

    # Test each zone has valid RGB color
    for zone_id in range(10):
        r, g, b = palette.get_zone_color(zone_id)

        # RGB should be in [0, 1]
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    print("✅ Zone colors are valid")


def test_particle_colors():
    """Test particle color retrieval."""
    zones = FrequencyZones(num_particles=50, seed=42)
    palette = ZonedPalette(zones)

    # Update with audio
    audio_frame = AudioFrame10Band(
        sub_bass=0.8, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    palette.update(audio_frame)

    # Test getting color for each particle
    for particle_idx in range(50):
        r, g, b = palette.get_particle_color(particle_idx)

        # RGB should be in [0, 1]
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    print("✅ Particle colors are valid")


def test_particle_color_variation():
    """Test that particles in same zone have slightly different colors."""
    zones = FrequencyZones(num_particles=100, seed=42)
    palette = ZonedPalette(zones)

    # Update with audio
    audio_frame = AudioFrame10Band(
        sub_bass=0.5, bass=0.5, low_mid=0.5, mid=0.5, high_mid=0.5,
        presence=0.5, brilliance=0.5, air=0.5, ultra=0.5, extreme=0.5,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    palette.update(audio_frame)

    # Find two particles in the same zone
    zone_0_particles = np.where(zones.particle_zones == 0)[0]
    if len(zone_0_particles) >= 2:
        p1 = zone_0_particles[0]
        p2 = zone_0_particles[1]

        color1 = palette.get_particle_color(p1)
        color2 = palette.get_particle_color(p2)

        # Colors should be different (due to particle offsets)
        assert color1 != color2

    print("✅ Particles in same zone have variation")


def test_color_batch_retrieval():
    """Test efficient batch color retrieval."""
    zones = FrequencyZones(num_particles=50, seed=42)
    palette = ZonedPalette(zones)

    # Update with audio
    audio_frame = AudioFrame10Band(
        sub_bass=0.5, bass=0.5, low_mid=0.5, mid=0.5, high_mid=0.5,
        presence=0.5, brilliance=0.5, air=0.5, ultra=0.5, extreme=0.5,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    palette.update(audio_frame)

    # Get colors for all particles
    particle_indices = np.arange(50)
    colors = palette.get_particle_color_array(particle_indices)

    # Should return correct shape
    assert colors.shape == (50, 3)

    # All values should be in [0, 1]
    assert np.all(colors >= 0.0)
    assert np.all(colors <= 1.0)

    print("✅ Batch color retrieval works")


def test_energy_affects_color():
    """Test that zone energy affects color brightness."""
    zones = FrequencyZones(num_particles=50, seed=42)
    palette = ZonedPalette(zones)

    # Low energy frame
    frame_low = AudioFrame10Band(
        sub_bass=0.1, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    palette.update(frame_low)

    # Get particle in zone 0
    zone_0_particles = np.where(zones.particle_zones == 0)[0]
    if len(zone_0_particles) > 0:
        p = zone_0_particles[0]
        r_low, g_low, b_low = palette.get_particle_color(p)
        brightness_low = (r_low + g_low + b_low) / 3

        # High energy frame
        frame_high = AudioFrame10Band(
            sub_bass=1.0, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
            presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
            beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
        )
        palette.update(frame_high)

        r_high, g_high, b_high = palette.get_particle_color(p)
        brightness_high = (r_high + g_high + b_high) / 3

        # Higher energy should give brighter color
        assert brightness_high > brightness_low

    print("✅ Energy affects color brightness")


def test_smooth_transitions():
    """Test that colors transition smoothly over multiple frames."""
    zones = FrequencyZones(num_particles=50, seed=42)
    palette = ZonedPalette(zones)

    # Frame 1: Bass heavy
    frame1 = AudioFrame10Band(
        sub_bass=1.0, bass=1.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )

    # Frame 2: Treble heavy
    frame2 = AudioFrame10Band(
        sub_bass=0.0, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=1.0, air=1.0, ultra=1.0, extreme=1.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.1
    )

    # Update with frame 1
    palette.update(frame1)
    shift_after_1 = palette.global_hue_shift

    # Update with frame 2 (should transition, not jump)
    palette.update(frame2)
    shift_after_2 = palette.global_hue_shift

    # Shift should change but not drastically (smooth transition)
    shift_delta = abs(shift_after_2 - shift_after_1)
    assert shift_delta < 50.0  # Less than 50° change per frame

    print("✅ Smooth color transitions work")


def test_palette_summary():
    """Test palette summary generation."""
    zones = FrequencyZones(num_particles=50, seed=42)
    palette = ZonedPalette(zones)

    summary = palette.get_palette_summary()

    # Should have expected keys
    assert 'global_hue_shift' in summary
    assert 'zone_hue_offsets' in summary
    assert 'zone_energies' in summary

    # Offsets should be 10 values
    assert len(summary['zone_hue_offsets']) == 10
    assert len(summary['zone_energies']) == 10

    print("✅ Palette summary works")


def test_preview_image_generation():
    """Test preview image generation."""
    zones = FrequencyZones(num_particles=50, seed=42)
    palette = ZonedPalette(zones)

    # Update with audio
    audio_frame = AudioFrame10Band(
        sub_bass=0.5, bass=0.5, low_mid=0.5, mid=0.5, high_mid=0.5,
        presence=0.5, brilliance=0.5, air=0.5, ultra=0.5, extreme=0.5,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    palette.update(audio_frame)

    # Generate preview
    preview = PalettePreview.generate_preview_image(palette, width=100, height=10)

    # Check shape
    assert preview.shape == (10, 100, 3)

    # Check dtype
    assert preview.dtype == np.uint8

    # Check values in [0, 255]
    assert np.all(preview >= 0)
    assert np.all(preview <= 255)

    print("✅ Preview image generation works")


if __name__ == "__main__":
    print("Running Zoned Palette Tests...\n")

    test_palette_initialization()
    test_zone_colors()
    test_particle_colors()
    test_particle_color_variation()
    test_color_batch_retrieval()
    test_energy_affects_color()
    test_smooth_transitions()
    test_palette_summary()
    test_preview_image_generation()

    print("\n✅ All tests passed!")
