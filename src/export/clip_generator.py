"""
Clip Export and Multi-Variation System for V3

Generates multiple clips from single render for A/B testing.

Features:
- Auto-clip extraction from full render
- Multi-variation rendering (different camera seeds)
- Batch export for social media platforms
- FFmpeg integration for clip extraction

Usage:
    generator = ClipGenerator(full_video_path="output/full.mp4")

    # Extract clips from moment markers
    generator.extract_clips(
        clip_markers=markers,
        output_dir="output/clips"
    )

    # Generate variations with different seeds
    generator.generate_variations(
        base_config=config,
        num_variations=3,
        output_dir="output/variations"
    )
"""

import subprocess
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ClipConfig:
    """Configuration for clip extraction."""
    start_time: float
    end_time: float
    output_path: str
    preset: Optional[str] = None  # Social media preset


class ClipGenerator:
    """
    Generates clips from rendered video.

    Uses FFmpeg for fast, high-quality clip extraction.
    """

    def __init__(self, full_video_path: str):
        """
        Initialize clip generator.

        Args:
            full_video_path: Path to full rendered video
        """
        self.full_video_path = full_video_path

        if not os.path.exists(full_video_path):
            raise FileNotFoundError(f"Video not found: {full_video_path}")

    def extract_clip(
        self,
        start_time: float,
        end_time: float,
        output_path: str,
        preset: Optional[str] = None
    ) -> bool:
        """
        Extract a single clip from full video.

        Args:
            start_time: Clip start time (seconds)
            end_time: Clip end time (seconds)
            output_path: Output file path
            preset: Optional social media preset

        Returns:
            True if successful
        """
        duration = end_time - start_time

        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-ss', str(start_time),  # Start time
            '-i', self.full_video_path,  # Input file
            '-t', str(duration),  # Duration
            '-c:v', 'libx264',  # Video codec
            '-preset', 'fast',  # Encoding speed
            '-crf', '23',  # Quality (lower = better)
        ]

        # Add preset-specific settings
        if preset:
            from presets.social_media import SocialMediaPresets
            preset_obj = SocialMediaPresets.get_preset(preset)
            if preset_obj:
                # Add resolution scaling if needed
                cmd.extend([
                    '-vf', f'scale={preset_obj.resolution[0]}:{preset_obj.resolution[1]}',
                    '-b:v', preset_obj.bitrate
                ])

        cmd.append(output_path)

        try:
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"Clip extraction timed out")
            return False
        except Exception as e:
            print(f"Error extracting clip: {e}")
            return False

    def extract_clips(
        self,
        clip_markers,
        output_dir: str,
        name_prefix: str = "clip"
    ) -> List[str]:
        """
        Extract multiple clips from markers.

        Args:
            clip_markers: List of ClipMarker objects
            output_dir: Output directory
            name_prefix: Prefix for clip filenames

        Returns:
            List of generated clip paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        generated_clips = []

        for i, marker in enumerate(clip_markers):
            # Generate filename
            output_path = os.path.join(
                output_dir,
                f"{name_prefix}_{i+1:02d}.mp4"
            )

            # Extract clip
            success = self.extract_clip(
                start_time=marker.start_time,
                end_time=marker.end_time,
                output_path=output_path
            )

            if success:
                generated_clips.append(output_path)
                print(f"✅ Generated: {output_path}")
            else:
                print(f"❌ Failed: {output_path}")

        return generated_clips

    def generate_social_variations(
        self,
        clip_markers,
        output_dir: str,
        platforms: List[str]
    ) -> dict:
        """
        Generate clips optimized for multiple platforms.

        Args:
            clip_markers: List of ClipMarker objects
            output_dir: Output directory
            platforms: List of platform names ('instagram', 'tiktok', etc.)

        Returns:
            Dict mapping platform -> list of clip paths
        """
        results = {}

        for platform in platforms:
            platform_dir = os.path.join(output_dir, platform)
            os.makedirs(platform_dir, exist_ok=True)

            clips = []
            for i, marker in enumerate(clip_markers):
                output_path = os.path.join(
                    platform_dir,
                    f"{platform}_clip_{i+1:02d}.mp4"
                )

                success = self.extract_clip(
                    start_time=marker.start_time,
                    end_time=marker.end_time,
                    output_path=output_path,
                    preset=platform
                )

                if success:
                    clips.append(output_path)

            results[platform] = clips
            print(f"✅ {platform}: {len(clips)} clips generated")

        return results


class BatchExporter:
    """Batch export utility for multiple renders."""

    @staticmethod
    def export_with_variations(
        base_render_script: str,
        num_variations: int,
        output_dir: str,
        camera_seeds: Optional[List[int]] = None
    ) -> List[str]:
        """
        Render multiple variations with different camera seeds.

        Args:
            base_render_script: Path to render script
            num_variations: Number of variations to generate
            output_dir: Output directory
            camera_seeds: Optional list of random seeds

        Returns:
            List of generated video paths
        """
        if camera_seeds is None:
            camera_seeds = list(range(num_variations))

        os.makedirs(output_dir, exist_ok=True)
        rendered_videos = []

        for i, seed in enumerate(camera_seeds[:num_variations]):
            output_path = os.path.join(output_dir, f"variation_{i+1:02d}.mp4")

            print(f"Rendering variation {i+1}/{num_variations} (seed={seed})...")

            # Run render script with seed parameter
            cmd = [
                'python3',
                base_render_script,
                '--camera-seed', str(seed),
                '-o', output_path,
                '-r', '720p',  # Fast preview quality
                '-q', 'draft'
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )

                if result.returncode == 0:
                    rendered_videos.append(output_path)
                    print(f"✅ Variation {i+1} complete")
                else:
                    print(f"❌ Variation {i+1} failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"❌ Variation {i+1} timed out")
            except Exception as e:
                print(f"❌ Variation {i+1} error: {e}")

        return rendered_videos
