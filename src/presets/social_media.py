"""
Social Media Presets for V3

Pre-configured settings for different platforms optimized for viral content.

Platforms:
- Instagram Reels: 1080x1920 (9:16), 15-30s
- TikTok: 1080x1920 (9:16), 15-60s
- YouTube Shorts: 1080x1920 (9:16), 60s max
- Twitter: 1280x720 (16:9), 30s

Each preset includes:
- Resolution and aspect ratio
- Duration limits
- Safe zones for text/UI
- Encoding settings
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SocialMediaPreset:
    """Social media platform preset."""
    name: str
    resolution: Tuple[int, int]  # (width, height)
    aspect_ratio: str
    min_duration: float  # seconds
    max_duration: float
    fps: int

    # Safe zones (percentage from edges)
    safe_zone_top: float
    safe_zone_bottom: float
    safe_zone_left: float
    safe_zone_right: float

    # Encoding
    bitrate: str
    codec: str


class SocialMediaPresets:
    """Collection of social media platform presets."""

    INSTAGRAM_REEL = SocialMediaPreset(
        name="Instagram Reel",
        resolution=(1080, 1920),
        aspect_ratio="9:16",
        min_duration=15.0,
        max_duration=30.0,
        fps=30,
        safe_zone_top=0.12,  # 12% from top (for app UI)
        safe_zone_bottom=0.15,  # 15% from bottom (for controls)
        safe_zone_left=0.05,
        safe_zone_right=0.05,
        bitrate="8M",
        codec="libx264"
    )

    TIKTOK = SocialMediaPreset(
        name="TikTok",
        resolution=(1080, 1920),
        aspect_ratio="9:16",
        min_duration=15.0,
        max_duration=60.0,
        fps=30,
        safe_zone_top=0.15,
        safe_zone_bottom=0.20,  # Large bottom safe zone
        safe_zone_left=0.05,
        safe_zone_right=0.05,
        bitrate="8M",
        codec="libx264"
    )

    YOUTUBE_SHORT = SocialMediaPreset(
        name="YouTube Short",
        resolution=(1080, 1920),
        aspect_ratio="9:16",
        min_duration=15.0,
        max_duration=60.0,
        fps=30,
        safe_zone_top=0.10,
        safe_zone_bottom=0.12,
        safe_zone_left=0.05,
        safe_zone_right=0.05,
        bitrate="10M",  # Higher bitrate for YouTube
        codec="libx264"
    )

    TWITTER = SocialMediaPreset(
        name="Twitter",
        resolution=(1280, 720),
        aspect_ratio="16:9",
        min_duration=5.0,
        max_duration=30.0,
        fps=30,
        safe_zone_top=0.08,
        safe_zone_bottom=0.08,
        safe_zone_left=0.08,
        safe_zone_right=0.08,
        bitrate="6M",
        codec="libx264"
    )

    @classmethod
    def get_preset(cls, name: str) -> SocialMediaPreset:
        """Get preset by name."""
        presets = {
            'instagram': cls.INSTAGRAM_REEL,
            'tiktok': cls.TIKTOK,
            'youtube': cls.YOUTUBE_SHORT,
            'twitter': cls.TWITTER,
        }
        return presets.get(name.lower())

    @classmethod
    def list_presets(cls) -> list:
        """List all available presets."""
        return ['instagram', 'tiktok', 'youtube', 'twitter']
