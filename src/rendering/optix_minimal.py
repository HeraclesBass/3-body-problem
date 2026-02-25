"""
Minimal OptiX test program - render a single glowing sphere.

This validates OptiX installation and demonstrates basic setup
before building the full Warp-OptiX bridge.
"""

import numpy as np
from PIL import Image

try:
    import pyoptix as optix
except ImportError:
    try:
        import optix  # Alternative package name
    except ImportError:
        print("ERROR: PyOptiX not installed!")
        print("Run: pip install pyoptix")
        exit(1)


class MinimalOptiXRenderer:
    """
    Minimal OptiX renderer for testing installation.

    Renders a single volumetric sphere to validate:
    - OptiX context creation
    - CUDA device detection
    - Basic ray tracing setup
    - Frame buffer export
    """

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        print("Initializing OptiX...")
        self.init_optix()
        print(f"✅ OptiX initialized successfully!")

    def init_optix(self):
        """Initialize OptiX context and detect devices."""
        # Create OptiX device context
        self.ctx = optix.DeviceContext()

        # Get device info
        device_count = self.ctx.get_device_count()
        print(f"Found {device_count} CUDA device(s)")

        for i in range(device_count):
            name = self.ctx.get_device_name(i)
            print(f"  Device {i}: {name}")

    def render_sphere(self):
        """
        Render a simple glowing sphere.

        For now, this is a placeholder that generates a test pattern.
        Full OptiX ray tracing will be added next.
        """
        print(f"Rendering {self.width}x{self.height} frame...")

        # Generate test pattern (will be replaced with OptiX rendering)
        img = self._generate_test_pattern()

        return img

    def _generate_test_pattern(self):
        """
        Generate test pattern to verify image export works.

        TODO: Replace with actual OptiX ray tracing.
        """
        # Create image array
        img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw gradient sphere
        center_x, center_y = self.width // 2, self.height // 2
        radius = min(self.width, self.height) // 3

        for y in range(self.height):
            for x in range(self.width):
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx*dx + dy*dy)

                if dist < radius:
                    # Inside sphere - glow falloff
                    intensity = 1.0 - (dist / radius)
                    intensity = intensity ** 2  # Quadratic falloff

                    # Cyan color (bioluminescent)
                    img_array[y, x, 0] = 0  # R
                    img_array[y, x, 1] = int(255 * intensity)  # G
                    img_array[y, x, 2] = int(255 * intensity)  # B

        return img_array

    def save_frame(self, img_array, filename):
        """Save rendered frame to PNG."""
        img = Image.fromarray(img_array, mode='RGB')
        img.save(filename)
        print(f"✅ Saved: {filename}")


def main():
    """Test OptiX installation."""
    print("=" * 60)
    print("OptiX Minimal Test")
    print("=" * 60)

    # Check OptiX availability
    print("\n1. Checking OptiX availability...")
    if not optix.is_available():
        print("❌ OptiX not available!")
        print("   Check NVIDIA driver and OptiX SDK installation")
        return 1
    print("✅ OptiX is available")

    # Create renderer
    print("\n2. Creating OptiX context...")
    try:
        renderer = MinimalOptiXRenderer(width=1280, height=720)
    except Exception as e:
        print(f"❌ Failed to create OptiX context: {e}")
        return 1

    # Render test frame
    print("\n3. Rendering test frame...")
    try:
        img = renderer.render_sphere()
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        return 1

    # Save output
    print("\n4. Saving output...")
    try:
        renderer.save_frame(img, "optix_test.png")
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return 1

    print("\n" + "=" * 60)
    print("✅ SUCCESS! OptiX is working correctly.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. View optix_test.png to see test pattern")
    print("  2. Implement actual OptiX ray tracing")
    print("  3. Add Warp physics integration")

    return 0


if __name__ == '__main__':
    exit(main())
