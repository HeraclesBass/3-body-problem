"""
Fast HLS streaming with PIL rendering (not matplotlib).

PIL is 10-20x faster than matplotlib for simple 2D graphics.
Combined with x264 ultrafast, this should achieve real-time.
"""

import asyncio
import subprocess
import os
import sys
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from physics.nbody import NBodySimulation, SimulationConfig
from audio.analyzer import AudioAnalyzer


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class FastHLSStreamer:
    """
    Real-time HLS streaming using PIL for fast rendering.
    """

    def __init__(
        self,
        audio_path: str,
        output_dir: str = "/tmp/hls",
        n_bodies: int = 3,
        resolution: tuple = (1280, 720),
        fps: float = 30.0,
        bitrate: str = "4M",
        use_gpu: bool = True
    ):
        self.audio_path = audio_path
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.fps = fps
        self.bitrate = bitrate
        self.width, self.height = resolution
        self.running = False

        # Create/clean output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for f in self.output_dir.glob("*.ts"):
            f.unlink()
        for f in self.output_dir.glob("*.m3u8"):
            f.unlink()

        # Load audio
        print(f"Loading audio: {audio_path}")
        self.audio = AudioAnalyzer(audio_path, fps=fps)
        print(f"  Duration: {self.audio.duration:.1f}s, Frames: {self.audio.total_frames}")

        # Initialize Warp
        import warp as wp
        wp.init()
        device = "cuda:0" if (use_gpu and wp.is_cuda_available()) else "cpu"
        print(f"Physics device: {device}")

        # Initialize simulation
        self.config = SimulationConfig(
            n_bodies=n_bodies,
            G=1.0,
            softening=0.02,
            dt=0.0002,
            trail_length=500,
            device=device
        )
        self.sim = NBodySimulation(self.config)
        self._init_bodies()

        # Colors (RGB tuples)
        self.colors = [
            (0, 255, 255),    # Cyan
            (255, 107, 157),  # Pink
            (157, 78, 221),   # Purple
            (0, 255, 136),    # Green
            (255, 170, 0),    # Orange
            (255, 68, 68),    # Red
            (68, 170, 255),   # Blue
            (255, 255, 68),   # Yellow
        ]

        # Trail buffer
        self.trail_history = [[] for _ in range(n_bodies)]
        self.max_trail = 200

        self.ffmpeg_process = None
        self.frame_count = 0

    def _init_bodies(self):
        if self.config.n_bodies == 3:
            self.sim.initialize_figure_eight(scale=1.2)
        else:
            self.sim.initialize_random(pos_range=1.8, vel_range=0.25, seed=42)

    def _world_to_screen(self, x, y):
        """Convert world coordinates to screen pixels."""
        # World range: -3 to 3 for x, -1.7 to 1.7 for y (16:9)
        sx = int((x + 3) / 6 * self.width)
        sy = int((1.7 - y) / 3.4 * self.height)  # Flip y
        return sx, sy

    def _render_frame(self, audio_frame) -> bytes:
        """Render frame using PIL (fast)."""
        # Create black image
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        positions = self.sim.get_positions()

        # Update trail history
        for i in range(self.config.n_bodies):
            self.trail_history[i].append(positions[i].copy())
            if len(self.trail_history[i]) > self.max_trail:
                self.trail_history[i].pop(0)

        # Draw trails
        for i in range(self.config.n_bodies):
            trail = self.trail_history[i]
            if len(trail) > 2:
                color = self.colors[i % len(self.colors)]
                n = len(trail)

                # Draw trail segments with fading
                for j in range(1, n):
                    alpha = int((j / n) * 180 * (0.5 + audio_frame.mid_energy * 0.5))
                    x1, y1 = self._world_to_screen(trail[j-1][0], trail[j-1][1])
                    x2, y2 = self._world_to_screen(trail[j][0], trail[j][1])

                    # Fade color
                    faded = tuple(int(c * alpha / 255) for c in color)
                    draw.line([(x1, y1), (x2, y2)], fill=faded, width=2)

        # Draw bodies with glow
        glow_mult = 1.0 + audio_frame.beat_strength * 1.5

        for i in range(self.config.n_bodies):
            color = self.colors[i % len(self.colors)]
            x, y = positions[i, 0], positions[i, 1]
            sx, sy = self._world_to_screen(x, y)

            # Glow layers (larger, more transparent circles)
            for radius, alpha in [(int(25 * glow_mult), 30), (int(15 * glow_mult), 60), (int(8 * glow_mult), 120)]:
                glow_color = tuple(int(c * alpha / 255) for c in color)
                draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius],
                           fill=glow_color)

            # Core
            core_r = int(5 + audio_frame.beat_strength * 8)
            draw.ellipse([sx - core_r, sy - core_r, sx + core_r, sy + core_r],
                        fill=color)

        # Time overlay
        t = audio_frame.time
        time_str = f'{int(t//60)}:{int(t%60):02d}'
        draw.text((20, 20), time_str, fill=(255, 255, 255, 128))

        # Beat indicator
        if audio_frame.beat_strength > 0.5:
            draw.ellipse([self.width - 40, 15, self.width - 20, 35],
                        fill=(255, 68, 68))

        # Convert to RGB bytes
        return img.tobytes()

    def _start_ffmpeg(self):
        """Start ffmpeg with ultrafast encoding."""
        playlist_path = self.output_dir / "stream.m3u8"

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',
            '-i', self.audio_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # Fastest encoding
            '-tune', 'zerolatency',  # Low latency
            '-crf', '28',  # Lower quality = faster
            '-b:v', self.bitrate,
            '-c:a', 'aac',
            '-b:a', '128k',
            '-pix_fmt', 'yuv420p',
            '-g', str(int(self.fps)),  # Keyframe every second
            '-shortest',
            '-f', 'hls',
            '-hls_time', '4',
            '-hls_list_size', '20',
            '-hls_flags', 'delete_segments+append_list',
            '-hls_init_time', '4',
            '-hls_segment_filename', str(self.output_dir / 'seg_%03d.ts'),
            str(playlist_path)
        ]

        print("Starting ffmpeg (ultrafast x264)...")
        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=self.width * self.height * 3 * 4
        )

    def run(self):
        """Run the streaming loop."""
        self.running = True
        self._start_ffmpeg()

        steps_per_frame = 15
        total_frames = self.audio.total_frames
        target_frame_time = 1.0 / self.fps

        print(f"\nStreaming {total_frames} frames @ {self.fps} FPS")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Target: real-time ({self.fps} fps)\n")

        start_time = time.time()
        frame_times = []

        try:
            while self.running and self.frame_count < total_frames:
                frame_start = time.time()

                # Get audio
                audio_frame = self.audio.get_frame(self.frame_count)
                audio_params = {
                    'bass_energy': audio_frame.bass_energy,
                    'mid_energy': audio_frame.mid_energy,
                    'treble_energy': audio_frame.treble_energy,
                    'modulation_depth': 0.8 + audio_frame.beat_strength * 1.2
                }

                # Step physics
                for _ in range(steps_per_frame):
                    self.sim.step(audio_params=audio_params)

                # Render
                frame_data = self._render_frame(audio_frame)

                # Send to ffmpeg
                try:
                    self.ffmpeg_process.stdin.write(frame_data)
                except BrokenPipeError:
                    print("\nFFmpeg pipe broken!")
                    break

                self.frame_count += 1
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)

                # Progress every second
                if self.frame_count % int(self.fps) == 0:
                    elapsed = time.time() - start_time
                    avg_fps = self.frame_count / elapsed
                    recent_fps = 1.0 / (sum(frame_times[-30:]) / len(frame_times[-30:])) if frame_times else 0
                    pct = self.frame_count / total_frames * 100
                    status = "✓ REALTIME" if recent_fps >= self.fps * 0.95 else f"({recent_fps:.1f} fps)"
                    print(f"\r  {pct:.1f}% | {self.frame_count}/{total_frames} | {avg_fps:.1f} fps avg | {status}    ", end='', flush=True)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.stop()

        total_time = time.time() - start_time
        print(f"\n\nCompleted {self.frame_count} frames in {total_time:.1f}s ({self.frame_count/total_time:.1f} fps)")

    def stop(self):
        self.running = False
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=10)
            except:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None


# Simple HTTP handler
class HLSHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(VIEWER_HTML.encode())
            return
        super().do_GET()

    def log_message(self, format, *args):
        pass


VIEWER_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Celestial Chaos - Live</title>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <style>
        body { background: #000; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; font-family: monospace; color: #fff; }
        h1 { color: #0ff; text-shadow: 0 0 20px #0ff; margin-bottom: 10px; }
        video { max-width: 95vw; max-height: 80vh; border: 2px solid #333; border-radius: 10px; }
        .controls { margin-top: 15px; }
        button { background: #111; color: #0ff; border: 1px solid #0ff; padding: 10px 20px; margin: 5px; cursor: pointer; border-radius: 5px; }
        button:hover { background: #0ff; color: #000; }
        #status { color: #888; margin: 10px; }
    </style>
</head>
<body>
    <h1>CELESTIAL CHAOS</h1>
    <div id="status">Loading...</div>
    <video id="video" controls autoplay muted></video>
    <div class="controls">
        <button onclick="document.getElementById('video').muted=false">🔊 Unmute</button>
        <button onclick="location.reload()">↻ Refresh</button>
    </div>
    <script>
        const video = document.getElementById('video');
        const status = document.getElementById('status');
        function load() {
            if (Hls.isSupported()) {
                const hls = new Hls({liveSyncDurationCount:3, liveMaxLatencyDurationCount:6});
                hls.loadSource('/stream.m3u8');
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, () => { status.textContent = '● LIVE'; status.style.color = '#0f0'; video.play(); });
                hls.on(Hls.Events.ERROR, (e,d) => { if(d.fatal) { status.textContent = 'Buffering...'; setTimeout(load, 2000); }});
            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                video.src = '/stream.m3u8';
                video.addEventListener('loadedmetadata', () => { status.textContent = '● LIVE'; video.play(); });
            }
        }
        load();
    </script>
</body>
</html>'''


def serve_hls(directory, port):
    os.chdir(directory)
    handler = lambda *a, **k: HLSHandler(*a, directory=directory, **k)
    HTTPServer(('0.0.0.0', port), handler).serve_forever()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', '-a', default='assets/audio/still-night.mp3')
    parser.add_argument('--bodies', '-n', type=int, default=3)
    parser.add_argument('--resolution', '-r', default='720p', choices=['480p', '720p', '1080p'])
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--bitrate', '-b', default='3M')
    parser.add_argument('--http-port', type=int, default=8765)
    parser.add_argument('--output-dir', default='/tmp/hls')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    res = {'480p': (854, 480), '720p': (1280, 720), '1080p': (1920, 1080)}[args.resolution]

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║       CELESTIAL CHAOS - Fast Live Stream                  ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Resolution: {args.resolution} ({res[0]}x{res[1]})  FPS: {args.fps}  Bodies: {args.bodies}      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # HTTP server
    threading.Thread(target=serve_hls, args=(args.output_dir, args.http_port), daemon=True).start()
    print(f"Viewer: http://0.0.0.0:{args.http_port}")

    FastHLSStreamer(
        audio_path=args.audio,
        output_dir=args.output_dir,
        n_bodies=args.bodies,
        resolution=res,
        fps=args.fps,
        bitrate=args.bitrate,
        use_gpu=not args.cpu
    ).run()


if __name__ == '__main__':
    main()
