"""
HLS Live Streaming Server for Celestial Chaos.

Uses ffmpeg + NVENC for GPU-accelerated encoding.
Outputs adaptive bitrate HLS stream viewable in any browser.
"""

import asyncio
import subprocess
import os
import sys
import time
import signal
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from physics.nbody import NBodySimulation, SimulationConfig
from audio.analyzer import AudioAnalyzer


class HLSStreamer:
    """
    High-quality HLS streaming with NVENC encoding.

    Pipeline:
    Simulation → Matplotlib render → Raw frames → ffmpeg (NVENC) → HLS segments
    """

    def __init__(
        self,
        audio_path: str,
        output_dir: str = "/tmp/hls",
        n_bodies: int = 3,
        resolution: tuple = (1920, 1080),
        fps: float = 30.0,
        bitrate: str = "8M",
        use_gpu: bool = True
    ):
        self.audio_path = audio_path
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.fps = fps
        self.bitrate = bitrate
        self.running = False

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clean old segments
        for f in self.output_dir.glob("*.ts"):
            f.unlink()
        for f in self.output_dir.glob("*.m3u8"):
            f.unlink()

        # Load audio
        print(f"Loading audio: {audio_path}")
        self.audio = AudioAnalyzer(audio_path, fps=fps)
        print(f"  Duration: {self.audio.duration:.1f}s")
        print(f"  Frames: {self.audio.total_frames}")

        # Initialize Warp
        import warp as wp
        wp.init()
        device = "cuda:0" if (use_gpu and wp.is_cuda_available()) else "cpu"
        print(f"Compute device: {device}")

        # Initialize simulation
        self.config = SimulationConfig(
            n_bodies=n_bodies,
            G=1.0,
            softening=0.02,
            dt=0.0002,
            trail_length=1000,
            device=device
        )
        self.sim = NBodySimulation(self.config)
        self._init_bodies()

        # Setup matplotlib for high quality rendering
        dpi = 100
        fig_w = resolution[0] / dpi
        fig_h = resolution[1] / dpi
        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, facecolor='black')
        self.ax.set_facecolor('black')

        # Colors
        self.colors = [
            '#00ffff', '#ff6b9d', '#9d4edd', '#00ff88',
            '#ffaa00', '#ff4444', '#44aaff', '#ffff44',
            '#ff00ff', '#00ffaa', '#ff8844', '#88ff44'
        ]

        self.ffmpeg_process = None
        self.frame_count = 0

    def _init_bodies(self):
        if self.config.n_bodies == 3:
            self.sim.initialize_figure_eight(scale=1.2)
        else:
            self.sim.initialize_random(pos_range=1.8, vel_range=0.25, seed=42)

    def _render_frame(self, audio_frame) -> bytes:
        """Render frame to raw RGB bytes."""
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-1.7, 1.7)  # 16:9 aspect
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        positions = self.sim.get_positions()

        # Background glow on bass
        if audio_frame.bass_energy > 0.3:
            circle = plt.Circle((0, 0), 3, color='#001133',
                               alpha=audio_frame.bass_energy * 0.3)
            self.ax.add_patch(circle)

        # Draw trails
        if self.sim.trail_enabled and self.frame_count > 30:
            try:
                trails, _ = self.sim.get_trails()
                for i in range(self.config.n_bodies):
                    trail = trails[i]
                    mask = np.any(trail != 0, axis=1)
                    if mask.sum() > 2:
                        valid = trail[mask][-500:]  # Last 500 points
                        color = self.colors[i % len(self.colors)]
                        n = len(valid)

                        # Draw fading trail
                        for j in range(1, n, 2):  # Skip every other for speed
                            alpha = (j / n) * 0.6 * (0.5 + audio_frame.mid_energy * 0.5)
                            self.ax.plot(
                                valid[j-1:j+1, 0], valid[j-1:j+1, 1],
                                color=color, alpha=min(alpha, 0.8),
                                linewidth=1.5 + audio_frame.treble_energy
                            )
            except:
                pass

        # Draw bodies with glow
        glow = 1.0 + audio_frame.beat_strength * 1.5

        for i in range(self.config.n_bodies):
            color = self.colors[i % len(self.colors)]
            x, y = positions[i, 0], positions[i, 1]

            # Glow layers
            for r, a in [(0.25 * glow, 0.05), (0.15 * glow, 0.12), (0.08 * glow, 0.25)]:
                c = plt.Circle((x, y), r, color=color, alpha=a)
                self.ax.add_patch(c)

            # Core
            size = 60 + audio_frame.beat_strength * 80
            self.ax.scatter([x], [y], c=[color], s=size, zorder=10)

        # Time overlay
        t = audio_frame.time
        self.ax.text(0.02, 0.97, f'{int(t//60)}:{int(t%60):02d}',
                    transform=self.ax.transAxes, color='white',
                    fontsize=14, alpha=0.5, fontfamily='monospace',
                    verticalalignment='top')

        # Beat indicator
        if audio_frame.beat_strength > 0.5:
            self.ax.text(0.98, 0.97, '●', transform=self.ax.transAxes,
                        color='#ff4444', fontsize=24, alpha=0.8,
                        verticalalignment='top', horizontalalignment='right')

        # Render to raw RGB
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        # Convert RGBA to RGB
        rgb = img[:, :, :3].copy()
        return rgb.tobytes()

    def _start_ffmpeg(self):
        """Start ffmpeg process for HLS encoding."""
        width, height = self.resolution

        # Use libx264 - NVENC not available on Lambda GH200
        print("Using libx264 software encoding (NVENC unavailable)")
        encoder = 'libx264'
        encoder_opts = ['-preset', 'fast', '-crf', '23']

        playlist_path = self.output_dir / "stream.m3u8"

        # Use filter_complex to properly sync video stdin with audio file
        cmd = [
            'ffmpeg', '-y',
            '-thread_queue_size', '512',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',  # stdin for video
            '-thread_queue_size', '512',
            '-i', self.audio_path,  # Audio file
            '-map', '0:v',  # Video from first input
            '-map', '1:a',  # Audio from second input
            '-c:v', encoder,
            *encoder_opts,
            '-b:v', self.bitrate,
            '-maxrate', self.bitrate,
            '-bufsize', '16M',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-g', str(int(self.fps * 2)),  # Keyframe every 2 sec
            '-shortest',  # Stop when shortest input ends
            '-f', 'hls',
            '-hls_time', '4',
            '-hls_list_size', '5',
            '-hls_flags', 'delete_segments',
            '-hls_segment_filename', str(self.output_dir / 'segment_%03d.ts'),
            str(playlist_path)
        ]

        print(f"Starting ffmpeg...")
        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=width * height * 3 * 2  # Buffer 2 frames
        )

    def run(self):
        """Run the streaming loop."""
        self.running = True
        self._start_ffmpeg()

        steps_per_frame = 15
        total_frames = self.audio.total_frames

        print(f"\nStreaming {total_frames} frames @ {self.fps} FPS")
        print(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        print(f"HLS output: {self.output_dir}/stream.m3u8\n")

        start_time = time.time()

        try:
            while self.running and self.frame_count < total_frames:
                # Get audio for this frame
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

                # Render frame
                frame_data = self._render_frame(audio_frame)

                # Send to ffmpeg
                try:
                    self.ffmpeg_process.stdin.write(frame_data)
                except BrokenPipeError:
                    print("FFmpeg pipe broken!")
                    break

                self.frame_count += 1

                # Progress
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = self.frame_count / elapsed
                    pct = self.frame_count / total_frames * 100
                    print(f"\r  Frame {self.frame_count}/{total_frames} ({pct:.1f}%) - {fps_actual:.1f} fps", end='', flush=True)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.stop()

        print(f"\n\nCompleted {self.frame_count} frames")

    def stop(self):
        """Stop streaming."""
        self.running = False
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait(timeout=10)
            self.ffmpeg_process = None


# Simple HLS-capable HTTP server
class HLSHandler(SimpleHTTPRequestHandler):
    """Serve HLS content with proper headers."""

    def __init__(self, *args, directory=None, **kwargs):
        self.hls_dir = directory
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        # CORS for video playback
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        if self.path.endswith('.m3u8'):
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
        elif self.path.endswith('.ts'):
            self.send_header('Content-Type', 'video/MP2T')
        super().end_headers()

    def do_GET(self):
        # Serve viewer page at root
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
    <title>Celestial Chaos - HLS Stream</title>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #000;
            color: #fff;
            font-family: 'Courier New', monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 20px;
        }
        h1 {
            color: #0ff;
            margin-bottom: 15px;
            text-shadow: 0 0 20px #0ff;
        }
        #status {
            color: #888;
            margin-bottom: 20px;
        }
        video {
            max-width: 95vw;
            max-height: 80vh;
            border: 2px solid #333;
            border-radius: 10px;
            box-shadow: 0 0 50px rgba(0, 255, 255, 0.3);
        }
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 15px;
        }
        button {
            background: rgba(0, 255, 255, 0.1);
            color: #0ff;
            border: 1px solid #0ff;
            padding: 12px 24px;
            cursor: pointer;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            transition: all 0.3s;
        }
        button:hover {
            background: #0ff;
            color: #000;
        }
        #info {
            margin-top: 15px;
            color: #666;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <h1>CELESTIAL CHAOS</h1>
    <div id="status">Loading stream...</div>
    <video id="video" controls autoplay muted></video>
    <div class="controls">
        <button onclick="document.getElementById('video').muted = false">🔊 Unmute</button>
        <button onclick="location.reload()">↻ Refresh</button>
    </div>
    <div id="info"></div>

    <script>
        const video = document.getElementById('video');
        const status = document.getElementById('status');
        const info = document.getElementById('info');

        function loadStream() {
            const streamUrl = '/stream.m3u8';

            if (Hls.isSupported()) {
                const hls = new Hls({
                    liveSyncDurationCount: 3,
                    liveMaxLatencyDurationCount: 6,
                    maxBufferLength: 10,
                    maxMaxBufferLength: 30
                });

                hls.loadSource(streamUrl);
                hls.attachMedia(video);

                hls.on(Hls.Events.MANIFEST_PARSED, () => {
                    status.textContent = '● LIVE - Click video or Unmute for audio';
                    status.style.color = '#0f0';
                    video.play();
                });

                hls.on(Hls.Events.ERROR, (event, data) => {
                    if (data.fatal) {
                        status.textContent = 'Stream not ready - retrying...';
                        status.style.color = '#ff0';
                        setTimeout(loadStream, 3000);
                    }
                });

                // Update info
                setInterval(() => {
                    if (hls.levels && hls.levels.length > 0) {
                        const level = hls.levels[hls.currentLevel];
                        if (level) {
                            info.textContent = `${level.width}x${level.height} @ ${level.bitrate/1000}kbps`;
                        }
                    }
                }, 2000);

            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                // Native HLS (Safari)
                video.src = streamUrl;
                video.addEventListener('loadedmetadata', () => {
                    status.textContent = '● LIVE';
                    status.style.color = '#0f0';
                    video.play();
                });
            } else {
                status.textContent = 'HLS not supported in this browser';
                status.style.color = '#f00';
            }
        }

        loadStream();
    </script>
</body>
</html>'''


def serve_hls(directory, port=8765):
    """Start HTTP server for HLS content."""
    os.chdir(directory)
    handler = lambda *args, **kwargs: HLSHandler(*args, directory=directory, **kwargs)
    server = HTTPServer(('0.0.0.0', port), handler)
    print(f"HLS server: http://0.0.0.0:{port}")
    server.serve_forever()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Celestial Chaos HLS Stream')
    parser.add_argument('--audio', '-a', default='assets/audio/still-night.mp3')
    parser.add_argument('--bodies', '-n', type=int, default=3)
    parser.add_argument('--resolution', '-r', default='1080p',
                       choices=['720p', '1080p', '1440p', '4k'])
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--bitrate', '-b', default='8M')
    parser.add_argument('--http-port', type=int, default=8765)
    parser.add_argument('--output-dir', default='/tmp/hls')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    res_map = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '4k': (3840, 2160)
    }
    resolution = res_map[args.resolution]

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║         CELESTIAL CHAOS - HLS Live Stream                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Resolution: {args.resolution} ({resolution[0]}x{resolution[1]})                              ║
    ║  FPS: {args.fps}  Bitrate: {args.bitrate}  Bodies: {args.bodies}                       ║
    ║  Audio: {args.audio:<47} ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Start HTTP server in background
    http_thread = threading.Thread(
        target=serve_hls,
        args=(args.output_dir, args.http_port),
        daemon=True
    )
    http_thread.start()

    # Run streamer
    streamer = HLSStreamer(
        audio_path=args.audio,
        output_dir=args.output_dir,
        n_bodies=args.bodies,
        resolution=resolution,
        fps=args.fps,
        bitrate=args.bitrate,
        use_gpu=not args.cpu
    )

    try:
        streamer.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
