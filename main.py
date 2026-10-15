#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-file Discord binaural/spatial audio bot.

Features:
- YouTube and NicoNico playback through yt-dlp -> FFmpeg -> PCM
- HRTF rendering from a SOFA file (auto-downloaded on first run)
- Procedural early reflections / reverberation
- Prebuffered playback while audio download/decode/DSP continue in parallel
- Direction and movement control; no Lavalink required

FFmpeg: uses imageio-ffmpeg when installed, otherwise searches PATH.
Required Python packages: discord.py[voice], yt-dlp, numpy, scipy, h5py, imageio-ffmpeg
Environment: DISCORD_TOKEN (required), optional DISCORD_GUILD_ID, HRTF_SOFA_PATH,
             YTDLP_COOKIES_FILE, INITIAL_BUFFER_SECONDS.

HRTF data attribution (CC BY 4.0): Hagen Wierstorf, Matthias Geier, Sascha Spors,
and Vera Erbes, "KEMAR HRTFs with low frequency correction".
https://github.com/spatialaudio/lf-corrected-kemar-hrtfs
"""
from __future__ import annotations

import asyncio
import collections
import io
import logging
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import discord
from discord import app_commands
from discord.ext import commands
import h5py
import numpy as np
from scipy import signal

# ------------------------------ Configuration ------------------------------
SAMPLE_RATE = 48_000
CHANNELS = 2
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 960 samples/channel
FRAME_BYTES = FRAME_SAMPLES * CHANNELS * 2     # int16 stereo = 3840 bytes
PROCESS_BLOCK_SAMPLES = 4_800                   # 100 ms DSP blocks
PROCESS_BLOCK_SECONDS = PROCESS_BLOCK_SAMPLES / SAMPLE_RATE

MAX_BUFFER_SECONDS = max(12.0, float(os.getenv("MAX_BUFFER_SECONDS", "12")))
INITIAL_BUFFER_SECONDS = min(max(3.0, float(os.getenv("INITIAL_BUFFER_SECONDS", "10"))), MAX_BUFFER_SECONDS - 1.0)
NEXT_TRACK_BUFFER_SECONDS = max(2.0, float(os.getenv("NEXT_TRACK_BUFFER_SECONDS", "4")))
MAX_BUFFER_FRAMES = int(MAX_BUFFER_SECONDS * 1000 / FRAME_MS)

CACHE_DIR = Path(os.getenv("SPATIAL_AUDIO_CACHE", str(Path.home() / ".cache" / "spatial-audio-bot")))
SOFA_PATH = Path(os.getenv("HRTF_SOFA_PATH", str(CACHE_DIR / "KEMAR_HRTFs_lfcorr.sofa"))).expanduser()
SOFA_URL = "https://raw.githubusercontent.com/spatialaudio/lf-corrected-kemar-hrtfs/master/KEMAR_HRTFs_lfcorr.sofa"
SOFA_LICENSE_URL = "https://raw.githubusercontent.com/spatialaudio/lf-corrected-kemar-hrtfs/master/LICENSE_for_data"
SOFA_ATTRIBUTION = (
    "HRTF data: Hagen Wierstorf, Matthias Geier, Sascha Spors and Vera Erbes, "
    '"KEMAR HRTFs with low frequency correction". Licensed under CC BY 4.0. '
    "https://creativecommons.org/licenses/by/4.0/ ; "
    "https://github.com/spatialaudio/lf-corrected-kemar-hrtfs\n"
)

# SOFA spherical azimuth convention is used internally: 0=front, +90=left,
# -90=right, +/-180=back. Thus increasing azimuth travels counter-clockwise
# when viewed from above.
DIRECTION_PRESETS: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "front_left": (45.0, 0.0),
    "left": (90.0, 0.0),
    "back_left": (135.0, 0.0),
    "back": (180.0, 0.0),
    "back_right": (-135.0, 0.0),
    "right": (-90.0, 0.0),
    "front_right": (-45.0, 0.0),
    "up": (0.0, 55.0),
    "down": (0.0, -35.0),
}
DIRECTION_LABELS = {
    "front": "前方", "front_left": "左前方", "left": "左", "back_left": "左後方",
    "back": "後方", "back_right": "右後方", "right": "右", "front_right": "右前方",
    "up": "上方", "down": "下方",
}
MOVEMENT_LABELS = {
    "circle": "円周移動", "left_right": "左右往復", "front_back": "前後往復", "random": "ランダム移動",
}
SPEED_MULTIPLIER = {"slow": 0.5, "normal": 1.0, "fast": 2.0}
SPEED_LABELS = {"slow": "ゆっくり", "normal": "普通", "fast": "速い"}
ROTATION_LABELS = {"clockwise": "時計回り", "counterclockwise": "反時計回り"}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("spatial-audio-bot")


def get_ffmpeg_executable() -> str:
    """Resolve FFmpeg from imageio-ffmpeg first, then from the system PATH."""
    try:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        if executable and Path(executable).is_file():
            return executable
    except Exception as exc:
        log.debug("imageio-ffmpeg could not provide an executable: %s", exc)

    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    raise RuntimeError(
        "FFmpegが見つかりません。MrtCloudのrequirements.txtに "
        "imageio-ffmpeg を追加して再起動してください。"
    )


# --------------------------------- Utilities --------------------------------
def wrap_azimuth(angle: float) -> float:
    """Normalize azimuth to [-180, 180]."""
    value = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if value == -180.0 and angle > 0 else value


def is_supported_media_url(url: str) -> bool:
    """Allow only the two requested public video providers (avoid arbitrary URLs)."""
    try:
        parsed = urllib.parse.urlparse(url.strip())
        host = (parsed.hostname or "").lower().rstrip(".")
        allowed = (
            host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
            or host == "youtube-nocookie.com" or host.endswith(".youtube-nocookie.com")
            or host == "nicovideo.jp" or host.endswith(".nicovideo.jp") or host == "nico.ms"
        )
        return parsed.scheme in ("http", "https") and allowed
    except Exception:
        return False


def read_exact(stream, wanted: int, *, allow_partial: bool = False) -> bytes:
    """Read up to wanted bytes from a pipe, tolerating short pipe reads."""
    chunks = bytearray()
    while len(chunks) < wanted:
        part = stream.read(wanted - len(chunks))
        if not part:
            break
        chunks.extend(part)
    if not chunks:
        return b""
    if len(chunks) < wanted and not allow_partial:
        return bytes(chunks)
    return bytes(chunks)


def download_file(url: str, destination: Path, max_bytes: int = 80 * 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "SpatialAudioBot/1.0"})
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=45) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError(f"ファイルが上限サイズ {max_bytes} bytes を超えました")
                out.write(chunk)
        if total < 1024:
            raise RuntimeError("ダウンロードされたHRTFファイルが小さすぎます")
        tmp.replace(destination)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def ensure_hrtf_file() -> Path:
    if SOFA_PATH.is_file() and SOFA_PATH.stat().st_size > 1024:
        return SOFA_PATH
    log.info("HRTF SOFAデータを初回取得します: %s", SOFA_PATH)
    SOFA_PATH.parent.mkdir(parents=True, exist_ok=True)
    download_file(SOFA_URL, SOFA_PATH)
    try:
        license_path = SOFA_PATH.parent / "HRTF_ATTRIBUTION.txt"
        download_file(SOFA_LICENSE_URL, license_path, max_bytes=1024 * 1024)
    except Exception as exc:
        log.warning("HRTFデータのライセンス表記ファイルを取得できませんでした: %s", exc)
        (SOFA_PATH.parent / "HRTF_ATTRIBUTION.txt").write_text(SOFA_ATTRIBUTION, encoding="utf-8")
    return SOFA_PATH


# ------------------------------- HRTF handling ------------------------------
class HRTFDatabase:
    """Load SOFA measurements and provide smooth local interpolation of HRIRs."""

    def __init__(self, positions: np.ndarray, left: np.ndarray, right: np.ndarray, sample_rate: int):
        self.positions = np.asarray(positions, dtype=np.float32)
        self.left = np.asarray(left, dtype=np.float32)
        self.right = np.asarray(right, dtype=np.float32)
        self.sample_rate = int(sample_rate)
        self.vectors = self._spherical_to_vectors(self.positions[:, 0], self.positions[:, 1])
        self.filter_len = int(self.left.shape[-1])

    @staticmethod
    def _spherical_to_vectors(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        # x points forward, +y points left, z points up.
        return np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el))).astype(np.float32)

    @classmethod
    def load(cls) -> "HRTFDatabase":
        path = ensure_hrtf_file()
        with h5py.File(path, "r") as sofa:
            for required in ("SourcePosition", "Data.IR", "Data.SamplingRate"):
                if required not in sofa:
                    raise RuntimeError(f"SOFAファイルに {required} がありません")
            positions = np.asarray(sofa["SourcePosition"][:], dtype=np.float64)
            irs = np.asarray(sofa["Data.IR"][:], dtype=np.float32)
            sr_value = np.asarray(sofa["Data.SamplingRate"][:]).reshape(-1)
            src_type = sofa["SourcePosition"].attrs.get("Type", "spherical")
            src_units = sofa["SourcePosition"].attrs.get("Units", "degree, degree, metre")
            if isinstance(src_type, bytes):
                src_type = src_type.decode("ascii", "ignore")
            if isinstance(src_units, bytes):
                src_units = src_units.decode("ascii", "ignore")
            if "spherical" not in str(src_type).lower():
                raise RuntimeError(f"この試作版は球面座標SOFAを想定しています (Type={src_type!r})")
            # This data set uses azimuth/elevation in degrees and distance in metres.
            units = str(src_units).lower()
            if "radian" in units or "rad," in units:
                positions[:, :2] = np.rad2deg(positions[:, :2])
            sr = int(round(float(sr_value[0])))

        if irs.ndim != 3:
            raise RuntimeError(f"Data.IR の次元が想定外です: {irs.shape}")
        # Standard SOFA convention: (measurement, receiver, sample).
        if irs.shape[1] < 2 and irs.shape[0] >= 2:
            # Try receiver-first files only when the dimensions clearly indicate it.
            irs = np.transpose(irs, (1, 0, 2))
        if irs.shape[1] < 2:
            raise RuntimeError(f"左右2耳のData.IRが必要です: {irs.shape}")
        if positions.shape[0] != irs.shape[0] or positions.shape[1] < 2:
            raise RuntimeError(f"SourcePositionとData.IRの形状が一致しません: {positions.shape}, {irs.shape}")

        irs = np.nan_to_num(irs[:, :2, :], copy=False)
        if sr != SAMPLE_RATE:
            from fractions import Fraction
            ratio = Fraction(SAMPLE_RATE, sr).limit_denominator(1000)
            log.info("HRTFを %d Hz から %d Hz に変換中", sr, SAMPLE_RATE)
            irs = signal.resample_poly(irs, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
        log.info("HRTF読み込み完了: %d方向, フィルター長%d samples", irs.shape[0], irs.shape[-1])
        return cls(positions[:, :3], irs[:, 0, :], irs[:, 1, :], SAMPLE_RATE)

    def filters_at(self, azimuth: float, elevation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        query = self._spherical_to_vectors(np.array([azimuth]), np.array([elevation]))[0]
        # Four nearest directions, weighted by spherical proximity. Dense local interpolation
        # avoids obvious step changes as the source moves between measured positions.
        dots = np.clip(self.vectors @ query, -1.0, 1.0)
        count = min(4, dots.size)
        ids = np.argpartition(dots, -count)[-count:]
        angular = np.arccos(dots[ids])
        exact = np.flatnonzero(angular < 1e-5)
        if exact.size:
            idx = ids[int(exact[0])]
            return self.left[idx].copy(), self.right[idx].copy()
        weights = 1.0 / np.maximum(angular, 1e-3) ** 2
        weights /= np.sum(weights)
        left = np.sum(self.left[ids] * weights[:, None], axis=0, dtype=np.float32)
        right = np.sum(self.right[ids] * weights[:, None], axis=0, dtype=np.float32)
        return left.astype(np.float32), right.astype(np.float32)


class OverlapAddConvolver:
    """Stateful overlap-add FIR convolution; handles filters longer than a block."""
    def __init__(self):
        self.tail: Optional[np.ndarray] = None
        self.filter_length: Optional[int] = None

    def process(self, block: np.ndarray, impulse: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        impulse = np.asarray(impulse, dtype=np.float32)
        full = signal.fftconvolve(block, impulse, mode="full").astype(np.float32, copy=False)
        n = block.size
        tail_len = impulse.size - 1
        if self.tail is None or self.filter_length != impulse.size:
            self.tail = np.zeros(tail_len, dtype=np.float32)
            self.filter_length = impulse.size
        if tail_len:
            full[:tail_len] += self.tail
            self.tail = full[n:n + tail_len].copy()
        return full[:n].copy()


class SpatialController:
    def __init__(self):
        self.lock = threading.RLock()
        self.azimuth = 0.0
        self.elevation = 0.0
        self.mode = "fixed"
        self.pattern = "circle"
        self.speed = "normal"
        self.rotation = "clockwise"
        self.seed = random.randrange(1, 2**31 - 1)
        self.last_audio_time = 0.0
        self.motion_start_time = 0.0

    def reset_timeline(self) -> None:
        """Start movement time at the beginning of a new track."""
        with self.lock:
            self.last_audio_time = 0.0
            self.motion_start_time = 0.0

    def set_direction(self, direction: str) -> None:
        with self.lock:
            self.azimuth, self.elevation = DIRECTION_PRESETS[direction]
            if self.mode == "moving":
                self.motion_start_time = self.last_audio_time

    def set_motion(self, mode: str) -> None:
        with self.lock:
            if mode != self.mode:
                self.motion_start_time = self.last_audio_time
            self.mode = mode

    def set_movement(self, pattern: str, speed: str, rotation: str) -> None:
        with self.lock:
            self.pattern, self.speed, self.rotation = pattern, speed, rotation
            self.mode = "moving"
            self.motion_start_time = self.last_audio_time
            self.seed = random.randrange(1, 2**31 - 1)

    def get_position(self, audio_time: float) -> tuple[float, float]:
        with self.lock:
            base_az, base_el = self.azimuth, self.elevation
            mode, pattern = self.mode, self.pattern
            speed, rotation, seed = self.speed, self.rotation, self.seed
            self.last_audio_time = max(self.last_audio_time, audio_time)
            motion_start_time = self.motion_start_time
        if mode != "moving":
            return wrap_azimuth(base_az), base_el

        mult = SPEED_MULTIPLIER.get(speed, 1.0)
        sign = -1.0 if rotation == "clockwise" else 1.0
        t = max(0.0, audio_time - motion_start_time)
        if pattern == "circle":
            # Default: one lap per 20 seconds; slow=40s, fast=10s.
            az = base_az + sign * 360.0 * (t / 20.0) * mult
            return wrap_azimuth(az), base_el
        if pattern == "left_right":
            # Smooth back-and-forth around the chosen direction (±90 degrees).
            phase = (t * mult / 8.0) % 2.0
            tri = 1.0 - abs(phase - 1.0) * 2.0  # -1 -> +1 -> -1, no endpoint jump
            return wrap_azimuth(base_az + 90.0 * tri), base_el
        if pattern == "front_back":
            # Travel along one side of the head from front to back and return.
            phase = (t * mult / 10.0) % 2.0
            tri = phase if phase <= 1.0 else 2.0 - phase
            az = base_az + 180.0 * tri
            return wrap_azimuth(az), base_el
        if pattern == "random":
            # Deterministic targets per segment, smoothly cross-faded over each segment.
            segment_seconds = 2.4 / mult
            seg = int(t // segment_seconds)
            frac = (t - seg * segment_seconds) / segment_seconds
            def target(which: int) -> tuple[float, float]:
                rng = random.Random((seed + which * 7919) & 0xFFFFFFFF)
                return rng.uniform(-180.0, 180.0), rng.uniform(-25.0, 45.0)
            a0, e0 = target(seg)
            a1, e1 = target(seg + 1)
            delta = wrap_azimuth(a1 - a0)
            # Smoothstep to reduce sudden angular velocity changes.
            f = frac * frac * (3.0 - 2.0 * frac)
            return wrap_azimuth(a0 + delta * f), e0 + (e1 - e0) * f
        return wrap_azimuth(base_az), base_el


class DSPRenderer:
    """Stereo virtual-source HRTF renderer followed by a lightweight room IR."""
    def __init__(self, hrtf: HRTFDatabase, controller: SpatialController):
        self.hrtf = hrtf
        self.controller = controller
        self.conv = {
            "ll": OverlapAddConvolver(), "lr": OverlapAddConvolver(),
            "rl": OverlapAddConvolver(), "rr": OverlapAddConvolver(),
            "rev_l": OverlapAddConvolver(), "rev_r": OverlapAddConvolver(),
        }
        self.reverb_l, self.reverb_r = self._make_room_ir()

    @staticmethod
    def _make_room_ir() -> tuple[np.ndarray, np.ndarray]:
        """Create a subtle but audible fixed room response (early reflections + diffuse tail)."""
        length = int(0.60 * SAMPLE_RATE)
        t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
        rng_l = np.random.default_rng(0xA51)
        rng_r = np.random.default_rng(0xB72)
        left = (rng_l.standard_normal(length).astype(np.float32)
                * np.exp(-t / 0.22).astype(np.float32) * 0.018)
        right = (rng_r.standard_normal(length).astype(np.float32)
                 * np.exp(-t / 0.235).astype(np.float32) * 0.018)
        left[0] = right[0] = 0.0  # keep the dry/direct path distinct
        reflections_l = [(0.011, 0.26), (0.019, -0.20), (0.031, 0.15), (0.044, -0.11),
                         (0.061, 0.085), (0.083, 0.060), (0.109, -0.040)]
        reflections_r = [(0.013, 0.24), (0.022, -0.18), (0.034, 0.16), (0.047, -0.095),
                         (0.066, 0.078), (0.087, 0.052), (0.116, -0.035)]
        for delay, gain in reflections_l:
            left[int(delay * SAMPLE_RATE)] += gain
        for delay, gain in reflections_r:
            right[int(delay * SAMPLE_RATE)] += gain
        # A little smooth decay makes the tail less metallic than pure sparse echoes.
        return left, right

    def process(self, stereo: np.ndarray, audio_time: float) -> np.ndarray:
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError(f"stereo PCM shape must be (N,2), got {stereo.shape}")
        if stereo.shape[0] == 0:
            return stereo.astype(np.float32, copy=False)
        az, el = self.controller.get_position(audio_time)
        # Model a modest virtual stereo source rather than summing the song to mono.
        # SOFA positive azimuth points left; original L and R are separated by +/-10°.
        source_left_az = wrap_azimuth(az + 10.0)
        source_right_az = wrap_azimuth(az - 10.0)
        h_ll, h_lr = self.hrtf.filters_at(source_left_az, el)
        h_rl, h_rr = self.hrtf.filters_at(source_right_az, el)
        in_l = stereo[:, 0]
        in_r = stereo[:, 1]
        # 4 directional paths preserve much of the original mix while moving the image.
        out_l = (self.conv["ll"].process(in_l, h_ll) + self.conv["rl"].process(in_r, h_rl)) * 0.50
        out_r = (self.conv["lr"].process(in_l, h_lr) + self.conv["rr"].process(in_r, h_rr)) * 0.50
        # Fixed room response. DSP is stateful across blocks so echoes do not restart each block.
        wet_l = self.conv["rev_l"].process(out_l, self.reverb_l)
        wet_r = self.conv["rev_r"].process(out_r, self.reverb_r)
        result = np.column_stack((out_l + wet_l * 0.38, out_r + wet_r * 0.38)).astype(np.float32, copy=False)
        result *= 0.92
        # Peak guard with block-level gain only when necessary; no normalisation on every block.
        peak = float(np.max(np.abs(result))) if result.size else 0.0
        if peak > 0.985:
            result *= 0.985 / peak
        return np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def flush(self) -> list[np.ndarray]:
        """Emit pending convolution/reverb tails after end-of-track."""
        result = []
        blocks = math.ceil(max(len(self.reverb_l), self.hrtf.filter_len) / PROCESS_BLOCK_SAMPLES) + 1
        zeros = np.zeros((PROCESS_BLOCK_SAMPLES, 2), dtype=np.float32)
        for i in range(blocks):
            result.append(self.process(zeros, 1e9 + i * PROCESS_BLOCK_SECONDS))
        return result


# ---------------------------- Media and playback ----------------------------
@dataclass
class Track:
    url: str
    title: str
    webpage_url: str
    duration: Optional[float] = None
    extractor: str = "unknown"


def make_ytdlp_options() -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": False,
        "socket_timeout": 20,
        "retries": 2,
        "extractor_retries": 1,
        "noplaylist": True,
    }
    cookiefile = os.getenv("YTDLP_COOKIES_FILE")
    if cookiefile:
        opts["cookiefile"] = cookiefile
    return opts


def extract_track_info(url: str) -> Track:
    if not is_supported_media_url(url):
        raise ValueError("YouTubeまたはニコニコ動画のURLを指定してください")
    from yt_dlp import YoutubeDL
    with YoutubeDL(make_ytdlp_options()) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError("動画情報を取得できませんでした")
    if "entries" in info and info.get("entries"):
        info = next((entry for entry in info["entries"] if entry), None)
        if info is None:
            raise RuntimeError("再生可能な動画が見つかりませんでした")
    webpage_url = info.get("webpage_url") or url
    return Track(
        url=url,
        title=str(info.get("title") or "タイトル不明")[:180],
        webpage_url=str(webpage_url),
        duration=float(info["duration"]) if info.get("duration") else None,
        extractor=str(info.get("extractor_key") or info.get("extractor") or "unknown"),
    )


def search_media(query: str) -> list[Track]:
    from yt_dlp import YoutubeDL
    q = query.strip()
    if not q:
        return []
    results: list[Track] = []
    opts = make_ytdlp_options()
    opts["extract_flat"] = "in_playlist"
    opts["playlistend"] = 5
    with YoutubeDL(opts) as ydl:
        # Search providers independently; one provider failing must not suppress the other.
        for prefix in ("ytsearch5:", "nicosearch5:"):
            try:
                info = ydl.extract_info(prefix + q, download=False)
                for entry in (info or {}).get("entries") or []:
                    if not entry:
                        continue
                    url = entry.get("url") or entry.get("webpage_url")
                    if not url:
                        continue
                    if not str(url).startswith(("http://", "https://")):
                        # Some flat search entries contain an ID rather than a URL.
                        eid = entry.get("id")
                        extractor = str(entry.get("extractor_key") or entry.get("ie_key") or "").lower()
                        if eid and ("nico" in extractor or prefix.startswith("nicosearch")):
                            url = f"https://www.nicovideo.jp/watch/{eid}"
                        elif eid:
                            url = f"https://www.youtube.com/watch?v={eid}"
                    if not is_supported_media_url(str(url)):
                        continue
                    results.append(Track(
                        url=str(url), title=str(entry.get("title") or "タイトル不明")[:180],
                        webpage_url=str(url),
                        duration=float(entry["duration"]) if entry.get("duration") else None,
                        extractor=str(entry.get("extractor_key") or entry.get("ie_key") or "unknown"),
                    ))
            except Exception as exc:
                log.warning("検索に失敗しました (%s): %s", prefix, exc)
    # Deduplicate URLs and cap the response.
    unique: list[Track] = []
    seen: set[str] = set()
    for track in results:
        if track.webpage_url not in seen:
            seen.add(track.webpage_url)
            unique.append(track)
    return unique[:10]


class ProcessedAudioSource(discord.AudioSource):
    """Download/decode/DSP producer thread plus a bounded 20ms PCM frame queue."""
    def __init__(self, track: Track, renderer: DSPRenderer, buffer_seconds: float):
        super().__init__()
        self.track = track
        self.renderer = renderer
        self.target_frames = max(1, int(buffer_seconds * 1000 / FRAME_MS))
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=MAX_BUFFER_FRAMES)
        self.ready_event = threading.Event()
        self.finished_event = threading.Event()
        self.cancel_event = threading.Event()
        self.error: Optional[str] = None
        self.underruns = 0
        self._thread: Optional[threading.Thread] = None
        self._yt_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._yt_stderr: Optional[io.BufferedRandom] = None
        self._ffmpeg_stderr: Optional[io.BufferedRandom] = None
        self._frames_written = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._producer, name="spatial-dsp-producer", daemon=True)
        self._thread.start()

    def wait_ready(self, timeout: float) -> bool:
        if not self.ready_event.wait(timeout):
            raise TimeoutError(f"音声準備が{timeout:.0f}秒以内に完了しませんでした")
        if self.error:
            raise RuntimeError(self.error)
        return True

    def _start_pipeline(self) -> None:
        ffmpeg_executable = get_ffmpeg_executable()
        cookies = os.getenv("YTDLP_COOKIES_FILE")
        ytdlp_cmd = [
            sys.executable, "-m", "yt_dlp", "--no-playlist", "--quiet", "--no-warnings",
            "--no-progress", "-f", "bestaudio/best", "-o", "-", self.track.url,
        ]
        if cookies:
            ytdlp_cmd[-1:-1] = ["--cookies", cookies]
        self._yt_stderr = tempfile.TemporaryFile()
        self._ffmpeg_stderr = tempfile.TemporaryFile()
        self._yt_proc = subprocess.Popen(
            ytdlp_cmd, stdout=subprocess.PIPE, stderr=self._yt_stderr,
            stdin=subprocess.DEVNULL, bufsize=0,
        )
        assert self._yt_proc.stdout is not None
        ffmpeg_cmd = [
            ffmpeg_executable, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", "pipe:0", "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "pipe:1",
        ]
        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=self._yt_proc.stdout, stdout=subprocess.PIPE,
            stderr=self._ffmpeg_stderr, bufsize=0,
        )
        self._yt_proc.stdout.close()

    def _get_error_details(self) -> str:
        details = []
        for label, handle in (("yt-dlp", self._yt_stderr), ("FFmpeg", self._ffmpeg_stderr)):
            if handle:
                try:
                    handle.flush()
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 2500))
                    raw = handle.read(2500).decode("utf-8", "replace").strip()
                    if raw:
                        details.append(f"{label}: {raw[-1200:]}")
                except Exception:
                    pass
        return "\n".join(details)

    def _put_frame(self, frame: bytes) -> bool:
        while not self.cancel_event.is_set():
            try:
                self.frames.put(frame, timeout=0.10)
                self._frames_written += 1
                if self.frames.qsize() >= self.target_frames:
                    self.ready_event.set()
                return True
            except queue.Full:
                continue
        return False

    def _emit_pcm(self, stereo: np.ndarray) -> bool:
        if stereo.size == 0:
            return True
        pcm = np.clip(stereo, -0.985, 0.985)
        pcm16 = (pcm * 32767.0).astype("<i2", copy=False)
        raw = pcm16.tobytes()
        for offset in range(0, len(raw), FRAME_BYTES):
            frame = raw[offset:offset + FRAME_BYTES]
            if len(frame) < FRAME_BYTES:
                frame += b"\x00" * (FRAME_BYTES - len(frame))
            if not self._put_frame(frame):
                return False
        return True

    def _process_and_emit(self, audio: np.ndarray, audio_time: float) -> bool:
        # Process final partial blocks at their real length so the convolution tail
        # remains aligned with the actual end of the decoded stream.
        if audio.shape[0] == 0:
            return True
        rendered = self.renderer.process(audio, audio_time)
        return self._emit_pcm(rendered)

    def _producer(self) -> None:
        audio_time = 0.0
        partial = bytearray()
        bytes_per_block = PROCESS_BLOCK_SAMPLES * CHANNELS * 4
        try:
            self._start_pipeline()
            assert self._ffmpeg_proc and self._ffmpeg_proc.stdout
            stdout = self._ffmpeg_proc.stdout
            while not self.cancel_event.is_set():
                data = read_exact(stdout, bytes_per_block, allow_partial=True)
                if not data:
                    break
                partial.extend(data)
                while len(partial) >= bytes_per_block and not self.cancel_event.is_set():
                    block_bytes = bytes(partial[:bytes_per_block])
                    del partial[:bytes_per_block]
                    block = np.frombuffer(block_bytes, dtype="<f4").reshape(-1, 2).copy()
                    if not self._process_and_emit(block, audio_time):
                        return
                    audio_time += PROCESS_BLOCK_SECONDS
            if not self.cancel_event.is_set() and partial:
                n_samples = len(partial) // (CHANNELS * 4)
                if n_samples:
                    block = np.frombuffer(bytes(partial[:n_samples * CHANNELS * 4]), dtype="<f4").reshape(-1, 2).copy()
                    if not self._process_and_emit(block, audio_time):
                        return
                    audio_time += n_samples / SAMPLE_RATE
            # Drain the room tail at natural EOF; skip/stop deliberately bypass this.
            if not self.cancel_event.is_set() and self._ffmpeg_proc and self._ffmpeg_proc.poll() is not None:
                for rendered in self.renderer.flush():
                    if self.cancel_event.is_set() or not self._emit_pcm(rendered):
                        break
            if self._yt_proc:
                self._yt_proc.wait(timeout=3)
            if self._ffmpeg_proc:
                self._ffmpeg_proc.wait(timeout=3)
            if not self.cancel_event.is_set():
                yt_code = self._yt_proc.returncode if self._yt_proc else 0
                ff_code = self._ffmpeg_proc.returncode if self._ffmpeg_proc else 0
                if self._frames_written == 0 or (ff_code not in (0, None) and self._frames_written < self.target_frames):
                    detail = self._get_error_details()
                    self.error = "音源ストリームの取得またはデコードに失敗しました。" + ("\n" + detail if detail else "")
        except Exception as exc:
            if not self.cancel_event.is_set():
                details = self._get_error_details()
                self.error = f"音声処理エラー: {type(exc).__name__}: {exc}"
                if details:
                    self.error += "\n" + details
                log.exception("DSP producer failed for %s", self.track.webpage_url)
        finally:
            self.ready_event.set()
            self.finished_event.set()
            self._terminate_processes()

    def _terminate_processes(self) -> None:
        for proc in (self._ffmpeg_proc, self._yt_proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        for proc in (self._ffmpeg_proc, self._yt_proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.wait(timeout=1.5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        for handle in (self._ffmpeg_stderr, self._yt_stderr):
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass

    def read(self) -> bytes:
        try:
            return self.frames.get(timeout=0.015)
        except queue.Empty:
            if self.finished_event.is_set():
                return b""
            self.underruns += 1
            # One 20ms silent frame is preferable to blocking Discord's voice thread.
            if self.underruns == 1 or self.underruns % 50 == 0:
                log.warning("Audio buffer underrun for %s (count=%d)", self.track.title, self.underruns)
            return b"\x00" * FRAME_BYTES

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:
        self.cancel_event.set()
        self.ready_event.set()
        self._terminate_processes()
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)


# -------------------------------- Discord bot -------------------------------
intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)

hrtf_database: Optional[HRTFDatabase] = None
hrtf_load_error: Optional[str] = None
hrtf_loading = False
hrtf_load_task: Optional[asyncio.Task] = None


@dataclass
class QueuedTrack:
    track: Track


class GuildPlayer:
    def __init__(self, guild_id: int):
        self.guild_id = guild_id
        self.voice: Optional[discord.VoiceClient] = None
        self.controller = SpatialController()
        self.queue: collections.deque[Track] = collections.deque()
        self.current: Optional[Track] = None
        self.source: Optional[ProcessedAudioSource] = None
        self.renderer: Optional[DSPRenderer] = None
        self._start_lock = asyncio.Lock()
        self._starting = False
        self.played_once = False
        self._generation = 0

    async def ensure_voice(self, interaction: discord.Interaction) -> None:
        member = interaction.user
        voice_state = getattr(member, "voice", None)
        if not voice_state or not voice_state.channel:
            raise RuntimeError("先にBOTを参加させたいボイスチャンネルへ入ってください")
        channel = voice_state.channel
        if self.voice and self.voice.is_connected():
            if self.voice.channel != channel:
                await self.voice.move_to(channel)
        else:
            self.voice = await channel.connect()

    async def enqueue(self, track: Track) -> int:
        self.queue.append(track)
        position = len(self.queue) + (1 if self.current or self._starting else 0)
        if self.current is None and not self._starting:
            await self.start_next()
        return position

    async def start_next(self) -> None:
        async with self._start_lock:
            if self._starting or self.current is not None:
                return
            if not self.queue:
                return
            if not self.voice or not self.voice.is_connected():
                return
            self._starting = True
            track = self.queue.popleft()
            self.current = track
            generation = self._generation + 1
            self._generation = generation
            try:
                if hrtf_database is None:
                    raise RuntimeError(f"HRTFの読み込みに失敗しています: {hrtf_load_error or '不明なエラー'}")
                # Each track gets fresh convolution/reverb state. Reset the movement clock
                # while preserving the chosen direction/pattern/speed.
                self.controller.reset_timeline()
                self.renderer = DSPRenderer(hrtf_database, self.controller)
                buffer_seconds = INITIAL_BUFFER_SECONDS if not self.played_once else NEXT_TRACK_BUFFER_SECONDS
                source = ProcessedAudioSource(track, self.renderer, buffer_seconds)
                self.source = source
                source.start()
                await asyncio.to_thread(source.wait_ready, 150.0)
                if source.error and source.frames.empty():
                    raise RuntimeError(source.error)
                if not self.voice or not self.voice.is_connected():
                    source.cleanup()
                    raise RuntimeError("準備中にDiscordのボイス接続が切断されました")
                def after_playback(error: Optional[Exception]) -> None:
                    if error:
                        log.error("Voice playback error: %s", error)
                    try:
                        asyncio.run_coroutine_threadsafe(self._after_track(generation, error), bot.loop)
                    except Exception:
                        log.exception("Failed to schedule next-track callback")
                self.voice.play(source, after=after_playback)
                self.played_once = True
                log.info("Playing %s in guild %s; prebuffer=%ss", track.title, self.guild_id, buffer_seconds)
            except Exception:
                if self.source:
                    self.source.cleanup()
                self.current = None
                self.source = None
                raise
            finally:
                self._starting = False

    async def _after_track(self, generation: int, error: Optional[Exception]) -> None:
        # Ignore stale callbacks caused by a skipped/stopped track.
        if generation != self._generation:
            return
        self.current = None
        self.source = None
        if self.queue and self.voice and self.voice.is_connected():
            try:
                await self.start_next()
            except Exception as exc:
                log.exception("Could not start next track")
                # Continue to a later track rather than leaving the queue permanently stuck.
                self.current = None
                self.source = None
                if self.queue:
                    await self.start_next()
        else:
            log.info("Playback ended in guild %s", self.guild_id)

    def skip(self) -> bool:
        if self.voice and self.voice.is_connected() and (self.voice.is_playing() or self.voice.is_paused()):
            # Keep the current generation so the AudioPlayer callback can advance the queue.
            if self.source:
                self.source.cancel_event.set()
            self.voice.stop()
            return True
        return False

    def stop(self) -> None:
        self.queue.clear()
        self._generation += 1
        self.current = None
        if self.source:
            self.source.cancel_event.set()
        if self.voice and self.voice.is_connected() and (self.voice.is_playing() or self.voice.is_paused()):
            self.voice.stop()
        self.source = None


players: dict[int, GuildPlayer] = {}


def get_player(guild_id: int) -> GuildPlayer:
    player = players.get(guild_id)
    if player is None:
        player = GuildPlayer(guild_id)
        players[guild_id] = player
    return player


def interaction_embed(title: str, description: str, *, color: discord.Color = discord.Color.blurple()) -> discord.Embed:
    return discord.Embed(title=title, description=description, color=color)


async def load_hrtf_background() -> None:
    global hrtf_database, hrtf_load_error, hrtf_loading
    if hrtf_database is not None or hrtf_loading:
        return
    hrtf_loading = True
    try:
        hrtf_database = await asyncio.to_thread(HRTFDatabase.load)
        hrtf_load_error = None
        log.info("Spatial audio engine is ready")
    except Exception as exc:
        hrtf_load_error = f"{type(exc).__name__}: {exc}"
        log.exception("HRTF data could not be loaded")
    finally:
        hrtf_loading = False


async def ensure_hrtf_loaded() -> None:
    global hrtf_load_task
    if hrtf_database is not None:
        return
    if hrtf_load_task is None or hrtf_load_task.done():
        hrtf_load_task = asyncio.create_task(load_hrtf_background())
    await hrtf_load_task


@bot.event
async def on_ready():
    global hrtf_load_task
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id if bot.user else "unknown")
    # Sync commands promptly; HRTF download/loading happens in the background.
    try:
        guild_id = os.getenv("DISCORD_GUILD_ID")
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
            log.info("Synced %d commands to guild %s", len(synced), guild_id)
        else:
            synced = await bot.tree.sync()
            log.info("Synced %d global commands", len(synced))
    except Exception:
        log.exception("Failed to sync application commands")
    if hrtf_database is None and (hrtf_load_task is None or hrtf_load_task.done()):
        hrtf_load_task = asyncio.create_task(load_hrtf_background())


@bot.tree.command(name="join", description="あなたのボイスチャンネルにBOTを接続します")
async def join(interaction: discord.Interaction):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True, thinking=True)
    player = get_player(interaction.guild_id)
    try:
        await player.ensure_voice(interaction)
        await interaction.followup.send(f"🔊 **{player.voice.channel.name}** に接続しました。", ephemeral=True)
    except Exception as exc:
        await interaction.followup.send(f"接続できませんでした: {exc}", ephemeral=True)


@bot.tree.command(name="play", description="YouTubeまたはニコニコ動画の音源を立体音響で再生します")
@app_commands.describe(url="YouTubeまたはニコニコ動画の動画URL")
async def play(interaction: discord.Interaction, url: str):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    if not is_supported_media_url(url):
        await interaction.response.send_message("YouTubeまたはニコニコ動画のURLを指定してください。", ephemeral=True)
        return
    await interaction.response.defer(thinking=True)
    try:
        if hrtf_database is None:
            await ensure_hrtf_loaded()
        if hrtf_database is None:
            raise RuntimeError(f"HRTFデータを読み込めていません: {hrtf_load_error or '初期化中'}")
        track = await asyncio.to_thread(extract_track_info, url)
        player = get_player(interaction.guild_id)
        await player.ensure_voice(interaction)
        was_busy = player.current is not None or player._starting
        position = await player.enqueue(track)
        if was_busy:
            await interaction.followup.send(
                f"📥 キューに追加しました（{position}番目）: **[{track.title}]({track.webpage_url})**")
        else:
            await interaction.followup.send(
                f"🎧 再生を開始します: **[{track.title}]({track.webpage_url})**\n"
                f"立体音響処理と先読み中です（初回は最大約{INITIAL_BUFFER_SECONDS:.0f}秒分を準備）。")
    except Exception as exc:
        log.exception("/play failed")
        if interaction.is_expired():
            return
        await interaction.followup.send(f"再生準備に失敗しました: {exc}", ephemeral=True)


@bot.tree.command(name="search", description="YouTubeとニコニコ動画から曲を検索します")
@app_commands.describe(query="曲名・動画タイトル・キーワード")
async def search(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)
    try:
        results = await asyncio.to_thread(search_media, query)
    except Exception as exc:
        await interaction.followup.send(f"検索に失敗しました: {exc}", ephemeral=True)
        return
    if not results:
        await interaction.followup.send("検索結果がありませんでした。サイトの仕様変更やログイン要件の可能性もあります。", ephemeral=True)
        return
    lines = []
    for i, track in enumerate(results, 1):
        site = "ニコニコ動画" if "nico" in track.extractor.lower() or "nicovideo.jp" in track.webpage_url else "YouTube"
        duration = f" ({int(track.duration // 60)}:{int(track.duration % 60):02d})" if track.duration else ""
        lines.append(f"**{i}. [{discord.utils.escape_markdown(track.title)}]({track.webpage_url})** · {site}{duration}")
    embed = interaction_embed(f"検索結果: {query[:150]}", "\n".join(lines))
    embed.set_footer(text="再生するには /play に動画URLを指定してください")
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="queue", description="再生中の曲と再生待ち一覧を表示します")
async def queue_command(interaction: discord.Interaction):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    player = get_player(interaction.guild_id)
    lines = []
    if player.current:
        lines.append(f"▶️ **再生中:** [{discord.utils.escape_markdown(player.current.title)}]({player.current.webpage_url})")
    else:
        lines.append("現在、再生中の曲はありません。")
    for i, track in enumerate(list(player.queue)[:15], 1):
        lines.append(f"{i}. [{discord.utils.escape_markdown(track.title)}]({track.webpage_url})")
    if len(player.queue) > 15:
        lines.append(f"ほか {len(player.queue) - 15} 曲")
    await interaction.response.send_message(embed=interaction_embed("再生キュー", "\n".join(lines)), ephemeral=True)


@bot.tree.command(name="skip", description="再生中の曲をスキップします")
async def skip(interaction: discord.Interaction):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    player = get_player(interaction.guild_id)
    if player.skip():
        await interaction.response.send_message("⏭️ スキップしました。", ephemeral=True)
    else:
        await interaction.response.send_message("スキップできる曲がありません。", ephemeral=True)


@bot.tree.command(name="pause", description="再生を一時停止します")
async def pause(interaction: discord.Interaction):
    player = get_player(interaction.guild_id or 0)
    if player.voice and player.voice.is_playing():
        player.voice.pause()
        await interaction.response.send_message("⏸️ 一時停止しました。", ephemeral=True)
    else:
        await interaction.response.send_message("再生中の曲がありません。", ephemeral=True)


@bot.tree.command(name="resume", description="一時停止した再生を再開します")
async def resume(interaction: discord.Interaction):
    player = get_player(interaction.guild_id or 0)
    if player.voice and player.voice.is_paused():
        player.voice.resume()
        await interaction.response.send_message("▶️ 再開しました。", ephemeral=True)
    else:
        await interaction.response.send_message("一時停止中の曲がありません。", ephemeral=True)


@bot.tree.command(name="stop", description="再生を停止してキューを空にします")
async def stop(interaction: discord.Interaction):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    player = get_player(interaction.guild_id)
    player.stop()
    await interaction.response.send_message("⏹️ 再生を停止し、キューを空にしました。", ephemeral=True)


@bot.tree.command(name="leave", description="BOTをボイスチャンネルから切断します")
async def leave(interaction: discord.Interaction):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    player = get_player(interaction.guild_id)
    player.stop()
    if player.voice and player.voice.is_connected():
        await player.voice.disconnect(force=True)
    player.voice = None
    await interaction.response.send_message("👋 ボイスチャンネルから退出しました。", ephemeral=True)


@bot.tree.command(name="direction", description="音源の方向を設定します")
@app_commands.describe(direction="音が聞こえる方向。移動中は開始位置になります")
@app_commands.choices(direction=[
    app_commands.Choice(name="前方", value="front"),
    app_commands.Choice(name="左前方", value="front_left"),
    app_commands.Choice(name="左", value="left"),
    app_commands.Choice(name="左後方", value="back_left"),
    app_commands.Choice(name="後方", value="back"),
    app_commands.Choice(name="右後方", value="back_right"),
    app_commands.Choice(name="右", value="right"),
    app_commands.Choice(name="右前方", value="front_right"),
    app_commands.Choice(name="上方", value="up"),
    app_commands.Choice(name="下方", value="down"),
])
async def direction(interaction: discord.Interaction, direction: app_commands.Choice[str]):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    player = get_player(interaction.guild_id)
    player.controller.set_direction(direction.value)
    mode = "移動の開始位置" if player.controller.mode == "moving" else "固定位置"
    await interaction.response.send_message(f"📍 音源の{mode}を **{DIRECTION_LABELS[direction.value]}** に設定しました。\n"
                                            f"先読み済みの音声があるため、再生中の反映には最大約{MAX_BUFFER_SECONDS:.0f}秒かかる場合があります。", ephemeral=True)


@bot.tree.command(name="motion", description="音源を固定するか移動させるかを設定します")
@app_commands.describe(mode="fixed=方向を固定、moving=音源を動かす")
@app_commands.choices(mode=[
    app_commands.Choice(name="固定", value="fixed"),
    app_commands.Choice(name="移動", value="moving"),
])
async def motion(interaction: discord.Interaction, mode: app_commands.Choice[str]):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    player = get_player(interaction.guild_id)
    player.controller.set_motion(mode.value)
    label = "固定" if mode.value == "fixed" else "移動"
    await interaction.response.send_message(f"音源の動作を **{label}** にしました。", ephemeral=True)


@bot.tree.command(name="movement", description="音源の動き方・速さ・回転方向を設定します")
@app_commands.describe(pattern="音源の動き方", speed="移動速度", rotation="円周移動の回転方向")
@app_commands.choices(pattern=[
    app_commands.Choice(name="円周", value="circle"),
    app_commands.Choice(name="左右往復", value="left_right"),
    app_commands.Choice(name="前後往復", value="front_back"),
    app_commands.Choice(name="ランダム", value="random"),
])
@app_commands.choices(speed=[
    app_commands.Choice(name="ゆっくり", value="slow"),
    app_commands.Choice(name="普通", value="normal"),
    app_commands.Choice(name="速い", value="fast"),
])
@app_commands.choices(rotation=[
    app_commands.Choice(name="時計回り", value="clockwise"),
    app_commands.Choice(name="反時計回り", value="counterclockwise"),
])
async def movement(
    interaction: discord.Interaction,
    pattern: app_commands.Choice[str],
    speed: app_commands.Choice[str],
    rotation: app_commands.Choice[str],
):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    player = get_player(interaction.guild_id)
    player.controller.set_movement(pattern.value, speed.value, rotation.value)
    await interaction.response.send_message(
        f"🌀 **{MOVEMENT_LABELS[pattern.value]}** / {SPEED_LABELS[speed.value]} / {ROTATION_LABELS[rotation.value]} に設定しました。\n"
        "音源の移動を有効にしました。", ephemeral=True)


@bot.tree.command(name="status", description="音源方向・移動設定とバッファ状態を表示します")
async def status(interaction: discord.Interaction):
    if not interaction.guild_id:
        await interaction.response.send_message("サーバー内で使用してください。", ephemeral=True)
        return
    player = get_player(interaction.guild_id)
    with player.controller.lock:
        direction_label = min(DIRECTION_PRESETS, key=lambda k: abs(wrap_azimuth(player.controller.azimuth - DIRECTION_PRESETS[k][0])) + abs(player.controller.elevation - DIRECTION_PRESETS[k][1]))
        mode = player.controller.mode
        pattern = player.controller.pattern
        speed = player.controller.speed
        rotation = player.controller.rotation
    buffer_info = "再生なし"
    if player.source:
        buffer_info = f"{player.source.frames.qsize() * FRAME_MS / 1000:.1f}秒"
    embed = interaction_embed("空間音響の状態", "\n".join([
        f"方向: **{DIRECTION_LABELS[direction_label]}**",
        f"動作: **{'移動' if mode == 'moving' else '固定'}**",
        f"動き方: **{MOVEMENT_LABELS.get(pattern, pattern)}**",
        f"速度: **{SPEED_LABELS.get(speed, speed)}**",
        f"回転: **{ROTATION_LABELS.get(rotation, rotation)}**",
        f"先読みバッファ: **{buffer_info}**",
    ]))
    await interaction.response.send_message(embed=embed, ephemeral=True)


async def main() -> None:
    token = (
        os.getenv("DISCORD_TOKEN")
        or os.getenv("DISCORD_BOT_TOKEN")
        or os.getenv("BOT_TOKEN")
    )
    if not token:
        raise SystemExit(
            "Discord Botトークンが未設定です。MrtCloudの環境変数/Secretsに "
            "DISCORD_TOKEN という名前で設定してください。"
        )
    try:
        log.info("FFmpeg executable: %s", get_ffmpeg_executable())
    except Exception as exc:
        log.warning("%s /play は音源再生時に失敗します。", exc)
    if not discord.opus.is_loaded():
        for opus_name in ("opus", "libopus.so.0", "libopus.so", "libopus-0.dll", "opus.dll", "libopus.dylib"):
            try:
                discord.opus.load_opus(opus_name)
                if discord.opus.is_loaded():
                    log.info("Loaded Opus library: %s", opus_name)
                    break
            except Exception:
                continue
    if not discord.opus.is_loaded():
        log.warning("libopusを読み込めません。PCM音声送信にはlibopusが必要です。")
    async with bot:
        await bot.start(token)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
