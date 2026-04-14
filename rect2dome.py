#!/usr/bin/env python3
"""rect2dome - Convert rectilinear images and videos to dome master format.

Requires:
  - ffmpeg / ffprobe (for video processing)

Usage as a script::

    python rect2dome.py input.mp4 output.mp4
    python rect2dome.py photo.jpg dome.tif

Usage as a module::

    from rect2dome import process_image, process_video
"""

import argparse
import logging
import mimetypes
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Supported extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts", ".m2ts"}


# ---------------------------------------------------------------------------
# Media analysis helpers
# ---------------------------------------------------------------------------


def detect_media_type(path: Path) -> str:
    """Return ``'image'`` or ``'video'`` for *path*.

    Detection is first attempted from the file extension; a MIME-type lookup
    is used as a fallback.

    :raises ValueError: when the media type cannot be determined.
    """
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("video/"):
            return "video"
    raise ValueError(f"Cannot determine media type for: {path}")


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Return *(width, height)* for an image or video file using ``ffprobe``."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = result.stdout.strip().split(",")
    return int(parts[0]), int(parts[1])


def get_video_framerate(path: Path) -> tuple[float, str]:
    """Return *(fps_float, fps_string)* for *path* using ``ffprobe``.

    The raw fraction string (e.g. ``"30000/1001"``) is also returned so that
    it can be forwarded verbatim to ``ffmpeg`` for lossless round-trips.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)
    return fps, fps_str


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

#: Filename pattern used for extracted frames (printf-style, for both Python
#: glob and ffmpeg ``-i`` arguments).
FRAME_PATTERN = "frame_%06d.tif"

#: Filename pattern used for processed/output frames.
PROCESSED_PATTERN = "out_%06d.tif"


def extract_frames(video_path: Path, frames_dir: Path) -> tuple[float, str]:
    """Extract every frame of *video_path* as a TIFF file into *frames_dir*.

    Returns *(fps_float, fps_string)* so that the original frame rate can be
    used when re-encoding the output video.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    fps, fps_str = get_video_framerate(video_path)
    output_pattern = frames_dir / FRAME_PATTERN
    logger.info("Extracting frames from %s at %.3f fps …", video_path, fps)
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), str(output_pattern)],
        check=True,
    )
    return fps, fps_str


def encode_video(
    frames_dir: Path,
    frame_pattern: str,
    output_path: Path,
    fps: str,
) -> None:
    """Encode processed frames in *frames_dir* into *output_path* using ``ffmpeg``.

    *fps* is passed verbatim to ``-framerate`` / ``-r`` so that fractional
    frame rates (e.g. ``"30000/1001"``) survive the round-trip intact.
    """
    input_pattern = frames_dir / frame_pattern
    logger.info("Encoding video to %s …", output_path)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", str(input_pattern),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(output_path),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Border helper
# ---------------------------------------------------------------------------


def add_border(img: "np.ndarray", border_size: int, border_color: str = "ffffff") -> "np.ndarray":
    """Add a solid-colour border around an image.

    The image is extended on all four sides by *border_size* pixels.  The
    border colour is specified as a 6-digit hex string (e.g. ``"ffffff"`` for
    white).  An optional leading ``#`` is stripped.

    :param img: Source image as a NumPy array.
    :param border_size: Width of the border in pixels on each side.  Zero or
        negative values return the original image unchanged.
    :param border_color: Hex colour string for the border (e.g. ``"ffffff"``).
    :returns: Image with the border added, or the original image when
        *border_size* <= 0.
    """
    if border_size <= 0:
        return img
    hex_color = border_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    if img.ndim == 3:
        color: int | tuple = (b, g, r, 255) if img.shape[2] == 4 else (b, g, r)
    else:
        color = int(0.299 * r + 0.587 * g + 0.114 * b)
    return cv2.copyMakeBorder(
        img,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=color,
    )


# ---------------------------------------------------------------------------
# Keyframe helpers
# ---------------------------------------------------------------------------


def parse_keyframe_spec(spec: str) -> dict:
    """Parse a keyframe specification string of the form ``K:f,l,ival,eval,T``.

    The fields are:

    * **K** – literal letter ``"K"`` marking this as a keyframe spec.
    * **f** – first frame of the keying sequence (0-based int).
    * **l** – last frame of the keying sequence (0-based int; negative values
      are resolved relative to the total frame count, e.g. ``-1`` = last
      frame).
    * **ival** – parameter value before (and at) frame *f*.
    * **eval** – parameter value at (and after) frame *l*.
    * **T** – interpolation type: ``"L"`` for linear, ``"S"`` for sinusoidal
      ease-in/ease-out.

    :param spec: Keyframe spec string.
    :returns: Dict with keys ``f``, ``l``, ``ival``, ``eval``, ``interp``.
    :raises ValueError: If the string is malformed.
    """
    if not spec.startswith("K:"):
        raise ValueError(f"Keyframe spec must start with 'K:': {spec!r}")
    parts = spec[2:].split(",", 4)
    if len(parts) != 5:
        raise ValueError(
            f"Keyframe spec must have exactly 5 comma-separated fields "
            f"(f,l,ival,eval,T): {spec!r}"
        )
    f_str, l_str, ival_str, eval_str, interp = parts
    interp = interp.strip()
    if interp not in ("L", "S"):
        raise ValueError(
            f"Interpolation type must be 'L' (linear) or 'S' (sinusoidal), "
            f"got {interp!r}: {spec!r}"
        )
    return {
        "f": int(f_str),
        "l": int(l_str),
        "ival": float(ival_str),
        "eval": float(eval_str),
        "interp": interp,
    }


def resolve_keyframable(param: "float | str", frame_idx: int, total_frames: int) -> float:
    """Return the numeric value of a potentially keyframed parameter at *frame_idx*.

    When *param* is already a :class:`float` (or :class:`int`), it is returned
    unchanged.  When *param* is a keyframe spec string (``"K:..."``), the
    value is interpolated according to the spec and the current frame index.

    :param param: Either a numeric value or a keyframe spec string.
    :param frame_idx: Zero-based index of the current frame.
    :param total_frames: Total number of frames (used to resolve negative ``l``
        values, e.g. ``-1`` becomes ``total_frames - 1``).
    :returns: Resolved float value.
    """
    if not isinstance(param, str):
        return float(param)
    spec = parse_keyframe_spec(param)
    f = spec["f"]
    l = spec["l"]
    ival: float = spec["ival"]
    eval_: float = spec["eval"]
    interp: str = spec["interp"]

    if l < 0:
        l = total_frames + l

    if frame_idx < f:
        return ival
    if frame_idx > l:
        return eval_
    if l <= f:
        # Degenerate or zero-length range: return ival at the boundary.
        return ival
    t = (frame_idx - f) / float(l - f)
    if interp == "L":
        return ival + t * (eval_ - ival)
    # interp == "S": sinusoidal ease-in/ease-out
    s = 0.5 * (1.0 - np.cos(np.pi * t))
    return ival + s * (eval_ - ival)


# ---------------------------------------------------------------------------
# OpenCV-based projection
# ---------------------------------------------------------------------------


def reproject_rectilinear_to_fisheye(
    src_img: "np.ndarray",
    output_size: int,
    lat_deg: float,
    lon_deg: float,
    hfov_deg: float,
    fisheye_fov_deg: float = 180.0,
    upright: bool = False,
) -> "np.ndarray":
    """Reproject a rectilinear image onto a square equidistant fisheye image.

    The function builds a pixel-coordinate map from every output fisheye pixel
    back to the corresponding source pixel and then calls :func:`cv2.remap` for
    efficient bilinear resampling.  Pixels that fall outside the rectilinear
    image (or behind the camera) are set to black.

    **Coordinate conventions**

    The fisheye coordinate frame is defined so that:

    * The centre of the fisheye image is the zenith / pole (θ = 0).
    * Moving down across the image corresponds to *longitude* = 0 (world +y).
    * Moving right across the image corresponds to *longitude* = 90 ° (world +x).
    * Latitude = 90 ° is the zenith; latitude = 0 ° is the horizon.

    The "up" direction in the rectilinear image is always oriented toward the
    fisheye zenith (the component of the zenith direction perpendicular to the
    optical axis).  When the optical axis already points at the zenith, the
    positive x-axis of the fisheye image is used as the "up" reference instead.

    **Placement modes**

    When *upright* is ``False`` (default), the image is placed tangentially to
    the sphere: the optical axis points directly at the point (``lat_deg``,
    ``lon_deg``), and that point corresponds to the centre of the source image.

    When *upright* is ``True``, the camera optical axis is kept strictly
    horizontal (lat = 0) and the image is shifted vertically so that its
    bottom row maps to the latitude given by ``lat_deg``.  This means the
    camera never "looks up" regardless of ``lat_deg``, so the image can never
    project across the zenith.  Horizontal lines in the source image remain
    parallel to latitude circles and the image appears undistorted (no
    keystone effect).

    :param src_img: Source rectilinear image as a NumPy array (H × W or H × W × C).
    :param output_size: Side length of the square output image in pixels.
    :param lat_deg: In the default (tangential) mode, the latitude (degrees) of
        the point on the unit sphere where the camera's optical axis is attached
        (centre of the image).  In upright mode, the latitude of the **lower
        boundary** of the reprojected image.  90 ° = zenith, 0 ° = horizon.
    :param lon_deg: Longitude (degrees) of the optical axis attachment point.
        0 ° = bottom of the fisheye image; 90 ° = right of the fisheye image.
    :param hfov_deg: Horizontal field of view of the rectilinear source image in
        degrees.
    :param fisheye_fov_deg: Total angular field of view of the output equidistant
        fisheye image in degrees.  Defaults to 180 °.
    :param upright: When ``True``, the camera is kept vertically upright and
        *lat_deg* defines the lower latitude boundary of the reprojected image
        rather than its centre.  Defaults to ``False``.
    :returns: Output fisheye image as a NumPy array of the same dtype as *src_img*
        with shape (*output_size*, *output_size*) or
        (*output_size*, *output_size*, C).
    """
    # Ensure the source image has an alpha channel so that pixels outside the
    # valid mapping region become fully transparent in the output.  Only 3-channel
    # (BGR) images are upgraded; grayscale and already-4-channel images are left
    # unchanged.
    if src_img.ndim == 3 and src_img.shape[2] == 3:
        if src_img.dtype.kind in ("u", "i"):
            alpha_val = np.iinfo(src_img.dtype).max
        else:
            alpha_val = 1.0
        alpha_ch = np.full(
            (*src_img.shape[:2], 1), alpha_val, dtype=src_img.dtype
        )
        src_img = np.concatenate([src_img, alpha_ch], axis=2)

    src_h, src_w = src_img.shape[:2]
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    hfov = np.radians(hfov_deg)
    fisheye_max_angle = np.radians(fisheye_fov_deg / 2.0)

    # Focal length of the rectilinear source image (in pixels).
    # tan(hfov/2) = (src_w/2) / focal → focal = (src_w/2) / tan(hfov/2)
    focal_rect = (src_w / 2.0) / np.tan(hfov / 2.0)

    if upright:
        # In upright mode the camera is kept perfectly horizontal (optical
        # axis at lat = 0).  Instead of tilting the axis, we apply a constant
        # vertical pixel offset to the source-image lookup so that the bottom
        # row of the source image maps to the requested lower latitude boundary.
        #
        # For a horizontal camera a direction at latitude φ (along the optical
        # meridian) projects to:
        #   src_y = focal_rect · (−tan φ) + src_h/2
        # The bottom row (src_y = src_h) therefore sits at φ = −vhalf (below
        # the horizon).  To shift the lower boundary to lat_deg we add:
        #   y_shift = src_h/2 + focal_rect · tan(lat_deg)
        # which moves that bottom row to φ = lat_deg without tilting the axis.
        # Because z_cam stays horizontal (z component = 0), directions toward
        # the zenith always have d_cam_z = 0 and are therefore always invalid
        # (black), satisfying the guarantee that the image never crosses the
        # zenith.
        y_shift = src_h / 2.0 + focal_rect * np.tan(lat)
        lat = 0.0
    else:
        y_shift = 0.0

    # Fisheye output image geometry.
    cx = output_size / 2.0
    cy = output_size / 2.0
    R = output_size / 2.0  # radius of the full fisheye circle

    # ------------------------------------------------------------------
    # Build the camera coordinate frame for the rectilinear projection.
    # Axes: x_cam = right in image, y_cam = down in image, z_cam = forward.
    # ------------------------------------------------------------------

    # Optical axis direction in world space (z_cam).
    # lon=0 corresponds to world +y (downward in the fisheye image); lon=90° to world +x (rightward).
    z_cam = np.array([
        np.cos(lat) * np.sin(lon),
        np.cos(lat) * np.cos(lon),
        np.sin(lat),
    ])

    # "Up" in the image corresponds to -y_cam, which we orient toward the
    # fisheye zenith (world +z) projected perpendicular to the optical axis.
    world_up = np.array([0.0, 0.0, 1.0])
    up_perp = world_up - np.dot(world_up, z_cam) * z_cam
    up_perp_norm = np.linalg.norm(up_perp)
    if up_perp_norm < 1e-9:
        # Optical axis is parallel to world_up; fall back to world +x as reference.
        ref = np.array([1.0, 0.0, 0.0])
        up_perp = ref - np.dot(ref, z_cam) * z_cam
        up_perp_norm = np.linalg.norm(up_perp)

    # y_cam points "down" in the image (away from zenith).
    y_cam = -up_perp / up_perp_norm

    # x_cam = y_cam × z_cam  (completing the right-handed frame: x × y = z  →  x = y × z)
    x_cam = np.cross(y_cam, z_cam)

    # ------------------------------------------------------------------
    # Build the inverse map: output (fisheye) pixel → source (rect) pixel.
    # ------------------------------------------------------------------

    # Pixel grid for the output image (row = y, col = x).
    py_out, px_out = np.mgrid[0:output_size, 0:output_size].astype(np.float64)

    # Offset from the fisheye image centre.
    dx = px_out - cx
    dy = py_out - cy
    r = np.sqrt(dx * dx + dy * dy)

    # Equidistant fisheye: zenith angle θ is proportional to radius.
    theta = (r / R) * fisheye_max_angle

    # Azimuth angle; handle the degenerate case r == 0 (exactly the centre pixel).
    with np.errstate(invalid="ignore"):
        cos_phi = np.where(r > 0.0, dx / r, 1.0)
        sin_phi = np.where(r > 0.0, dy / r, 0.0)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # 3-D direction vector in world space.
    d_x = sin_theta * cos_phi
    d_y = sin_theta * sin_phi
    d_z = cos_theta

    # Project direction into the rectilinear camera frame.
    d_cam_x = x_cam[0] * d_x + x_cam[1] * d_y + x_cam[2] * d_z
    d_cam_y = y_cam[0] * d_x + y_cam[1] * d_y + y_cam[2] * d_z
    d_cam_z = z_cam[0] * d_x + z_cam[1] * d_y + z_cam[2] * d_z

    # Perspective projection onto the rectilinear image plane (z_cam == 1).
    # Points behind the camera (d_cam_z <= 0) are invalid.
    valid = d_cam_z > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        src_x = np.where(valid, focal_rect * d_cam_x / d_cam_z + src_w / 2.0, -1.0)
        src_y = np.where(valid, focal_rect * d_cam_y / d_cam_z + src_h / 2.0 + y_shift, -1.0)

    # Mark coordinates outside the source image as invalid (remap will fill with 0).
    valid &= (src_x >= 0.0) & (src_x < src_w) & (src_y >= 0.0) & (src_y < src_h)
    map_x = np.where(valid, src_x, -1.0).astype(np.float32)
    map_y = np.where(valid, src_y, -1.0).astype(np.float32)

    return cv2.remap(
        src_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------------------------------------------------------------------------
# High-level processing functions
# ---------------------------------------------------------------------------


def process_image(
    input_path: Path,
    output_path: Path,
    output_size: int = 4096,
    lat_deg: "float | str" = 90.0,
    lon_deg: "float | str" = 0.0,
    hfov_deg: "float | str" = 90.0,
    fisheye_fov_deg: float = 180.0,
    upright: bool = False,
    border_size: int = 0,
    border_color: str = "ffffff",
) -> None:
    """Convert a single rectilinear image to an equidistant fisheye dome master.

    The image is read with OpenCV, reprojected via
    :func:`reproject_rectilinear_to_fisheye`, and written back with OpenCV.
    No intermediate files or external tools are required.

    :param input_path: Source image file.
    :param output_path: Destination file (TIFF recommended for lossless output).
    :param output_size: Side length of the (square) output image in pixels.
    :param lat_deg: Latitude of the camera's optical axis (90 ° = zenith).
        Can be a float or a keyframe spec string ``"K:f,l,ival,eval,T"``.
    :param lon_deg: Longitude of the camera's optical axis (0 ° = bottom).
        Can be a float or a keyframe spec string.
    :param hfov_deg: Horizontal field of view of the source image in degrees.
        Can be a float or a keyframe spec string.
    :param fisheye_fov_deg: Total angular FOV of the output fisheye in degrees.
    :param upright: When ``True``, the camera is kept horizontally upright and
        *lat_deg* defines the lower latitude boundary of the reprojected image.
    :param border_size: Width in pixels of the border to add around the input
        image on each side before reprojection.  0 means no border.
    :param border_color: Hex colour string for the border (e.g. ``"ffffff"``).
    """
    src_img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if src_img is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")
    if border_size > 0:
        src_img = add_border(src_img, border_size, border_color)
    logger.info(
        "Processing image %s (%dx%d) …", input_path, src_img.shape[1], src_img.shape[0]
    )
    result = reproject_rectilinear_to_fisheye(
        src_img,
        output_size,
        resolve_keyframable(lat_deg, 0, 1),
        resolve_keyframable(lon_deg, 0, 1),
        resolve_keyframable(hfov_deg, 0, 1),
        fisheye_fov_deg,
        upright,
    )
    cv2.imwrite(str(output_path), result)
    logger.info("Output written to %s", output_path)


def process_video(
    input_path: Path,
    output_path: Path,
    output_size: int = 4096,
    lat_deg: "float | str" = 90.0,
    lon_deg: "float | str" = 0.0,
    hfov_deg: "float | str" = 90.0,
    fisheye_fov_deg: float = 180.0,
    upright: bool = False,
    work_dir: Path | None = None,
    border_size: int = 0,
    border_color: str = "ffffff",
) -> None:
    """Convert every frame of a video to an equidistant fisheye dome master.

    Frames are extracted with ``ffmpeg``, each frame is reprojected via
    :func:`reproject_rectilinear_to_fisheye`, and the results are re-encoded
    into *output_path*.  Processing a single image uses the same reprojection
    logic via :func:`process_image`, just without the extraction and encoding
    steps.

    A temporary directory is used when *work_dir* is ``None``.  The caller is
    responsible for cleaning up an explicitly supplied *work_dir*.

    :param input_path: Source video file.
    :param output_path: Destination video file.
    :param output_size: Side length of the (square) output frames in pixels.
    :param lat_deg: Latitude of the camera's optical axis (90 ° = zenith).
        Can be a float or a keyframe spec string ``"K:f,l,ival,eval,T"``.
    :param lon_deg: Longitude of the camera's optical axis (0 ° = bottom).
        Can be a float or a keyframe spec string.
    :param hfov_deg: Horizontal field of view of the source frames in degrees.
        Can be a float or a keyframe spec string.
    :param fisheye_fov_deg: Total angular FOV of the output fisheye in degrees.
    :param upright: When ``True``, the camera is kept horizontally upright and
        *lat_deg* defines the lower latitude boundary of the reprojected frames.
    :param work_dir: Optional directory for intermediate files.
    :param border_size: Width in pixels of the border to add around each input
        frame on each side before reprojection.  0 means no border.
    :param border_color: Hex colour string for the border (e.g. ``"ffffff"``).
    """
    managed = work_dir is None
    if managed:
        work_dir = Path(tempfile.mkdtemp(prefix="rect2dome_"))
    try:
        frames_dir = work_dir / "frames"
        processed_dir = work_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        fps, fps_str = extract_frames(input_path, frames_dir)

        frame_files = sorted(frames_dir.glob("frame_*.tif"))
        if not frame_files:
            raise FileNotFoundError(f"No frames were extracted from {input_path}")

        total_frames = len(frame_files)
        logger.info("Processing %d frame(s) …", total_frames)
        for idx, frame_path in enumerate(frame_files):
            logger.info(
                "  Frame %d/%d: %s", idx + 1, total_frames, frame_path.name
            )
            src_img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if src_img is None:
                raise FileNotFoundError(f"Failed to read frame: {frame_path}")
            if border_size > 0:
                src_img = add_border(src_img, border_size, border_color)
            result = reproject_rectilinear_to_fisheye(
                src_img,
                output_size,
                resolve_keyframable(lat_deg, idx, total_frames),
                resolve_keyframable(lon_deg, idx, total_frames),
                resolve_keyframable(hfov_deg, idx, total_frames),
                fisheye_fov_deg,
                upright,
            )
            cv2.imwrite(str(processed_dir / f"out_{idx:06d}.tif"), result)

        encode_video(processed_dir, PROCESSED_PATTERN, output_path, fps_str)
        logger.info("Output video written to %s", output_path)
    finally:
        if managed:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _keyframable_float(value: str) -> "float | str":
    """argparse type that accepts either a plain float or a keyframe spec string.

    A plain numeric string is converted to :class:`float`.  A string starting
    with ``"K:"`` is returned as-is (it will be parsed at processing time via
    :func:`parse_keyframe_spec`).
    """
    try:
        return float(value)
    except ValueError:
        if value.startswith("K:"):
            # Eagerly validate the spec so the user gets an immediate error.
            parse_keyframe_spec(value)
            return value
        raise argparse.ArgumentTypeError(
            f"Expected a number or a keyframe spec (K:f,l,ival,eval,T), got {value!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``rect2dome`` CLI."""
    parser = argparse.ArgumentParser(
        prog="rect2dome",
        description=(
            "Convert rectilinear images and videos to dome master format.\n"
            "\n"
            "Requires ffmpeg to be installed for video processing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input image or video file.")
    parser.add_argument(
        "output",
        nargs="?",
        help=(
            "Output file path.  Defaults to <input>_dome.<ext> for images "
            "and <input>_dome.mp4 for videos."
        ),
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=4096,
        metavar="PIXELS",
        help="Side length of the square output image/frame in pixels.",
    )
    parser.add_argument(
        "--lat",
        type=_keyframable_float,
        default=90.0,
        metavar="DEGREES",
        help=(
            "Latitude of the camera's optical axis in degrees (90 = zenith). "
            "Accepts a plain number or a keyframe spec K:f,l,ival,eval,T."
        ),
    )
    parser.add_argument(
        "--lon",
        type=_keyframable_float,
        default=0.0,
        metavar="DEGREES",
        help=(
            "Longitude of the camera's optical axis in degrees (0 = bottom of dome). "
            "Accepts a plain number or a keyframe spec K:f,l,ival,eval,T."
        ),
    )
    parser.add_argument(
        "--hfov",
        type=_keyframable_float,
        default=90.0,
        metavar="DEGREES",
        help=(
            "Horizontal field of view of the input image in degrees. "
            "Accepts a plain number or a keyframe spec K:f,l,ival,eval,T."
        ),
    )
    parser.add_argument(
        "--output-fov",
        type=float,
        default=180.0,
        metavar="DEGREES",
        help="Total angular field of view of the output fisheye image in degrees.",
    )
    parser.add_argument(
        "--border-size",
        type=int,
        default=0,
        metavar="PIXELS",
        help=(
            "Width in pixels of a solid-colour border to add around each input "
            "frame on all sides before reprojection.  0 means no border."
        ),
    )
    parser.add_argument(
        "--border-color",
        default="ffffff",
        metavar="RRGGBB",
        help="Hex colour code (without #) for the border, e.g. 'ffffff' for white.",
    )
    parser.add_argument(
        "--upright",
        action="store_true",
        help=(
            "Keep the camera horizontally upright; --lat then defines the lower "
            "latitude boundary of the reprojected image instead of its centre."
        ),
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate frame files after video processing.",
    )
    parser.add_argument(
        "--work-dir",
        metavar="DIR",
        help="Directory for intermediate video frames (default: system temp dir).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) output.",
    )
    return parser


def _default_output_path(input_path: Path, media_type: str) -> Path:
    """Return a sensible default output path derived from *input_path*."""
    if media_type == "video":
        return input_path.with_name(input_path.stem + "_dome.mp4")
    return input_path.with_name(input_path.stem + "_dome" + input_path.suffix)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``rect2dome`` command-line interface.

    :returns: Exit code (0 = success, non-zero = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    try:
        media_type = detect_media_type(input_path)
    except ValueError as exc:
        parser.error(str(exc))

    output_path = (
        Path(args.output) if args.output else _default_output_path(input_path, media_type)
    )

    logger.info("Input : %s (%s)", input_path, media_type)
    logger.info("Output: %s", output_path)

    common_kwargs = dict(
        output_size=args.output_size,
        lat_deg=args.lat,
        lon_deg=args.lon,
        hfov_deg=args.hfov,
        fisheye_fov_deg=args.output_fov,
        upright=args.upright,
        border_size=args.border_size,
        border_color=args.border_color,
    )

    try:
        if media_type == "image":
            process_image(
                input_path=input_path,
                output_path=output_path,
                **common_kwargs,
            )
        else:
            # Resolve working directory for video frame extraction.
            # When --keep-frames is set a named temp dir is created and kept.
            # When --work-dir is given that directory is used and never cleaned up.
            # Otherwise process_video manages its own temporary directory.
            if args.work_dir:
                work_dir: Path | None = Path(args.work_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
            elif args.keep_frames:
                work_dir = Path(tempfile.mkdtemp(prefix="rect2dome_"))
                logger.info("Intermediate files will be kept in %s", work_dir)
            else:
                work_dir = None
            process_video(
                input_path=input_path,
                output_path=output_path,
                work_dir=work_dir,
                **common_kwargs,
            )
    except subprocess.CalledProcessError as exc:
        logger.error("External tool failed (exit code %d): %s", exc.returncode, exc.cmd)
        return 1
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
