#!/usr/bin/env python3
"""rect2dome - Convert rectangular/equirectangular images and videos to dome master format.

Requires:
  - Hugin (specifically the ``nona`` tool)
  - ffmpeg / ffprobe (for video processing)

Usage as a script::

    python rect2dome.py input.mp4 output.mp4
    python rect2dome.py panorama.jpg dome.tif

Usage as a module::

    from rect2dome import process_image, process_video
"""

import argparse
import logging
import mimetypes
import os
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

# Panotools projection constants (for documentation purposes)
PROJECTION_RECTILINEAR = 0
PROJECTION_CYLINDRICAL = 1
PROJECTION_EQUIRECTANGULAR = 2
PROJECTION_FISHEYE_EQUIDISTANT = 3
PROJECTION_FISHEYE_ORTHOGRAPHIC = 10


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
# Panotools / nona helpers
# ---------------------------------------------------------------------------


def create_pto_file(
    image_path: Path,
    pto_path: Path,
    src_width: int,
    src_height: int,
    output_width: int,
    output_height: int,
    input_projection: int = PROJECTION_EQUIRECTANGULAR,
    output_projection: int = PROJECTION_FISHEYE_EQUIDISTANT,
    hfov: float = 360.0,
    output_fov: float = 180.0,
) -> Path:
    """Write a minimal panotools PTO project file and return its path.

    The project is configured so that *nona* reprojects the single input image
    from *input_projection* to *output_projection*.

    :param image_path: Absolute path to the source image.
    :param pto_path: Where to write the ``.pto`` file.
    :param src_width: Width of the source image in pixels.
    :param src_height: Height of the source image in pixels.
    :param output_width: Width of the output image in pixels.
    :param output_height: Height of the output image in pixels.
    :param input_projection: Panotools projection code for the input image.
    :param output_projection: Panotools projection code for the output image.
    :param hfov: Horizontal field of view of the input image in degrees.
    :param output_fov: Field of view of the output image in degrees.
    :returns: *pto_path* (unchanged).
    """
    pto_content = (
        "# Hugin project file - generated by rect2dome\n"
        f"p f{output_projection} w{output_width} h{output_height}"
        f' v{output_fov} n"TIFF_m c:NONE r:CROP"\n'
        "m g1 i0\n"
        f"i f{input_projection} w{src_width} h{src_height}"
        f" v{hfov:.6f} r0 p0 y0"
        " TrX0 TrY0 TrZ0 j0 a0.0 b0.0 c0.0 d0 e0 g0 t0"
        f' Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 Tpy0 Tpp0 n"{image_path}"\n'
        "v\n"
    )
    pto_path.write_text(pto_content)
    return pto_path


def run_nona(pto_path: Path, output_prefix: Path) -> Path:
    """Run ``nona`` with *pto_path* and write output to *output_prefix*.

    ``nona`` appends ``0000.tif`` to the prefix for a single-image project,
    so this function returns that expected output path.

    :returns: Path of the TIFF file produced by ``nona``.
    """
    logger.debug("Running nona: nona -o %s %s", output_prefix, pto_path)
    subprocess.run(
        ["nona", "-o", str(output_prefix), str(pto_path)],
        check=True,
    )
    nona_output = Path(str(output_prefix) + "0000.tif")
    if not nona_output.exists():
        raise FileNotFoundError(
            f"nona did not produce the expected output file: {nona_output}"
        )
    return nona_output


# ---------------------------------------------------------------------------
# OpenCV-based projection (nona-free alternative)
# ---------------------------------------------------------------------------


def reproject_rectilinear_to_fisheye(
    src_img: "np.ndarray",
    output_size: int,
    lat_deg: float,
    lon_deg: float,
    hfov_deg: float,
    fisheye_fov_deg: float = 180.0,
) -> "np.ndarray":
    """Reproject a rectilinear image onto a square equidistant fisheye image.

    The function builds a pixel-coordinate map from every output fisheye pixel
    back to the corresponding source pixel and then calls :func:`cv2.remap` for
    efficient bilinear resampling.  Pixels that fall outside the rectilinear
    image (or behind the camera) are set to black.

    **Coordinate conventions**

    The fisheye coordinate frame is defined so that:

    * The centre of the fisheye image is the zenith / pole (θ = 0).
    * Moving right across the image corresponds to *longitude* = 0 (world +x).
    * Moving down across the image corresponds to *longitude* = 90 ° (world +y).
    * Latitude = 90 ° is the zenith; latitude = 0 ° is the horizon.

    The "up" direction in the rectilinear image is always oriented toward the
    fisheye zenith (the component of the zenith direction perpendicular to the
    optical axis).  When the optical axis already points at the zenith, the
    positive x-axis of the fisheye image is used as the "up" reference instead.

    :param src_img: Source rectilinear image as a NumPy array (H × W or H × W × C).
    :param output_size: Side length of the square output image in pixels.
    :param lat_deg: Latitude (degrees) of the point on the unit sphere where the
        rectilinear camera's optical axis is attached.  90 ° = zenith (centre of
        fisheye), 0 ° = horizon (edge of a 180 ° fisheye).
    :param lon_deg: Longitude (degrees) of the optical axis attachment point.
        0 ° = right of the fisheye image; 90 ° = bottom of the fisheye image.
    :param hfov_deg: Horizontal field of view of the rectilinear source image in
        degrees.
    :param fisheye_fov_deg: Total angular field of view of the output equidistant
        fisheye image in degrees.  Defaults to 180 °.
    :returns: Output fisheye image as a NumPy array of the same dtype as *src_img*
        with shape (*output_size*, *output_size*) or
        (*output_size*, *output_size*, C).
    """
    src_h, src_w = src_img.shape[:2]
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    hfov = np.radians(hfov_deg)
    fisheye_max_angle = np.radians(fisheye_fov_deg / 2.0)

    # Focal length of the rectilinear source image (in pixels).
    # tan(hfov/2) = (src_w/2) / focal → focal = (src_w/2) / tan(hfov/2)
    focal_rect = (src_w / 2.0) / np.tan(hfov / 2.0)

    # Fisheye output image geometry.
    cx = output_size / 2.0
    cy = output_size / 2.0
    R = output_size / 2.0  # radius of the full fisheye circle

    # ------------------------------------------------------------------
    # Build the camera coordinate frame for the rectilinear projection.
    # Axes: x_cam = right in image, y_cam = down in image, z_cam = forward.
    # ------------------------------------------------------------------

    # Optical axis direction in world space (z_cam).
    z_cam = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
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

    # x_cam = y_cam × z_cam  (right-hand rule: x × y = z  →  x = y × z)
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
        src_y = np.where(valid, focal_rect * d_cam_y / d_cam_z + src_h / 2.0, -1.0)

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
    output_size: int = 2048,
    input_projection: int = PROJECTION_EQUIRECTANGULAR,
    output_projection: int = PROJECTION_FISHEYE_EQUIDISTANT,
    hfov: float = 360.0,
    output_fov: float = 180.0,
    work_dir: Path | None = None,
) -> None:
    """Convert a single image from *input_projection* to *output_projection*.

    A temporary directory is used when *work_dir* is ``None``.  The caller is
    responsible for cleaning up an explicitly supplied *work_dir*.

    :param input_path: Source image file.
    :param output_path: Destination file (TIFF recommended).
    :param output_size: Side length of the (square) output image in pixels.
    :param input_projection: Panotools projection code for the source image.
    :param output_projection: Panotools projection code for the output image.
    :param hfov: Horizontal field of view of the source image in degrees.
    :param output_fov: Field of view of the output image in degrees.
    :param work_dir: Optional directory for intermediate files.
    """
    managed = work_dir is None
    if managed:
        work_dir = Path(tempfile.mkdtemp(prefix="rect2dome_"))
    try:
        src_width, src_height = get_image_dimensions(input_path)
        logger.info("Processing image %s (%dx%d) …", input_path, src_width, src_height)

        pto_path = create_pto_file(
            image_path=input_path.resolve(),
            pto_path=work_dir / "project.pto",
            src_width=src_width,
            src_height=src_height,
            output_width=output_size,
            output_height=output_size,
            input_projection=input_projection,
            output_projection=output_projection,
            hfov=hfov,
            output_fov=output_fov,
        )

        nona_output = run_nona(pto_path, work_dir / "output")
        shutil.move(str(nona_output), str(output_path))
        logger.info("Output written to %s", output_path)
    finally:
        if managed:
            shutil.rmtree(work_dir, ignore_errors=True)


def process_video(
    input_path: Path,
    output_path: Path,
    output_size: int = 2048,
    input_projection: int = PROJECTION_EQUIRECTANGULAR,
    output_projection: int = PROJECTION_FISHEYE_EQUIDISTANT,
    hfov: float = 360.0,
    output_fov: float = 180.0,
    work_dir: Path | None = None,
) -> None:
    """Convert every frame of a video from *input_projection* to *output_projection*.

    Frames are extracted with ``ffmpeg``, each frame is processed by ``nona``,
    and the results are re-encoded into *output_path*.

    A temporary directory is used when *work_dir* is ``None``.  The caller is
    responsible for cleaning up an explicitly supplied *work_dir*.

    :param input_path: Source video file.
    :param output_path: Destination video file.
    :param output_size: Side length of the (square) output frames in pixels.
    :param input_projection: Panotools projection code for the source frames.
    :param output_projection: Panotools projection code for the output frames.
    :param hfov: Horizontal field of view of the source frames in degrees.
    :param output_fov: Field of view of the output frames in degrees.
    :param work_dir: Optional directory for intermediate files.
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

        logger.info("Processing %d frame(s) …", len(frame_files))
        for idx, frame_path in enumerate(frame_files):
            logger.info(
                "  Frame %d/%d: %s", idx + 1, len(frame_files), frame_path.name
            )
            src_width, src_height = get_image_dimensions(frame_path)

            pto_path = create_pto_file(
                image_path=frame_path.resolve(),
                pto_path=work_dir / f"project_{idx:06d}.pto",
                src_width=src_width,
                src_height=src_height,
                output_width=output_size,
                output_height=output_size,
                input_projection=input_projection,
                output_projection=output_projection,
                hfov=hfov,
                output_fov=output_fov,
            )

            nona_output = run_nona(pto_path, processed_dir / f"frame_{idx:06d}")
            # Rename to a clean sequential pattern for ffmpeg
            final_frame = processed_dir / f"out_{idx:06d}.tif"
            shutil.move(str(nona_output), str(final_frame))

        encode_video(processed_dir, PROCESSED_PATTERN, output_path, fps_str)
        logger.info("Output video written to %s", output_path)
    finally:
        if managed:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``rect2dome`` CLI."""
    parser = argparse.ArgumentParser(
        prog="rect2dome",
        description=(
            "Convert rectangular/equirectangular images and videos to dome master format.\n"
            "\n"
            "Requires Hugin (nona) and ffmpeg to be installed."
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
        default=2048,
        metavar="PIXELS",
        help="Side length of the square output image/frame in pixels.",
    )
    parser.add_argument(
        "--input-projection",
        type=int,
        default=PROJECTION_EQUIRECTANGULAR,
        metavar="CODE",
        help=(
            "Panotools projection code for the input "
            "(0=rectilinear, 2=equirectangular, 3=fisheye)."
        ),
    )
    parser.add_argument(
        "--output-projection",
        type=int,
        default=PROJECTION_FISHEYE_EQUIDISTANT,
        metavar="CODE",
        help=(
            "Panotools projection code for the output "
            "(3=equidistant fisheye / domemaster, 10=orthographic fisheye)."
        ),
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=360.0,
        metavar="DEGREES",
        help="Horizontal field of view of the input image in degrees.",
    )
    parser.add_argument(
        "--output-fov",
        type=float,
        default=180.0,
        metavar="DEGREES",
        help="Field of view of the output image in degrees.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate frame files after video processing.",
    )
    parser.add_argument(
        "--work-dir",
        metavar="DIR",
        help="Directory for intermediate files (default: system temp dir).",
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

    # Resolve working directory.
    # When --keep-frames is set a named temp dir is created and left in place.
    # When --work-dir is given that directory is used and never cleaned up.
    # Otherwise each process_* helper manages its own temporary directory.
    if args.work_dir:
        work_dir: Path | None = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    elif args.keep_frames:
        work_dir = Path(tempfile.mkdtemp(prefix="rect2dome_"))
        logger.info("Intermediate files will be kept in %s", work_dir)
    else:
        work_dir = None  # each process_* function creates and cleans its own temp dir

    try:
        if media_type == "image":
            process_image(
                input_path=input_path,
                output_path=output_path,
                output_size=args.output_size,
                input_projection=args.input_projection,
                output_projection=args.output_projection,
                hfov=args.hfov,
                output_fov=args.output_fov,
                work_dir=work_dir,
            )
        else:
            process_video(
                input_path=input_path,
                output_path=output_path,
                output_size=args.output_size,
                input_projection=args.input_projection,
                output_projection=args.output_projection,
                hfov=args.hfov,
                output_fov=args.output_fov,
                work_dir=work_dir,
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
