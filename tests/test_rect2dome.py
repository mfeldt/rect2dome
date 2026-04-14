"""Tests for rect2dome.py

All tests that exercise external tools (ffmpeg) mock the subprocess
calls so that the test suite can run without those tools installed.
"""

import subprocess
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

import rect2dome
from rect2dome import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    _default_output_path,
    add_border,
    build_parser,
    detect_media_type,
    encode_video,
    extract_frames,
    get_image_dimensions,
    get_video_framerate,
    main,
    parse_keyframe_spec,
    process_image,
    process_video,
    reproject_rectilinear_to_fisheye,
    resolve_keyframable,
)


# ---------------------------------------------------------------------------
# detect_media_type
# ---------------------------------------------------------------------------


class TestDetectMediaType:
    @pytest.mark.parametrize("ext", sorted(IMAGE_EXTENSIONS))
    def test_image_extensions(self, tmp_path, ext):
        p = tmp_path / f"file{ext}"
        p.touch()
        assert detect_media_type(p) == "image"

    @pytest.mark.parametrize("ext", sorted(VIDEO_EXTENSIONS))
    def test_video_extensions(self, tmp_path, ext):
        p = tmp_path / f"file{ext}"
        p.touch()
        assert detect_media_type(p) == "video"

    def test_unknown_extension_raises(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.touch()
        with pytest.raises(ValueError, match="Cannot determine media type"):
            detect_media_type(p)

    def test_fallback_to_mime_image(self, tmp_path):
        # .jpeg is already in IMAGE_EXTENSIONS; use a mime-only path via mock
        p = tmp_path / "picture.pnm"
        p.touch()
        with mock.patch("mimetypes.guess_type", return_value=("image/x-portable-anymap", None)):
            assert detect_media_type(p) == "image"

    def test_fallback_to_mime_video(self, tmp_path):
        p = tmp_path / "clip.ogv"
        p.touch()
        with mock.patch("mimetypes.guess_type", return_value=("video/ogg", None)):
            assert detect_media_type(p) == "video"


# ---------------------------------------------------------------------------
# get_image_dimensions
# ---------------------------------------------------------------------------


class TestGetImageDimensions:
    def test_returns_width_height(self, tmp_path):
        img = tmp_path / "test.tif"
        img.touch()
        mock_result = mock.MagicMock()
        mock_result.stdout = "1920,1080\n"
        with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
            w, h = get_image_dimensions(img)
        assert w == 1920
        assert h == 1080
        cmd = mock_run.call_args[0][0]
        assert "ffprobe" in cmd
        assert str(img) in cmd

    def test_propagates_subprocess_error(self, tmp_path):
        img = tmp_path / "bad.tif"
        img.touch()
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffprobe"),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                get_image_dimensions(img)


# ---------------------------------------------------------------------------
# get_video_framerate
# ---------------------------------------------------------------------------


class TestGetVideoFramerate:
    def test_integer_fps(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        mock_result = mock.MagicMock()
        mock_result.stdout = "25\n"
        with mock.patch("subprocess.run", return_value=mock_result):
            fps, fps_str = get_video_framerate(video)
        assert fps == pytest.approx(25.0)
        assert fps_str == "25"

    def test_fractional_fps(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        mock_result = mock.MagicMock()
        mock_result.stdout = "30000/1001\n"
        with mock.patch("subprocess.run", return_value=mock_result):
            fps, fps_str = get_video_framerate(video)
        assert fps == pytest.approx(30000 / 1001)
        assert fps_str == "30000/1001"


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------


class TestExtractFrames:
    def test_calls_ffmpeg_and_returns_fps(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.touch()
        frames_dir = tmp_path / "frames"

        fps_result = mock.MagicMock()
        fps_result.stdout = "24\n"
        ffmpeg_result = mock.MagicMock()

        with mock.patch("subprocess.run", side_effect=[fps_result, ffmpeg_result]) as mock_run:
            fps, fps_str = extract_frames(video, frames_dir)

        assert fps == pytest.approx(24.0)
        assert fps_str == "24"
        assert frames_dir.exists()
        # Second call should be ffmpeg
        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        assert "ffmpeg" in ffmpeg_cmd


# ---------------------------------------------------------------------------
# encode_video
# ---------------------------------------------------------------------------


class TestEncodeVideo:
    def test_calls_ffmpeg(self, tmp_path):
        frames_dir = tmp_path / "processed"
        frames_dir.mkdir()
        output = tmp_path / "out.mp4"

        with mock.patch("subprocess.run") as mock_run:
            encode_video(frames_dir, "out_%06d.tif", output, "30000/1001")

        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert str(output) in cmd
        assert "30000/1001" in cmd


# ---------------------------------------------------------------------------
# process_image (uses real small images, no subprocess calls)
# ---------------------------------------------------------------------------


class TestProcessImage:
    def _make_src(self, tmp_path, suffix=".tif"):
        """Create a small valid source image and return its path."""
        src = tmp_path / f"source{suffix}"
        img = np.full((8, 16, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(src), img)
        return src

    def _run(self, tmp_path, output_size=32, **kwargs):
        src = self._make_src(tmp_path)
        dst = tmp_path / "dome.tif"
        process_image(input_path=src, output_path=dst, output_size=output_size, **kwargs)
        assert dst.exists()
        return dst

    def test_basic(self, tmp_path):
        self._run(tmp_path)

    def test_custom_output_size(self, tmp_path):
        dst = self._run(tmp_path, output_size=64)
        out = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
        assert out.shape == (64, 64, 4)

    def test_raises_for_unreadable_image(self, tmp_path):
        src = tmp_path / "empty.tif"
        src.touch()  # zero-byte file → cv2.imread returns None
        with pytest.raises(FileNotFoundError, match="Failed to read image"):
            process_image(input_path=src, output_path=tmp_path / "out.tif")


# ---------------------------------------------------------------------------
# process_video (integration-level, ffmpeg mocked)
# ---------------------------------------------------------------------------


class TestProcessVideo:
    def _make_frame(self, path: Path) -> None:
        """Write a minimal valid TIFF frame to *path*."""
        img = np.full((8, 16, 3), 64, dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def test_basic(self, tmp_path):
        src = tmp_path / "clip.mp4"
        src.touch()
        dst = tmp_path / "clip_dome.mp4"

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        frames_dir = work_dir / "frames"

        fps_result = mock.MagicMock()
        fps_result.stdout = "25\n"
        ffmpeg_extract_result = mock.MagicMock()
        ffmpeg_encode_result = mock.MagicMock()

        def fake_run(cmd, **kw):
            if "ffprobe" in cmd and "r_frame_rate" in cmd:
                return fps_result
            if "ffmpeg" in cmd and "-framerate" not in cmd:
                # Frame extraction: create real (readable) fake frames
                frames_dir.mkdir(parents=True, exist_ok=True)
                self._make_frame(frames_dir / "frame_000001.tif")
                self._make_frame(frames_dir / "frame_000002.tif")
                return ffmpeg_extract_result
            if "ffmpeg" in cmd and "-framerate" in cmd:
                return ffmpeg_encode_result
            return mock.MagicMock()

        with mock.patch("subprocess.run", side_effect=fake_run):
            process_video(
                input_path=src,
                output_path=dst,
                work_dir=work_dir,
                output_size=16,
            )


# ---------------------------------------------------------------------------
# CLI – argument parsing and default output path
# ---------------------------------------------------------------------------


class TestDefaultOutputPath:
    def test_image(self):
        p = _default_output_path(Path("pano.jpg"), "image")
        assert p == Path("pano_dome.jpg")

    def test_video(self):
        p = _default_output_path(Path("clip.mp4"), "video")
        assert p == Path("clip_dome.mp4")

    def test_image_preserves_extension(self):
        p = _default_output_path(Path("shot.tiff"), "image")
        assert p.suffix == ".tiff"


class TestBuildParser:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["input.mp4"])
        assert args.output_size == 4096
        assert args.lat == 90.0
        assert args.lon == 0.0
        assert args.hfov == 90.0
        assert args.output_fov == 180.0
        assert not args.upright
        assert not args.keep_frames
        assert not args.verbose
        assert args.border_size == 0
        assert args.border_color == "ffffff"

    def test_explicit_output(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg", "out.tif"])
        assert args.output == "out.tif"

    def test_output_size_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg", "--output-size", "4096"])
        assert args.output_size == 4096

    def test_version(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--version"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert rect2dome.__version__ in out


class TestMain:
    def test_missing_input_exits_nonzero(self, tmp_path):
        # parser.error() calls sys.exit(2), so SystemExit is expected
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path / "nonexistent.jpg")])
        assert exc.value.code != 0

    def test_unknown_extension_exits_nonzero(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.touch()
        # parser.error() calls sys.exit(2), so SystemExit is expected
        with pytest.raises(SystemExit) as exc:
            main([str(p)])
        assert exc.value.code != 0

    def test_image_processing(self, tmp_path):
        src = tmp_path / "pano.jpg"
        img = np.full((8, 16, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(src), img)
        dst = tmp_path / "pano_dome.jpg"

        rc = main([str(src), str(dst), "--output-size", "16"])

        assert rc == 0
        assert dst.exists()

    def test_unreadable_image_returns_1(self, tmp_path):
        src = tmp_path / "pano.jpg"
        src.touch()  # zero-byte → cv2.imread returns None → FileNotFoundError
        dst = tmp_path / "pano_dome.jpg"

        rc = main([str(src), str(dst)])

        assert rc == 1


# ---------------------------------------------------------------------------
# reproject_rectilinear_to_fisheye
# ---------------------------------------------------------------------------


class TestReprojectRectilinearToFisheye:
    """Tests for the OpenCV-based rectilinear → equidistant fisheye reprojection."""

    def _solid_image(self, height=480, width=640, channels=3, value=200):
        """Return a solid-colour uint8 image."""
        img = np.full((height, width, channels), value, dtype=np.uint8)
        return img

    # ------------------------------------------------------------------
    # Output shape and dtype
    # ------------------------------------------------------------------

    def test_output_shape_square(self):
        src = self._solid_image()
        out = reproject_rectilinear_to_fisheye(src, 256, lat_deg=90.0, lon_deg=0.0, hfov_deg=90.0)
        assert out.shape == (256, 256, 4)

    def test_output_dtype_preserved(self):
        src = self._solid_image().astype(np.float32)
        out = reproject_rectilinear_to_fisheye(src, 128, lat_deg=90.0, lon_deg=0.0, hfov_deg=90.0)
        assert out.dtype == np.float32

    def test_output_shape_grayscale(self):
        src = np.full((480, 640), 128, dtype=np.uint8)
        out = reproject_rectilinear_to_fisheye(src, 256, lat_deg=90.0, lon_deg=0.0, hfov_deg=90.0)
        assert out.shape == (256, 256)

    # ------------------------------------------------------------------
    # Pixels outside the fisheye circle should be black (no contribution
    # from a source pointed at the zenith for a 180 ° output FOV).
    # ------------------------------------------------------------------

    def test_corners_are_black_for_small_hfov(self):
        """Corners of the output image are outside the fisheye circle and must be 0."""
        src = self._solid_image(value=255)
        out = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=90.0, lon_deg=0.0, hfov_deg=10.0
        )
        # Corners are well outside any visible region.
        for corner in [(0, 0), (0, 255), (255, 0), (255, 255)]:
            assert np.all(out[corner] == 0), f"corner {corner} should be black"

    # ------------------------------------------------------------------
    # Centre pixel correctness: camera pointing at zenith (lat=90°)
    # ------------------------------------------------------------------

    def test_center_pixel_is_nonzero_when_looking_at_zenith(self):
        """When the camera points at the zenith, the fisheye centre pixel should
        map back to the centre of the source image and thus be non-zero."""
        src = self._solid_image(value=200)
        out = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=90.0, lon_deg=0.0, hfov_deg=60.0
        )
        centre = out[128, 128]
        assert np.any(centre > 0), "Centre pixel should receive source colour"

    # ------------------------------------------------------------------
    # Equatorial placement (lat=0)
    # ------------------------------------------------------------------

    def test_camera_at_equator_produces_output(self):
        """A camera placed at lat=0, lon=0 should paint the bottom edge of the
        fisheye with non-zero pixels (lon=0 now points downward)."""
        src = self._solid_image(value=180)
        out = reproject_rectilinear_to_fisheye(
            src, 512, lat_deg=0.0, lon_deg=0.0, hfov_deg=60.0
        )
        # The bottom slice of the output (toward lat=0, lon=0) should
        # contain some non-zero pixels.
        bottom_strip = out[400:, 256]
        assert np.any(bottom_strip > 0), "Bottom strip should receive source colour"

    # ------------------------------------------------------------------
    # Longitude rotation: the mapping should differ between lon=0 and lon=90°
    # ------------------------------------------------------------------

    def test_different_longitudes_give_different_results(self):
        src = self._solid_image(value=150)
        out0 = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=0.0, lon_deg=0.0, hfov_deg=90.0
        )
        out90 = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=0.0, lon_deg=90.0, hfov_deg=90.0
        )
        # The two outputs should have their non-black regions in different locations.
        assert not np.array_equal(out0, out90)

    # ------------------------------------------------------------------
    # Wider FOV covers more pixels than a narrower FOV
    # ------------------------------------------------------------------

    def test_wider_hfov_covers_more_pixels(self):
        src = self._solid_image(value=200)
        out_narrow = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=90.0, lon_deg=0.0, hfov_deg=30.0
        )
        out_wide = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=90.0, lon_deg=0.0, hfov_deg=120.0
        )
        nonzero_narrow = np.count_nonzero(np.any(out_narrow > 0, axis=-1))
        nonzero_wide = np.count_nonzero(np.any(out_wide > 0, axis=-1))
        assert nonzero_wide > nonzero_narrow

    # ------------------------------------------------------------------
    # Zenith singularity: lat=90° should not raise
    # ------------------------------------------------------------------

    def test_no_error_at_zenith_singularity(self):
        src = self._solid_image()
        reproject_rectilinear_to_fisheye(
            src, 128, lat_deg=90.0, lon_deg=0.0, hfov_deg=60.0
        )  # should not raise

    # ------------------------------------------------------------------
    # upright mode
    # ------------------------------------------------------------------

    def test_upright_output_shape(self):
        """upright=True must not change the output shape."""
        src = self._solid_image()
        out = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=0.0, lon_deg=0.0, hfov_deg=90.0, upright=True
        )
        assert out.shape == (256, 256, 4)

    def test_upright_differs_from_tangential(self):
        """upright=True and upright=False should produce different images for the
        same lat/lon, because lat_deg has a different geometric meaning."""
        src = self._solid_image(value=200)
        out_tangential = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=20.0, lon_deg=0.0, hfov_deg=90.0, upright=False
        )
        out_upright = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=20.0, lon_deg=0.0, hfov_deg=90.0, upright=True
        )
        assert not np.array_equal(out_tangential, out_upright)

    def test_upright_never_crosses_zenith(self):
        """The zenith (fisheye centre) must always be black in upright mode.

        A horizontal optical axis has d_cam_z = 0 for the zenith direction, so
        the zenith is always outside the valid half-space regardless of lat_deg
        or hfov_deg.  Use parameters that would have caused the old (buggy)
        implementation to tilt the axis past the zenith.
        """
        src = self._solid_image(value=200)
        # With the old code, lat_deg=45° + hfov=120° on a square image gave
        # vhalf≈60°, pushing the optical axis to 105° — past the zenith.
        output_size = 256
        center = output_size // 2
        out = reproject_rectilinear_to_fisheye(
            src, output_size, lat_deg=45.0, lon_deg=0.0, hfov_deg=120.0, upright=True
        )
        assert np.all(out[center, center] == 0), "Zenith must be black for a horizontal upright camera"

    def test_upright_zenith_black_for_zero_lat(self):
        """Even with lat_deg=0, the zenith must be black because the optical axis
        is horizontal and cannot 'see' a direction 90° above it."""
        src = self._solid_image(value=200)
        output_size = 256
        center = output_size // 2
        out = reproject_rectilinear_to_fisheye(
            src, output_size, lat_deg=0.0, lon_deg=0.0, hfov_deg=90.0, upright=True
        )
        assert np.all(out[center, center] == 0), "Zenith must be black for horizontal camera"

    def test_upright_false_is_default(self):
        """upright defaults to False; explicit False should match the default."""
        src = self._solid_image(value=150)
        out_default = reproject_rectilinear_to_fisheye(
            src, 128, lat_deg=30.0, lon_deg=0.0, hfov_deg=60.0
        )
        out_explicit = reproject_rectilinear_to_fisheye(
            src, 128, lat_deg=30.0, lon_deg=0.0, hfov_deg=60.0, upright=False
        )
        np.testing.assert_array_equal(out_default, out_explicit)

    # ------------------------------------------------------------------
    # Alpha-channel / transparency
    # ------------------------------------------------------------------

    def test_3channel_input_yields_4channel_output(self):
        """A BGR source image must produce a BGRA output (4-channel)."""
        src = self._solid_image(channels=3)
        out = reproject_rectilinear_to_fisheye(
            src, 128, lat_deg=90.0, lon_deg=0.0, hfov_deg=60.0
        )
        assert out.ndim == 3 and out.shape[2] == 4

    def test_transparent_background(self):
        """Pixels outside the valid projection region must have alpha = 0."""
        src = self._solid_image(channels=3, value=255)
        out = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=90.0, lon_deg=0.0, hfov_deg=5.0
        )
        # Corners are always outside the fisheye circle for a tiny HFOV.
        for corner in [(0, 0), (0, 255), (255, 0), (255, 255)]:
            assert out[corner][3] == 0, f"corner {corner} should be fully transparent"

    def test_valid_pixels_are_opaque(self):
        """The centre pixel (projected from source) must have alpha = 255."""
        src = self._solid_image(channels=3, value=200)
        out = reproject_rectilinear_to_fisheye(
            src, 256, lat_deg=90.0, lon_deg=0.0, hfov_deg=60.0
        )
        assert out[128, 128, 3] == 255, "Centre pixel should be fully opaque"

    def test_4channel_input_unchanged_channels(self):
        """A 4-channel input must pass through without gaining an extra channel."""
        src = np.full((32, 64, 4), 200, dtype=np.uint8)
        out = reproject_rectilinear_to_fisheye(
            src, 64, lat_deg=90.0, lon_deg=0.0, hfov_deg=60.0
        )
        assert out.shape == (64, 64, 4)

    def test_grayscale_input_unchanged_shape(self):
        """A grayscale (2-D) input must not have an alpha channel added."""
        src = np.full((480, 640), 128, dtype=np.uint8)
        out = reproject_rectilinear_to_fisheye(
            src, 128, lat_deg=90.0, lon_deg=0.0, hfov_deg=60.0
        )
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# add_border
# ---------------------------------------------------------------------------


class TestAddBorder:
    def test_no_border_returns_same_array(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = add_border(img, 0)
        assert out is img

    def test_negative_border_returns_same_array(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = add_border(img, -5)
        assert out is img

    def test_border_increases_size(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        out = add_border(img, 5)
        assert out.shape == (20, 30, 3)

    def test_white_border_color(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = add_border(img, 1, "ffffff")
        # Top-left corner pixel should be white (BGR: 255, 255, 255).
        assert np.all(out[0, 0] == 255)

    def test_custom_border_color(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = add_border(img, 1, "ff0000")  # pure red in RGB → BGR = (0, 0, 255)
        assert out[0, 0, 0] == 0    # B
        assert out[0, 0, 1] == 0    # G
        assert out[0, 0, 2] == 255  # R

    def test_border_color_with_hash_prefix(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = add_border(img, 1, "#ffffff")
        assert np.all(out[0, 0] == 255)

    def test_border_preserves_inner_content(self):
        img = np.full((4, 4, 3), 42, dtype=np.uint8)
        out = add_border(img, 2, "000000")
        # Inner region (rows 2:6, cols 2:6) should still be 42.
        assert np.all(out[2:6, 2:6] == 42)

    def test_border_bgra_image_gets_opaque_alpha(self):
        img = np.zeros((4, 4, 4), dtype=np.uint8)
        out = add_border(img, 1, "ffffff")
        # Alpha channel of the border should be 255 (opaque).
        assert out[0, 0, 3] == 255

    def test_border_grayscale_image(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        out = add_border(img, 1, "808080")
        assert out.shape == (6, 6)
        # Expected luminance: convert #808080 (BGR: 128, 128, 128) via OpenCV.
        expected = int(cv2.cvtColor(np.array([[[128, 128, 128]]], dtype=np.uint8), cv2.COLOR_BGR2GRAY)[0, 0])
        assert out[0, 0] == expected


# ---------------------------------------------------------------------------
# parse_keyframe_spec
# ---------------------------------------------------------------------------


class TestParseKeyframeSpec:
    def test_valid_linear(self):
        spec = parse_keyframe_spec("K:0,10,45.0,90.0,L")
        assert spec["f"] == 0
        assert spec["l"] == 10
        assert spec["ival"] == pytest.approx(45.0)
        assert spec["eval"] == pytest.approx(90.0)
        assert spec["interp"] == "L"

    def test_valid_sinusoidal(self):
        spec = parse_keyframe_spec("K:5,-1,10.0,80.0,S")
        assert spec["f"] == 5
        assert spec["l"] == -1
        assert spec["interp"] == "S"

    def test_missing_k_prefix_raises(self):
        with pytest.raises(ValueError, match="must start with"):
            parse_keyframe_spec("5,10,45,90,L")

    def test_wrong_field_count_raises(self):
        with pytest.raises(ValueError, match="5 comma-separated fields"):
            parse_keyframe_spec("K:0,10,45,90")

    def test_invalid_interp_type_raises(self):
        with pytest.raises(ValueError, match="Interpolation type"):
            parse_keyframe_spec("K:0,10,45.0,90.0,X")


# ---------------------------------------------------------------------------
# resolve_keyframable
# ---------------------------------------------------------------------------


class TestResolveKeyframable:
    def test_float_passthrough(self):
        assert resolve_keyframable(45.0, 5, 10) == pytest.approx(45.0)

    def test_int_passthrough(self):
        assert resolve_keyframable(90, 0, 1) == pytest.approx(90.0)

    def test_before_keyframe_range_returns_ival(self):
        # Frames 0-4 should all return ival=10.0
        for idx in range(5):
            assert resolve_keyframable("K:5,15,10.0,90.0,L", idx, 20) == pytest.approx(10.0)

    def test_after_keyframe_range_returns_eval(self):
        # Frames 16-19 should all return eval=90.0
        for idx in range(16, 20):
            assert resolve_keyframable("K:5,15,10.0,90.0,L", idx, 20) == pytest.approx(90.0)

    def test_linear_midpoint(self):
        # At frame 10 (midpoint of [5, 15]), t=0.5 → (10+90)/2 = 50
        val = resolve_keyframable("K:5,15,10.0,90.0,L", 10, 20)
        assert val == pytest.approx(50.0)

    def test_linear_at_start(self):
        val = resolve_keyframable("K:5,15,10.0,90.0,L", 5, 20)
        assert val == pytest.approx(10.0)

    def test_linear_at_end(self):
        val = resolve_keyframable("K:5,15,10.0,90.0,L", 15, 20)
        assert val == pytest.approx(90.0)

    def test_sinusoidal_midpoint(self):
        import math
        # At midpoint t=0.5: s = 0.5*(1 - cos(π*0.5)) = 0.5*(1-0) = 0.5 → 50.0
        val = resolve_keyframable("K:5,15,10.0,90.0,S", 10, 20)
        assert val == pytest.approx(50.0)

    def test_negative_l_resolves_to_last_frame(self):
        # l=-1 with total_frames=20 → l=19; frame 19 should return eval
        val = resolve_keyframable("K:5,-1,10.0,90.0,L", 19, 20)
        assert val == pytest.approx(90.0)

    def test_negative_l_mid_interpolation(self):
        # l=-1 with total_frames=20 → l=19; frame 12 (midpoint of [5,19]) → t=0.5
        val = resolve_keyframable("K:5,-1,10.0,90.0,L", 12, 20)
        assert val == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# CLI: border and keyframable params
# ---------------------------------------------------------------------------


class TestBuildParserNewArgs:
    def test_border_size_default(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg"])
        assert args.border_size == 0

    def test_border_size_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg", "--border-size", "20"])
        assert args.border_size == 20

    def test_border_color_default(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg"])
        assert args.border_color == "ffffff"

    def test_border_color_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg", "--border-color", "000000"])
        assert args.border_color == "000000"

    def test_lat_accepts_float(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg", "--lat", "45.0"])
        assert args.lat == pytest.approx(45.0)

    def test_lat_accepts_keyframe_spec(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg", "--lat", "K:0,10,45.0,90.0,L"])
        assert args.lat == "K:0,10,45.0,90.0,L"

    def test_lon_accepts_keyframe_spec(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg", "--lon", "K:0,-1,0.0,180.0,S"])
        assert args.lon == "K:0,-1,0.0,180.0,S"

    def test_hfov_accepts_keyframe_spec(self):
        parser = build_parser()
        args = parser.parse_args(["in.jpg", "--hfov", "K:5,15,60.0,120.0,L"])
        assert args.hfov == "K:5,15,60.0,120.0,L"

