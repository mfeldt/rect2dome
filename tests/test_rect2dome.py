"""Tests for rect2dome.py

All tests that exercise external tools (ffmpeg, nona) mock the subprocess
calls so that the test suite can run without those tools installed.
"""

import shutil
import subprocess
import textwrap
from pathlib import Path
from unittest import mock

import pytest

import rect2dome
from rect2dome import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_FISHEYE_EQUIDISTANT,
    _default_output_path,
    build_parser,
    create_pto_file,
    detect_media_type,
    encode_video,
    extract_frames,
    get_image_dimensions,
    get_video_framerate,
    main,
    process_image,
    process_video,
    run_nona,
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
# create_pto_file
# ---------------------------------------------------------------------------


class TestCreatePtoFile:
    def _make_pto(self, tmp_path, **kwargs):
        image = tmp_path / "source.tif"
        pto = tmp_path / "project.pto"
        defaults = dict(
            image_path=image,
            pto_path=pto,
            src_width=4096,
            src_height=2048,
            output_width=2048,
            output_height=2048,
        )
        defaults.update(kwargs)
        return create_pto_file(**defaults), pto

    def test_creates_file(self, tmp_path):
        _, pto = self._make_pto(tmp_path)
        assert pto.exists()

    def test_panorama_line_present(self, tmp_path):
        _, pto = self._make_pto(tmp_path, output_projection=3, output_fov=180.0)
        content = pto.read_text()
        assert "p f3" in content
        assert "w2048" in content
        assert "h2048" in content
        assert "v180.0" in content

    def test_image_line_contains_source_path(self, tmp_path):
        image = tmp_path / "source.tif"
        _, pto = self._make_pto(tmp_path, image_path=image)
        assert str(image) in pto.read_text()

    def test_returns_pto_path(self, tmp_path):
        result, pto = self._make_pto(tmp_path)
        assert result == pto

    def test_custom_projections_and_fov(self, tmp_path):
        _, pto = self._make_pto(
            tmp_path,
            input_projection=0,
            output_projection=10,
            hfov=90.0,
            output_fov=180.0,
        )
        content = pto.read_text()
        assert "p f10" in content
        assert "f0" in content
        assert "90.000000" in content


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
# run_nona
# ---------------------------------------------------------------------------


class TestRunNona:
    def test_calls_nona_and_returns_output_path(self, tmp_path):
        pto = tmp_path / "project.pto"
        pto.touch()
        expected_output = tmp_path / "output0000.tif"
        expected_output.touch()  # simulate nona creating the file

        with mock.patch("subprocess.run") as mock_run:
            result = run_nona(pto, tmp_path / "output")

        assert result == expected_output
        cmd = mock_run.call_args[0][0]
        assert "nona" in cmd
        assert str(pto) in cmd

    def test_raises_if_output_missing(self, tmp_path):
        pto = tmp_path / "project.pto"
        pto.touch()

        with mock.patch("subprocess.run"):
            with pytest.raises(FileNotFoundError, match="nona did not produce"):
                run_nona(pto, tmp_path / "output")


# ---------------------------------------------------------------------------
# process_image (integration-level, all subprocesses mocked)
# ---------------------------------------------------------------------------


class TestProcessImage:
    def _run(self, tmp_path, **kwargs):
        src = tmp_path / "pano.tif"
        src.touch()
        dst = tmp_path / "dome.tif"

        # Sequence of subprocess.run calls:
        #   1. get_image_dimensions (ffprobe)
        #   2. nona
        dim_result = mock.MagicMock()
        dim_result.stdout = "4096,2048\n"
        nona_result = mock.MagicMock()

        work_dir = tmp_path / "work"
        work_dir.mkdir()

        def fake_run(cmd, **kw):
            if "nona" in cmd:
                # Simulate nona creating its output
                prefix = cmd[cmd.index("-o") + 1]
                (Path(prefix + "0000.tif")).touch()
                return nona_result
            return dim_result

        with mock.patch("subprocess.run", side_effect=fake_run):
            process_image(
                input_path=src,
                output_path=dst,
                work_dir=work_dir,
                **kwargs,
            )

        assert dst.exists()

    def test_basic(self, tmp_path):
        self._run(tmp_path)

    def test_custom_output_size(self, tmp_path):
        self._run(tmp_path, output_size=4096)


# ---------------------------------------------------------------------------
# process_video (integration-level, all subprocesses mocked)
# ---------------------------------------------------------------------------


class TestProcessVideo:
    def test_basic(self, tmp_path):
        src = tmp_path / "clip.mp4"
        src.touch()
        dst = tmp_path / "clip_dome.mp4"

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        frames_dir = work_dir / "frames"

        fps_result = mock.MagicMock()
        fps_result.stdout = "25\n"
        dim_result = mock.MagicMock()
        dim_result.stdout = "3840,2160\n"
        nona_result = mock.MagicMock()
        ffmpeg_extract_result = mock.MagicMock()
        ffmpeg_encode_result = mock.MagicMock()

        call_count = {"n": 0}

        def fake_run(cmd, **kw):
            call_count["n"] += 1
            if "ffprobe" in cmd and "r_frame_rate" in cmd:
                return fps_result
            if "ffprobe" in cmd and "width" in cmd:
                return dim_result
            if "ffmpeg" in cmd and "-framerate" not in cmd:
                # Frame extraction: create fake frames
                frames_dir.mkdir(parents=True, exist_ok=True)
                (frames_dir / "frame_000001.tif").touch()
                (frames_dir / "frame_000002.tif").touch()
                return ffmpeg_extract_result
            if "nona" in cmd:
                prefix = cmd[cmd.index("-o") + 1]
                (Path(prefix + "0000.tif")).touch()
                return nona_result
            if "ffmpeg" in cmd and "-framerate" in cmd:
                return ffmpeg_encode_result
            return mock.MagicMock()

        with mock.patch("subprocess.run", side_effect=fake_run):
            process_video(
                input_path=src,
                output_path=dst,
                work_dir=work_dir,
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
        assert args.output_size == 2048
        assert args.hfov == 360.0
        assert args.output_fov == 180.0
        assert args.input_projection == PROJECTION_EQUIRECTANGULAR
        assert args.output_projection == PROJECTION_FISHEYE_EQUIDISTANT
        assert not args.keep_frames
        assert not args.verbose

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
        src.touch()
        dst = tmp_path / "pano_dome.jpg"

        dim_result = mock.MagicMock()
        dim_result.stdout = "3600,1800\n"

        def fake_run(cmd, **kw):
            if "nona" in cmd:
                prefix = cmd[cmd.index("-o") + 1]
                (Path(prefix + "0000.tif")).touch()
            return dim_result

        with mock.patch("subprocess.run", side_effect=fake_run):
            rc = main([str(src), str(dst)])

        assert rc == 0
        assert dst.exists()

    def test_subprocess_error_returns_1(self, tmp_path):
        src = tmp_path / "pano.jpg"
        src.touch()
        dst = tmp_path / "pano_dome.jpg"

        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffprobe"),
        ):
            rc = main([str(src), str(dst)])

        assert rc == 1
