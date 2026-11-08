"""What the whiteboard machinery times, draws, writes and checks.

Most of these tests come from l3a0/quantitative-trading at ``715d1262``. The
timing, drawing and clearance tests were in ``tests/test_usdcad_tests_whiteboard.py``
there, run on the USD.CAD post's fourteen scenes. Here they run on two of
those scenes, "The question" and "ADF", which are schematic and carry no data,
so they are copied below as :func:`_question` and :func:`_adf` with the
helpers renamed. The checks' tests were in ``tests/test_sketch_draft.py``,
where they ran through the sketch-notes skill's draft script. Here they call
:func:`ithildincore.whiteboard.draft` and its parts directly. The pacing and
GIF writer tests are new, because the pacing and the writer were copies in
each post's module there.

No test here compares a GIF's bytes with a committed file. A GIF's bytes
depend on the font the machine has, so that comparison belongs to the
repository that commits the GIFs, on the machine that drew them.
quantitative-trading holds it for every GIF it commits.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgb
from PIL import Image

from ithildincore import palette as colours
from ithildincore import whiteboard
from ithildincore.palette import ACCENT, GOOD, INK, INKS, LOST, MUTED
from ithildincore.whiteboard import (
    APART_LABELS_PX,
    BLENDS,
    CLEAR_PX,
    FALLBACK_FONT,
    FRAME_MS,
    HEIGHT,
    HOLD_MS,
    INSIDE_PX,
    LABEL_MS,
    PACED_HOLD_MS,
    PACED_LABEL_MS,
    TITLE_MS,
    WIDTH,
    Fill,
    Frame,
    Label,
    Scene,
    Stroke,
    arrow,
    clearances,
    curve,
    dot,
    draft,
    frame_problems,
    inks_for,
    layout_problems,
    line,
    paced_timeline,
    render,
    stroke,
    timeline,
    write_gif,
)


def _question() -> Scene:
    """ "The question" from quantitative-trading's USD.CAD post, a wiggly price
    around a dashed mean and an arrow pulling it back."""
    x = np.linspace(0.4, 9.6, 140)
    price = 2.0 + 0.55 * np.sin(1.7 * x) + 0.3 * np.sin(4.3 * x + 1) + 0.12 * np.sin(9.1 * x)
    return Scene(
        "The question",
        "question",
        (
            stroke("mean", line([(0.4, 2.0), (9.6, 2.0)]), color=MUTED, dashed=True, frames=6),
            stroke("price", curve(x, price), frames=14),
            Label("Random walk?", 0.6, 3.55, names="price"),
            stroke("pull", arrow(6.9, 3.35, 6.9, 2.15), color=ACCENT, frames=6),
            Label("Pulled back?", 7.15, 3.55, names="pull", color=ACCENT),
        ),
    )


def _adf() -> Scene:
    """ "ADF" from the same post: a high close followed by a fall."""
    before = np.linspace(0.4, 5.5, 70)
    wiggle = 0.12 * np.sin(5.0 * before)
    rise = 1.4 + wiggle + 2.0 * np.exp(-(((before - 5.5) / 0.55) ** 2))
    after = np.linspace(5.5, 9.6, 60)
    fall = 1.4 + 2.0 * np.exp(-(after - 5.5) / 1.1) + 0.1 * np.sin(5.0 * (after - 5.5))
    peak = float(rise[-1])
    return Scene(
        "ADF",
        "adf",
        (
            stroke("mean", line([(0.4, 1.4), (9.6, 1.4)]), color=MUTED, dashed=True, frames=6),
            stroke("rise", curve(before, rise), frames=10),
            dot("high", 5.5, peak, color=ACCENT),
            Label("high close…", 5.75, peak + 0.15, names="high", color=ACCENT),
            stroke("fall", curve(after, fall), frames=10),
            Label("…then a fall", 7.0, 2.35, names="fall"),
            Label("λ < 0", 8.3, 0.6, names="fall", color=ACCENT, size=16),
        ),
    )


def _scene(*steps) -> Scene:
    return Scene("A test sketch", "test_sketch", steps)


def _line_across(frames: int = 8) -> Stroke:
    return stroke("line", line([(1.0, 2.0), (9.0, 2.0)]), frames=frames)


def _inked(image: Image.Image) -> int:
    """How many pixels are not the white board, which is palette entry 0."""
    return int((np.asarray(image) != 0).sum())


def _durations(path) -> list[int]:
    with Image.open(path) as image:
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(image.info["duration"])
        return durations


@pytest.fixture(scope="module")
def scenes() -> tuple[Scene, Scene]:
    return _question(), _adf()


@pytest.fixture(scope="module")
def question():
    """The question's frames drawn in the colour-blind safe inks."""
    scene = _question()
    steps = timeline(scene)
    return scene, steps, render(scene, steps, INKS)


class TestTheShapes:
    def test_a_line_keeps_every_corner_and_fills_each_segment_by_length(self) -> None:
        x, y = line([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)])
        assert (x[0], y[0]) == (0.0, 0.0)
        assert (x[6], y[6]) == (1.0, 0.0)
        assert (x[-1], y[-1]) == (1.0, 2.0)
        assert len(x) == 1 + 6 + 12

    def test_an_arrow_ends_on_its_tip_with_a_head_either_side(self) -> None:
        x, y = arrow(0.0, 0.0, 2.0, 0.0)
        tip = np.flatnonzero((x == 2.0) & (y == 0.0))
        assert len(tip) == 2
        assert x[-1] < 2.0 and y[-1] < 0.0
        assert x[tip[1] - 1] < 2.0 and y[tip[1] - 1] > 0.0

    def test_a_dot_is_one_point_drawn_on_one_frame(self) -> None:
        mark = dot("d", 1.0, 2.0)
        assert (mark.points, mark.frames, mark.color) == (1, 1, colours.BRASS)

    def test_a_stroke_takes_its_style(self) -> None:
        drawn = stroke("s", line([(0, 0), (1, 1)]), color=LOST, width=3.0, dashed=True, frames=2)
        assert (drawn.gid, drawn.color, drawn.width, drawn.dashed, drawn.frames) == (
            "s",
            LOST,
            3.0,
            True,
            2,
        )


class TestTheTimeline:
    def test_each_scene_takes_the_frames_and_time_it_did_in_its_post(self, scenes) -> None:
        """quantitative-trading pinned these two counts and totals for the
        USD.CAD post's GIFs."""
        frames = [timeline(scene) for scene in scenes]
        assert [len(f) for f in frames] == [29, 31]
        assert [sum(frame.ms for frame in f) for f in frames] == [7350, 7800]

    def test_each_opens_on_its_title_alone(self, scenes) -> None:
        for scene in scenes:
            first = timeline(scene)[0]
            assert (first.done, first.partial, first.ms) == (0, 0, TITLE_MS)

    def test_each_holds_its_finished_drawing_for_four_seconds(self, scenes) -> None:
        assert HOLD_MS == 4000
        for scene in scenes:
            steps = timeline(scene)
            last = steps[-1]
            assert (last.done, last.partial, last.ms) == (len(scene.steps), 0, HOLD_MS)
            assert all(frame.ms < HOLD_MS for frame in steps[:-1])

    def test_strokes_draw_at_ten_frames_a_second_and_labels_pop_in(self, scenes) -> None:
        for scene in scenes:
            for frame in timeline(scene)[1:-1]:
                if frame.partial:
                    assert frame.ms == FRAME_MS
                    continue
                latest = scene.steps[frame.done - 1]
                assert frame.ms == (FRAME_MS if isinstance(latest, Stroke) else LABEL_MS)
        assert (FRAME_MS, LABEL_MS) == (100, 350)

    def test_a_tint_pops_in_like_a_label(self) -> None:
        scene = _scene(_line_across(), Fill("band", 1.0, 1.0, 4.0, 1.5, colours.TINT))
        assert [f.ms for f in timeline(scene)][-2:] == [FRAME_MS, HOLD_MS]
        scene = _scene(Fill("band", 1.0, 1.0, 4.0, 1.5, colours.TINT), _line_across())
        assert timeline(scene)[1] == Frame(1, 0, LABEL_MS)

    def test_each_frame_shows_more_than_the_one_before(self, scenes) -> None:
        """Pillow merges two identical frames in a row, so each frame shows a
        later step, or more of the stroke being drawn, and a stroke drawn in
        part shows at least two points, because a line of one point inks nothing."""
        for scene in scenes:
            steps = timeline(scene)
            for before, after in zip(steps, steps[1:], strict=False):
                assert (after.done, after.partial) > (before.done, before.partial)
                assert after.partial == 0 or after.partial >= 2

    def test_a_stroke_grows_frame_by_frame(self, scenes) -> None:
        """The question's price line shows a longer prefix on each of its 14 frames."""
        price = scenes[0].steps[1]
        shown = [f.partial for f in timeline(scenes[0]) if f.done == 1 and f.partial]
        assert len(shown) == price.frames - 1 == 13
        assert shown == sorted(set(shown))
        assert shown[-1] < price.points


class TestThePacing:
    def test_paced_labels_show_900_ms_and_the_drawing_holds_12_seconds(self, scenes) -> None:
        assert (PACED_LABEL_MS, PACED_HOLD_MS) == (900, 12000)
        for scene in scenes:
            plain, paced = timeline(scene), paced_timeline(scene)
            assert [(f.done, f.partial) for f in paced] == [(f.done, f.partial) for f in plain]
            for before, after in zip(plain[1:-1], paced[1:-1], strict=True):
                labelled = (
                    after.partial == 0
                    and after.done
                    and not isinstance(scene.steps[after.done - 1], Stroke)
                )
                assert after.ms == (PACED_LABEL_MS if labelled else before.ms)
            assert (paced[0].ms, paced[-1].ms) == (plain[0].ms, PACED_HOLD_MS)

    def test_a_tint_is_paced_like_a_label(self) -> None:
        scene = _scene(Fill("band", 1.0, 1.0, 4.0, 1.5, colours.TINT), _line_across())
        assert paced_timeline(scene)[1] == Frame(1, 0, PACED_LABEL_MS)

    def test_the_pace_can_be_set(self, scenes) -> None:
        paced = paced_timeline(scenes[0], label_ms=600, hold_ms=8000)
        assert paced[-1].ms == 8000
        assert {f.ms for f in paced[1:-1]} == {FRAME_MS, 600}


class TestTheDrawing:
    def test_ink_only_grows_within_a_scene(self, question) -> None:
        _, _, images = question
        inked = [_inked(image) for image in images]
        assert inked == sorted(inked)
        assert inked[0] > 0, "the title shows on the first frame"

    def test_a_partial_stroke_inks_less_than_the_whole(self, question) -> None:
        _, steps, images = question
        price = [k for k, f in enumerate(steps) if f.done == 1 and f.partial]
        whole = next(k for k, f in enumerate(steps) if f.done == 2)
        assert _inked(images[price[0]]) < _inked(images[price[-1]]) < _inked(images[whole])

    def test_each_label_adds_ink_on_the_frame_it_appears(self, question) -> None:
        scene, steps, images = question
        labels = [i + 1 for i, s in enumerate(scene.steps) if isinstance(s, Label)]
        for done in labels:
            k = next(k for k, f in enumerate(steps) if f.done == done and f.partial == 0)
            assert _inked(images[k]) > _inked(images[k - 1]), done

    def test_every_frame_uses_the_fixed_palette_of_its_inks(self, question) -> None:
        _, _, images = question
        expected = whiteboard.palette(INKS).getpalette()
        for image in images:
            assert (image.mode, image.size) == ("P", (WIDTH, HEIGHT))
            assert image.getpalette() == expected

    @pytest.mark.parametrize("inks", [colours.BRASS_INKS, INKS], ids=["brass", "new"])
    def test_the_palette_is_white_and_five_inks_at_five_blends(self, inks) -> None:
        flat = whiteboard.palette(inks).getpalette()
        assert flat[:3] == [255, 255, 255]
        assert len({tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)}) == 26

    def test_the_renderer_defaults_to_the_brass_inks(self) -> None:
        """A default is a body change for every caller relying on it, so it
        stays the set quantitative-trading's posts drew their GIFs in."""
        assert whiteboard.palette().getpalette() == (
            whiteboard.palette(colours.BRASS_INKS).getpalette()
        )
        scene = _scene(_line_across())
        (image,) = render(scene, timeline(scene)[-1:])
        assert image.getpalette() == whiteboard.palette(colours.BRASS_INKS).getpalette()

    def test_a_tint_lies_beneath_a_stroke_drawn_before_it(self) -> None:
        """The stroke comes first in the steps, so only the zorder keeps it on top."""
        across = stroke("line", line([(1.0, 2.0), (9.0, 2.0)]), color=INK, width=6.0)
        scene = _scene(across, Fill("band", 1.0, 1.5, 9.0, 2.5, colours.TINT))
        (image,) = render(scene, timeline(scene)[-1:], INKS)
        rgb = np.asarray(image.convert("RGB"))
        middle = rgb[HEIGHT - round(2.0 / 4.7 * HEIGHT), WIDTH // 2]
        assert tuple(middle) == tuple(round(255 * c) for c in to_rgb(INK))

    def test_no_glyph_falls_back_to_an_empty_box(self, scenes, caplog) -> None:
        """λ and the ellipsis must each be drawn by some installed font."""
        with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
            warnings.simplefilter("always")
            for scene in scenes:
                render(scene, timeline(scene)[-1:], INKS)
        messages = [str(w.message) for w in caught] + [r.getMessage() for r in caplog.records]
        assert not [m for m in messages if "glyph" in m.lower()], messages

    def test_a_fallback_font_keeps_dejavu_behind_the_hand_font(self) -> None:
        assert whiteboard.style("Comic Sans MS")["font.family"] == ["Comic Sans MS", FALLBACK_FONT]
        assert whiteboard.style(FALLBACK_FONT)["font.family"] == [FALLBACK_FONT]

    def test_the_style_follows_hand_font_when_no_font_is_given(self, monkeypatch) -> None:
        monkeypatch.setattr(whiteboard, "hand_font", lambda: FALLBACK_FONT)
        assert whiteboard.style()["font.family"] == [FALLBACK_FONT]


class TestTheGif:
    def test_it_holds_every_frame_at_its_time_and_loops(self, tmp_path) -> None:
        scene = _question()
        frames = paced_timeline(scene)
        path = write_gif(scene, frames, tmp_path / "question.gif", INKS)
        assert path == tmp_path / "question.gif"
        with Image.open(path) as image:
            assert (image.size, image.n_frames) == ((WIDTH, HEIGHT), len(frames))
            assert image.info["loop"] == 0
        assert _durations(path) == [f.ms for f in frames]

    def test_it_draws_on_the_inks_it_is_given(self, tmp_path) -> None:
        scene = _question()
        frames = timeline(scene)
        path = write_gif(scene, frames, tmp_path / "question.gif", INKS)
        with Image.open(path) as image:
            assert image.getpalette() == whiteboard.palette(INKS).getpalette()

    def test_it_writes_what_render_draws(self, tmp_path) -> None:
        scene = _adf()
        frames = timeline(scene)
        path = write_gif(scene, frames, tmp_path / "adf.gif")
        drawn = render(scene, frames)
        with Image.open(path) as image:
            for k, frame in enumerate(drawn):
                image.seek(k)
                written = np.asarray(image.convert("RGB"))
                assert np.array_equal(written, np.asarray(frame.convert("RGB"))), k

    def test_two_draws_give_the_same_bytes(self, tmp_path) -> None:
        scene = _adf()
        one = write_gif(scene, timeline(scene), tmp_path / "one.gif", INKS)
        two = write_gif(scene, timeline(scene), tmp_path / "two.gif", INKS)
        assert one.read_bytes() == two.read_bytes()


class TestTheClearances:
    @pytest.mark.parametrize("font", [None, FALLBACK_FONT], ids=["installed", "fallback"])
    def test_every_label_and_the_title_is_measured(self, scenes, font) -> None:
        for scene in scenes:
            found = clearances(scene, font)
            assert set(found) == {*scene.labels, scene.title}
            for text, pixels in found.items():
                assert pixels >= CLEAR_PX, (scene.title, text, round(pixels, 1))

    def test_the_measure_sees_a_label_on_a_line(self, scenes) -> None:
        adf = scenes[1]
        steps = tuple(
            dataclasses.replace(s, x=0.6, y=1.4) if isinstance(s, Label) and s.size == 16 else s
            for s in adf.steps
        )
        assert clearances(dataclasses.replace(adf, steps=steps))["λ < 0"] < 0

    def test_the_measure_sees_a_label_on_a_tint(self) -> None:
        """A label over a tint reads zero, and one beside it reads its gap."""
        band = Fill("band", 3.0, 1.0, 5.0, 2.0, colours.TINT)
        on = Label("±1 SD", 4.0, 1.5, names="band", ha="center")
        assert clearances(_scene(band, on))["±1 SD"] == 0
        beside = dataclasses.replace(on, x=band.x1 + 1, ha="left")
        # One board unit is a tenth of the frame's width, 68 pixels.
        gap = clearances(_scene(band, beside))["±1 SD"]
        assert gap == pytest.approx(WIDTH / 10, abs=3)

    def test_the_font_changes_the_layout(self, scenes) -> None:
        if whiteboard.hand_font() == FALLBACK_FONT:
            pytest.skip("no hand font is installed, so both layouts are DejaVu Sans")
        installed, fallback = clearances(scenes[0]), clearances(scenes[0], FALLBACK_FONT)
        assert installed != fallback


def _found(scene: Scene, tmp_path) -> list[str]:
    return draft(scene, tmp_path)


class TestAClearDraft:
    def test_a_schematic_scene_breaks_no_rule(self, tmp_path) -> None:
        assert _found(_adf(), tmp_path) == []

    def test_it_writes_the_gif_and_its_last_frame(self, tmp_path) -> None:
        _found(_adf(), tmp_path)
        assert {path.name for path in tmp_path.iterdir()} == {"adf.gif", "adf_last.png"}

    def test_a_paced_draft_is_timed_by_the_paced_timeline(self, tmp_path) -> None:
        draft(_adf(), tmp_path, paced=True)
        assert _durations(tmp_path / "adf.gif") == [f.ms for f in paced_timeline(_adf())]


class TestEachBrokenRuleIsFlagged:
    def test_a_label_on_its_line(self, tmp_path) -> None:
        scene = _scene(_line_across(), Label("on the line", 3.0, 2.05, names="line"))
        assert any("px from a stroke" in found for found in _found(scene, tmp_path))

    @pytest.mark.parametrize(
        ("x", "y", "ha"),
        [(9.8, 4.6, "left"), (9.9, 3.0, "left"), (5.0, 4.68, "left"), (0.02, 3.0, "right")],
        ids=["corner", "right edge", "top edge", "left edge"],
    )
    def test_a_label_off_the_board(self, tmp_path, x, y, ha) -> None:
        scene = _scene(_line_across(), Label("off the board", x, y, names="line", ha=ha))
        assert any("px inside the board" in found for found in _found(scene, tmp_path))

    def test_two_labels_touching(self, tmp_path) -> None:
        scene = _scene(
            _line_across(),
            Label("one", 5.0, 3.5, names="line"),
            Label("two", 5.3, 3.5, names="line"),
        )
        assert any("px apart" in found for found in _found(scene, tmp_path))

    def test_a_label_naming_a_stroke_drawn_after_it(self, tmp_path) -> None:
        scene = _scene(Label("too early", 3.0, 3.5, names="line"), _line_across())
        assert any("not drawn before it" in found for found in _found(scene, tmp_path))

    @pytest.mark.parametrize(
        ("points", "frames"), [(5, 8), (6, 6)], ids=["more frames", "as many frames"]
    )
    def test_a_stroke_with_too_few_points_for_its_frames(self, tmp_path, points, frames) -> None:
        short = Stroke("short", np.linspace(1.0, 9.0, points), np.full(points, 2.0), frames=frames)
        found = _found(_scene(short), tmp_path)
        assert any(f"{frames} frames for {points} points" in f for f in found)
        assert any("the GIF holds" in f for f in found)

    def test_a_one_point_dot_drawn_over_many_frames(self, tmp_path) -> None:
        lone = Stroke("dot", np.array([5.0]), np.array([2.0]))
        found = _found(_scene(lone), tmp_path)
        assert any("8 frames for 1 points" in f for f in found)
        assert any("the GIF holds" in f for f in found)

    def test_a_gif_that_lost_frames_is_flagged_against_its_timeline(self, tmp_path) -> None:
        scene = _scene(_line_across())
        path = write_gif(scene, timeline(scene), tmp_path / "t.gif")
        assert frame_problems(scene, len(timeline(scene)), path) == []
        assert frame_problems(scene, len(timeline(scene)) + 1, path) == [
            f"the GIF holds {len(timeline(scene))} frames where the timeline has "
            f"{len(timeline(scene)) + 1}"
        ]

    def test_a_label_naming_a_tint_drawn_before_it_is_ok(self, tmp_path) -> None:
        scene = _scene(
            Fill("band", 1.0, 1.0, 4.0, 1.5, MUTED),
            Label("a band", 5.0, 3.5, names="band"),
        )
        assert not any("not drawn before it" in found for found in _found(scene, tmp_path))

    def test_a_dot_drawn_in_one_frame_is_ok(self, tmp_path) -> None:
        assert _found(_scene(dot("dot", 5.0, 2.0)), tmp_path) == []

    def test_the_thresholds_are_the_ones_the_rules_name(self) -> None:
        assert (CLEAR_PX, INSIDE_PX, APART_LABELS_PX) == (10, 5, 4)


class TestBothFontsAreChecked:
    def test_the_layout_runs_in_the_hand_font_and_the_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(whiteboard, "hand_font", lambda: "Comic Sans MS")
        laid = []
        monkeypatch.setattr(whiteboard, "clearances", lambda scene, font: laid.append(font) or {})
        monkeypatch.setattr(whiteboard, "boxes", lambda scene, font: [])
        layout_problems(_scene(_line_across()))
        assert laid == ["Comic Sans MS", FALLBACK_FONT]

    def test_the_fallback_is_checked_once_when_it_is_the_hand_font(self, monkeypatch) -> None:
        monkeypatch.setattr(whiteboard, "hand_font", lambda: FALLBACK_FONT)
        laid = []
        monkeypatch.setattr(whiteboard, "clearances", lambda scene, font: laid.append(font) or {})
        monkeypatch.setattr(whiteboard, "boxes", lambda scene, font: [])
        layout_problems(_scene(_line_across()))
        assert laid == [FALLBACK_FONT]

    def test_a_label_only_the_fallback_pushes_off_the_board_is_flagged(self, monkeypatch) -> None:
        """A label set to end just inside the right edge in the hand font runs
        past it in the wider DejaVu Sans."""
        if whiteboard.hand_font() == FALLBACK_FONT:
            pytest.skip("no hand font is installed to compare with DejaVu Sans")
        words = "a long label that ends near the edge"
        probe = _scene(_line_across(), Label(words, 0.0, 3.5, names="line"))
        hand = next(b for t, b in whiteboard.boxes(probe) if t == words)
        wide = next(b for t, b in whiteboard.boxes(probe, FALLBACK_FONT) if t == words)
        assert wide.width > hand.width, "DejaVu Sans lays this label out wider"
        x = (WIDTH - INSIDE_PX - 1 - hand.width) / WIDTH * 10
        found = layout_problems(_scene(_line_across(), Label(words, x, 3.5, names="line")))
        flagged = [f for f in found if "inside the board" in f]
        assert flagged and all(f.startswith(FALLBACK_FONT) for f in flagged)


class TestTheTwoColourSets:
    @staticmethod
    def _wide(color: str) -> Stroke:
        return stroke("band", line([(1.0, 2.0), (9.0, 2.0)]), color=color, width=12.0)

    @staticmethod
    def _rgbs(png) -> set[tuple[int, ...]]:
        with Image.open(png) as image:
            return {rgb for _, rgb in image.convert("RGB").getcolors(1 << 16)}

    def test_a_scene_in_the_new_colours_draws_in_them(self, tmp_path) -> None:
        assert _found(_scene(self._wide(ACCENT)), tmp_path) == []
        berry = tuple(round(255 * c) for c in to_rgb(ACCENT))
        assert berry in self._rgbs(tmp_path / "test_sketch_last.png")

    @pytest.mark.parametrize("color", [colours.BRASS, INK])
    def test_a_scene_in_the_brass_colours_keeps_the_brass_inks(self, color) -> None:
        assert inks_for(_scene(self._wide(color))) == (colours.BRASS_INKS, [])

    @pytest.mark.parametrize("tint", [colours.TINT, colours.REJECT_TINT])
    def test_a_new_tint_alone_moves_the_scene(self, tint) -> None:
        scene = _scene(Fill("band", 1.0, 1.0, 4.0, 1.5, tint))
        assert inks_for(scene) == (INKS, [])

    @pytest.mark.parametrize(
        ("mark", "colour"),
        [
            (dot("dot", 5.0, 3.0), colours.BRASS),
            (Fill("zone", 1.0, 3.0, 4.0, 3.5, colours.GREEN_TINT), colours.GREEN_TINT),
        ],
        ids=["a dot in its default brass", "the brass set's tint"],
    )
    def test_a_mark_left_in_the_brass_set_is_flagged(self, tmp_path, mark, colour) -> None:
        found = _found(_scene(self._wide(GOOD), mark), tmp_path)
        assert len(found) == 1
        assert "brass set" in found[0] and colour.lower() in found[0]

    def test_a_lone_dot_is_flagged_once_the_scene_has_moved(self) -> None:
        scene = _scene(_line_across(), dot("dot", 5.0, 3.0))
        assert inks_for(scene) == (colours.BRASS_INKS, [])
        inks, found = inks_for(scene, moved=True)
        assert inks == INKS
        assert len(found) == 1 and colours.BRASS.lower() in found[0]

    def test_each_set_is_marked_by_its_three_colours_and_two_tints(self) -> None:
        assert whiteboard.BRASS_COLOURS == {
            to_hex(c)
            for c in (
                colours.BRASS,
                colours.GREEN,
                colours.RED,
                colours.GREEN_TINT,
                colours.RED_TINT,
            )
        }
        assert whiteboard.NEW_COLOURS == {
            to_hex(c) for c in (ACCENT, GOOD, LOST, colours.TINT, colours.REJECT_TINT)
        }


class TestTheTints:
    @pytest.mark.parametrize(
        ("tint", "ink"),
        [
            (colours.TINT, GOOD),
            (colours.REJECT_TINT, LOST),
            (colours.GREEN_TINT, colours.GREEN),
            (colours.RED_TINT, colours.RED),
        ],
    )
    def test_each_is_its_ink_at_the_renderers_lightest_blend(self, tint, ink) -> None:
        """The palette writes its blend out rather than import the renderer, so
        this holds that the two agree, to the hex code a scene names."""
        assert tint == to_hex(whiteboard.blend(ink, BLENDS[-1]))

    @staticmethod
    def _held(inks) -> set[tuple[int, ...]]:
        flat = whiteboard.palette(inks).getpalette()
        return {tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)}

    @pytest.mark.parametrize(
        ("tint", "inks", "other"),
        [
            (colours.TINT, INKS, colours.BRASS_INKS),
            (colours.REJECT_TINT, INKS, colours.BRASS_INKS),
            (colours.GREEN_TINT, colours.BRASS_INKS, INKS),
            (colours.RED_TINT, colours.BRASS_INKS, INKS),
        ],
    )
    def test_each_adds_no_colour_to_its_sets_palette(self, tint, inks, other) -> None:
        rgb = tuple(round(255 * c) for c in to_rgb(tint))
        assert rgb in self._held(inks)
        assert rgb not in self._held(other)


def test_the_board_maps_ten_units_across_onto_the_frame() -> None:
    """A unit is 68 pixels across and about 68 up, so a shape keeps its proportions."""
    scene = _scene(_line_across())
    with matplotlib.rc_context(whiteboard.style()):
        canvas, artists = whiteboard.new_board(scene)
        whiteboard.show(scene, artists, Frame(len(scene.steps), 0, 0))
        canvas.draw()
        to_pixels = artists[0].axes.transData
        (x0, y0), (x1, y1) = to_pixels.transform([(0.0, 0.0), (1.0, 1.0)])
    assert x1 - x0 == pytest.approx(WIDTH / 10)
    assert y1 - y0 == pytest.approx(HEIGHT / 4.7)
    assert math.isclose(x1 - x0, y1 - y0, rel_tol=0.01)
