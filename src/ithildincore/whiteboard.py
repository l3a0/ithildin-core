"""Whiteboard sketches: scenes drawn stroke by stroke and written as looping GIFs.

A design doc or a blog post explains a mechanism faster with a picture that
builds in front of the reader than with a paragraph, and the account's posts
and design docs draw that picture the same way. A scene module describes one
sketch as data. This module times it, draws it in a hand-drawn style, writes it
as a GIF, and checks its layout, so every repository's sketches share one look
and one set of rules.

A sketch is a :class:`Scene`: a title, a slug its file is named for, and its
steps in drawing order. A step is one of three things.

1. A :class:`Stroke`, a line revealed from its first point to its last over a
   few frames, or a dot, or a row of dots.
2. A :class:`Fill`, a flat tint beneath every stroke, such as a band or a
   reject zone.
3. A :class:`Label`, text that pops in after the stroke or tint it names.

The shape helpers build a stroke's points. :func:`line` runs through corners,
:func:`arrow` adds a head, and :func:`box`, :func:`span`, :func:`check` and
:func:`hatched` draw a rectangle, a bar with end ticks, a check mark and a
shaded rectangle. :func:`stroke` and :func:`dot` wrap points into a step. For
example::

    scene = Scene(
        "A pull back",
        "pull_back",
        (
            stroke("mean", line([(0.4, 2.0), (9.6, 2.0)]), color=MUTED, dashed=True),
            stroke("pull", arrow(6.9, 3.35, 6.9, 2.15), color=ACCENT, frames=6),
            Label("Pulled back?", 7.15, 3.55, names="pull", color=ACCENT),
        ),
    )
    frames = timeline(scene)
    write_gif(scene, frames, folder / "pull_back.gif", INKS)

:func:`timeline` orders one scene's steps into frames, and
:func:`paced_timeline` slows the labels and the final hold for a sketch that
carries more than a reader takes in at the usual pace. :func:`render` draws
the frames and :func:`write_gif` saves them. A test reads the scene and its
timeline without drawing anything. :func:`clearances` measures how far each
label and the title sit from the nearest stroke or tint on the finished
drawing, and :func:`layout_problems`, :func:`frame_problems` and
:func:`inks_for` are the checks a draft runs before anyone sees it.
:func:`draft` runs all of them on one scene.

**Colours.** The renderer maps every frame onto a fixed palette, the white
board and five inks each blended with white at :data:`BLENDS`, rather than one
fitted to each frame. It defaults to :data:`ithildincore.palette.BRASS_INKS`,
and :func:`dot` defaults to the brass set's subject colour, because the posts
that drew these GIFs first draw in that set and a default is a body change for
every caller that relies on it. New work passes
:data:`ithildincore.palette.INKS` to :func:`render` or :func:`write_gif` and
names each colour from the colour-blind safe set, and :func:`inks_for` flags a
scene that mixes the two sets.

**Determinism.** Matplotlib's path sketch adds the same wobble on every draw,
the palette is fixed, and nothing else here is random. So two draws on one
machine give the same bytes, and a redraw with no change is a no-op diff.
Another machine can differ, because the hand-drawn font is whichever of
:data:`HAND_FONTS` is installed, with matplotlib's own DejaVu Sans as the
fallback, and font rasterising varies across platforms. A repository that
commits GIFs pins their bytes on the machine that drew them and skips the
comparison elsewhere.

**Dependencies.** This module needs matplotlib and Pillow, and
:mod:`ithildincore.palette` needs matplotlib, which the estimators do not.
They are the ``whiteboard`` extra, so a consumer of the estimators alone does
not install them::

    ithildincore[whiteboard] @ git+https://github.com/l3a0/ithildin-core@<commit>

**Where this came from.** l3a0/quantitative-trading at ``715d1262``, where the
machinery sat in ``src/chan/usdcad_tests_whiteboard.py`` beside the USD.CAD
post's scenes, the pacing was copied into each post's module, and the checks
sat in ``.claude/skills/sketch-notes/draft.py``. The arithmetic and the drawing
are copied unchanged, and quantitative-trading's committed GIFs redraw byte for
byte through this module. Six things changed on the way over.

1. The helpers lost their leading underscore, because other repositories call
   them: ``_line`` is :func:`line`, and so on for ``_curve``, ``_arrow``,
   ``_box``, ``_span``, ``_check``, ``_hatched``, ``_stroke``, ``_dot``,
   ``_blend`` and ``_style``. ``_board`` is :func:`new_board` and ``_show`` is
   :func:`show`.
2. ``Scene.gif`` named every file with the USD.CAD post's prefix, so it stayed
   with that post. A scene module names its own files.
3. The inks moved to :mod:`ithildincore.palette`, so this module's ``INKS``,
   ``TINT`` and ``REJECT_TINT`` are that module's ``BRASS_INKS``,
   ``GREEN_TINT`` and ``RED_TINT``.
4. :func:`paced_timeline` is the paced timeline each post's
   ``sketch_timeline`` copied, and :func:`write_gif` is the writer each post
   copied, as l3a0/quantitative-trading#708 asked.
5. The draft script's checks are library functions here, and
   :func:`clearances`, :func:`boxes` and :func:`style` take the font to lay
   out in, where the script swapped this module's :func:`hand_font` instead.
6. The prose was rewritten for a reader outside that repository.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle
from PIL import Image

from ithildincore.palette import (
    ACCENT,
    BRASS,
    BRASS_INKS,
    GOOD,
    GREEN,
    GREEN_TINT,
    INK,
    INKS,
    LOST,
    RED,
    RED_TINT,
    REJECT_TINT,
    TINT,
)

#: The frame in pixels, and the figure size and dpi that give it.
WIDTH, HEIGHT = 680, 320
DPI = 100

#: One drawing frame lasts this long, so strokes are drawn at 10 frames a second.
FRAME_MS = 100
#: A scene's title shows alone for this long before its first stroke.
TITLE_MS = 400
#: A label, or a tint, pops in on one frame that lasts this long.
LABEL_MS = 350
#: The finished drawing shows this long before the GIF loops back to its title.
HOLD_MS = 4000
#: The slower pace of :func:`paced_timeline`: a label or a tint shows for
#: :data:`PACED_LABEL_MS` and the finished drawing holds for
#: :data:`PACED_HOLD_MS`.
PACED_LABEL_MS, PACED_HOLD_MS = 900, 12000

BOARD = "#FFFFFF"
#: Each ink is laid on the white board at these strengths, so a frame's
#: anti-aliased edges map onto colours the palette already holds.
BLENDS = (1.0, 0.8, 0.6, 0.4, 0.2)

#: Hand-drawn fonts in order of preference. None is downloaded, so the first
#: one installed is used, and matplotlib's own DejaVu Sans when none is.
HAND_FONTS = ("Humor Sans", "Comic Neue", "Comic Sans MS")
FALLBACK_FONT = "DejaVu Sans"

#: The board's coordinates. Ten units across and 4.7 up keep a unit about as
#: tall as it is wide at 680 by 320 pixels.
XLIM, YLIM = (0.0, 10.0), (0.0, 4.7)


@dataclass(frozen=True)
class Stroke:
    """A line drawn from its first point to its last over ``frames`` frames.

    A stroke of one point is a dot, drawn in a single frame when ``frames`` is
    1, as :func:`dot` sets it. A stroke with a
    ``dots`` size is a row of dots of that size in points, one on each of its
    points and no line between them, revealed a few more on each frame.
    """

    gid: str
    x: np.ndarray
    y: np.ndarray
    color: str = INK
    width: float = 2.2
    dashed: bool = False
    frames: int = 8
    dots: float = 0.0

    @property
    def points(self) -> int:
        return len(self.x)


@dataclass(frozen=True)
class Fill:
    """A flat rectangle of ``color`` from (x0, y0) to (x1, y1), a tint that
    shades a region rather than a line that draws one.

    It appears whole on one frame, like a label, and it sits beneath every
    stroke, so a line drawn across it stays on top.
    """

    gid: str
    x0: float
    y0: float
    x1: float
    y1: float
    color: str


@dataclass(frozen=True)
class Label:
    """Text that appears at its own place in the scene's steps. ``names`` is the
    stroke or tint before it that it labels, which :func:`layout_problems`
    holds is drawn first."""

    text: str
    x: float
    y: float
    names: str
    color: str = INK
    ha: str = "left"
    size: float = 13
    math: bool = False


@dataclass(frozen=True)
class Scene:
    """One scene: its title, the slug its GIF is named for, and its strokes,
    tints and labels in drawing order."""

    title: str
    slug: str
    steps: tuple[Stroke | Fill | Label, ...]

    @property
    def labels(self) -> list[str]:
        return [step.text for step in self.steps if isinstance(step, Label)]

    @property
    def strokes(self) -> list[Stroke]:
        return [step for step in self.steps if isinstance(step, Stroke)]

    @property
    def fills(self) -> list[Fill]:
        return [step for step in self.steps if isinstance(step, Fill)]


@dataclass(frozen=True)
class Frame:
    """One GIF frame: the steps ``scene.steps[:done]`` are complete, and when
    ``partial`` is above zero the stroke at ``done`` shows that many points."""

    done: int
    partial: int
    ms: int


# --- Shapes -------------------------------------------------------------------


def line(points: Sequence[tuple[float, float]], per_unit: float = 6.0) -> tuple:
    """The polyline through ``points``, with points added along each segment
    so it can be revealed a little at a time. A segment gets points in
    proportion to its length, and each corner is kept exactly."""
    xs, ys = [points[0][0]], [points[0][1]]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) * per_unit))
        t = np.arange(1, steps + 1) / steps
        xs.extend(x0 + (x1 - x0) * t)
        ys.extend(y0 + (y1 - y0) * t)
    return np.array(xs), np.array(ys)


def curve(x: np.ndarray, y: np.ndarray) -> tuple:
    """A curve through the given points, as floats, revealed in their order."""
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def arrow(x0: float, y0: float, x1: float, y1: float) -> tuple:
    """A shaft from (x0, y0) to (x1, y1) with a two-stroke head at the end."""
    angle, head = math.atan2(y1 - y0, x1 - x0), 0.22
    left = (x1 - head * math.cos(angle - 0.5), y1 - head * math.sin(angle - 0.5))
    right = (x1 - head * math.cos(angle + 0.5), y1 - head * math.sin(angle + 0.5))
    return line([(x0, y0), (x1, y1), left, (x1, y1), right])


def box(x0: float, y0: float, x1: float, y1: float) -> tuple:
    """A rectangle drawn from its lower left corner, counterclockwise."""
    return line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])


def span(x0: float, x1: float, y: float, tick: float = 0.13) -> tuple:
    """A bar from ``x0`` to ``x1`` at height ``y``, with an upright tick at each end."""
    return line([(x0, y + tick), (x0, y - tick), (x0, y), (x1, y), (x1, y + tick), (x1, y - tick)])


def check(x: float, y: float) -> tuple:
    """A check mark whose short stroke starts at (x, y)."""
    return line([(x, y), (x + 0.1, y - 0.13), (x + 0.32, y + 0.2)], per_unit=10)


def hatched(x0: float, y0: float, x1: float, y1: float) -> tuple:
    """A rectangle shaded with parallel diagonals, drawn as one stroke.

    After the outline, the pen climbs one diagonal, steps along the top edge,
    comes down the next diagonal parallel to the first, and steps along the
    bottom edge, so every move between diagonals retraces the outline.
    """
    rise, gap = (y1 - y0) / 2, 0.3
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    feet = np.arange(x0 + gap, x1 - rise, gap)
    for k, foot in enumerate(feet):
        diagonal = [(foot, y0), (foot + rise, y1)]
        points += diagonal if k % 2 == 0 else diagonal[::-1]
    return line(points)


def stroke(gid: str, xy: tuple, **style) -> Stroke:
    """A :class:`Stroke` through the points ``xy``, one of the shapes above."""
    return Stroke(gid, xy[0], xy[1], **style)


def dot(gid: str, x: float, y: float, color: str = BRASS) -> Stroke:
    """A dot at (x, y), drawn on one frame. Its colour defaults to the brass
    set's subject colour, as the module docstring explains."""
    return Stroke(gid, np.array([x]), np.array([y]), color=color, frames=1)


# --- Timing -------------------------------------------------------------------


def timeline(scene: Scene) -> list[Frame]:
    """Every frame of one scene's GIF in order, with how long each one shows.

    The GIF opens on the scene's title alone. A stroke takes its ``frames``
    frames, each showing a longer prefix of its points, and a tint or a label
    takes one frame of :data:`LABEL_MS`. The last frame, the finished drawing,
    shows for :data:`HOLD_MS`, and then the GIF loops.
    """
    frames = [Frame(0, 0, TITLE_MS)]
    for done, step in enumerate(scene.steps):
        if not isinstance(step, Stroke):
            frames.append(Frame(done + 1, 0, LABEL_MS))
            continue
        for k in range(1, step.frames + 1):
            shown = math.ceil(step.points * k / step.frames)
            if shown == step.points:
                frames.append(Frame(done + 1, 0, FRAME_MS))
            else:
                frames.append(Frame(done, max(shown, 2), FRAME_MS))
    last = frames[-1]
    frames[-1] = Frame(last.done, last.partial, HOLD_MS)
    return frames


def paced_timeline(
    scene: Scene, label_ms: int = PACED_LABEL_MS, hold_ms: int = PACED_HOLD_MS
) -> list[Frame]:
    """:func:`timeline`'s frames, slowed for a scene a reader finds too fast.

    Each frame on which a label or a tint appears shows for ``label_ms``
    rather than :data:`LABEL_MS`, and the finished drawing holds for
    ``hold_ms`` rather than :data:`HOLD_MS`. Strokes keep their pace.
    """
    paced = [
        Frame(f.done, f.partial, label_ms)
        if f.partial == 0 and f.done and not isinstance(scene.steps[f.done - 1], Stroke)
        else f
        for f in timeline(scene)
    ]
    last = paced[-1]
    paced[-1] = Frame(last.done, last.partial, hold_ms)
    return paced


# --- Drawing ------------------------------------------------------------------


def hand_font() -> str:
    """The first of :data:`HAND_FONTS` installed here, or DejaVu Sans."""
    for family in HAND_FONTS:
        try:
            findfont(FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        return family
    return FALLBACK_FONT


def style(font: str | None = None) -> dict:
    """Matplotlib settings for the hand-drawn look, ``plt.xkcd`` without pyplot.

    ``font`` is the family to lay text out in, and :func:`hand_font` when it
    is not given. The sketch wobbles every path. DejaVu Sans follows the hand
    font in the family list, so a glyph the hand font lacks, such as λ, falls
    back to it rather than to an empty box.
    """
    font = font if font is not None else hand_font()
    return {
        "path.sketch": (1.0, 100.0, 2.0),
        "font.family": [font] if font == FALLBACK_FONT else [font, FALLBACK_FONT],
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "text.parse_math": False,
    }


def blend(ink: str, amount: float) -> np.ndarray:
    """``ink`` laid on the white board at ``amount``, as RGB from 0 to 1."""
    board = np.array(matplotlib.colors.to_rgb(BOARD))
    return amount * np.array(matplotlib.colors.to_rgb(ink)) + (1 - amount) * board


def palette(inks: Sequence[str] = BRASS_INKS) -> Image.Image:
    """The GIF's palette as an image Pillow can quantize against: white, then
    each of ``inks`` at each of :data:`BLENDS` with white."""
    colours = [np.array(matplotlib.colors.to_rgb(BOARD))]
    for ink in inks:
        colours += [blend(ink, amount) for amount in BLENDS]
    flat = [round(255 * channel) for colour in colours for channel in colour]
    holder = Image.new("P", (1, 1))
    holder.putpalette(flat + flat[:3] * (256 - len(colours)))
    return holder


def new_board(scene: Scene) -> tuple[FigureCanvasAgg, list]:
    """A canvas holding the scene's title, and one artist for each step, every
    stroke empty and every tint and label hidden. The title is the last artist.

    A tint is a rectangle at zorder 1, beneath the strokes' 2, so it is drawn
    first whatever order the steps come in. Call it inside
    ``matplotlib.rc_context(style())``, as :func:`render` does.
    """
    fig = Figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    fig.patch.set_facecolor(BOARD)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_axis_off()
    artists = []
    for step in scene.steps:
        if isinstance(step, Stroke):
            (drawn,) = ax.plot([], [], color=step.color, lw=step.width, gid=step.gid)
            if step.dashed:
                drawn.set_linestyle((0, (4, 3)))
            if step.points == 1 or step.dots:
                drawn.set_linestyle("none")
                drawn.set_marker("o")
                drawn.set_markersize(step.dots or 9)
            artists.append(drawn)
        elif isinstance(step, Fill):
            artists.append(
                ax.add_patch(
                    Rectangle(
                        (step.x0, step.y0),
                        step.x1 - step.x0,
                        step.y1 - step.y0,
                        facecolor=step.color,
                        edgecolor="none",
                        zorder=1,
                        gid=step.gid,
                        visible=False,
                    )
                )
            )
        else:
            text = ax.text(
                step.x,
                step.y,
                step.text,
                color=step.color,
                fontsize=step.size,
                ha=step.ha,
                va="center",
                parse_math=step.math,
                visible=False,
            )
            artists.append(text)
    title = ax.text(0.3, 4.3, scene.title, color=INK, fontsize=16, fontweight="bold", va="center")
    return canvas, artists + [title]


def show(scene: Scene, artists: Sequence, frame: Frame) -> None:
    """Set each artist of :func:`new_board` to what ``frame`` shows."""
    for index, (step, artist) in enumerate(zip(scene.steps, artists, strict=False)):
        if not isinstance(step, Stroke):
            artist.set_visible(index < frame.done)
            continue
        shown = step.points if index < frame.done else 0
        if index == frame.done:
            shown = frame.partial
        artist.set_data(step.x[:shown], step.y[:shown])


def render(
    scene: Scene, frames: Sequence[Frame], inks: Sequence[str] = BRASS_INKS
) -> list[Image.Image]:
    """Draw every frame of one scene, each mapped onto :func:`palette` of ``inks``."""
    colours = palette(inks)
    images = []
    with matplotlib.rc_context(style()):
        canvas, artists = new_board(scene)
        for frame in frames:
            show(scene, artists, frame)
            canvas.draw()
            rgb = Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3])
            images.append(rgb.quantize(palette=colours, dither=Image.Dither.NONE))
    return images


def write_gif(
    scene: Scene, frames: Sequence[Frame], path: Path, inks: Sequence[str] = BRASS_INKS
) -> Path:
    """Draw ``frames`` of ``scene`` on ``inks`` and save them to ``path`` as a
    GIF that loops forever, each frame shown for its ``ms``. Returns ``path``.

    Pillow's optimizer is off, so the palette :func:`render` fixed reaches the
    file unchanged.
    """
    first, *rest = render(scene, frames, inks)
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=[frame.ms for frame in frames],
        loop=0,
        optimize=False,
    )
    return path


# --- Checks -------------------------------------------------------------------


def clearances(scene: Scene, font: str | None = None) -> dict[str, float]:
    """How far each label, and the title, sits from the nearest stroke or
    tint on the finished drawing, in pixels, laid out in ``font``. A text
    that appears more than once reports its nearest copy.

    A text's box is the one matplotlib lays it out in. A stroke is its points
    with each segment filled in every pixel or so, and the distance is measured
    to the edge of its ink, half its line width out from the line, or the
    dot's radius for a dot. The wobble the sketch adds is not in the points,
    so a label needs a few pixels to spare to stay clear of it on the board.

    A tint counts the same as a stroke, measured to its rectangle's edge, and
    a label over it reads zero. A label could stay readable over a pale tint,
    but this keeps every label off it, so the tint never has to be checked
    against the text it would sit behind.
    """
    with matplotlib.rc_context(style(font)):
        canvas, artists = new_board(scene)
        show(scene, artists, Frame(len(scene.steps), 0, 0))
        canvas.draw()
        renderer = canvas.get_renderer()
        to_pixels = artists[0].axes.transData
        inks = []
        for step in scene.strokes:
            xy = to_pixels.transform(np.column_stack([step.x, step.y]))
            if step.dots:
                reach = step.dots * DPI / 72 / 2
            elif step.points > 1:
                gaps = np.hypot(*np.diff(xy, axis=0).T)
                xy = np.vstack(
                    [
                        np.linspace(a, b, max(2, math.ceil(gap)), endpoint=False)
                        for a, b, gap in zip(xy[:-1], xy[1:], gaps, strict=True)
                    ]
                    + [xy[-1:]]
                )
                reach = step.width * DPI / 72 / 2
            else:
                reach = 9 * DPI / 72 / 2
            inks.append((xy, reach))
        tints = [
            to_pixels.transform([(fill.x0, fill.y0), (fill.x1, fill.y1)]) for fill in scene.fills
        ]
        texts = [
            (step.text, artist)
            for step, artist in zip(scene.steps, artists, strict=False)
            if isinstance(step, Label)
        ] + [(scene.title, artists[-1])]
        found = {}
        for text, artist in texts:
            extent = artist.get_window_extent(renderer)
            nearest = math.inf
            for xy, reach in inks:
                dx = np.maximum.reduce(
                    [extent.x0 - xy[:, 0], xy[:, 0] - extent.x1, np.zeros(len(xy))]
                )
                dy = np.maximum.reduce(
                    [extent.y0 - xy[:, 1], xy[:, 1] - extent.y1, np.zeros(len(xy))]
                )
                nearest = min(nearest, float(np.hypot(dx, dy).min()) - reach)
            for (x0, y0), (x1, y1) in tints:
                dx = max(extent.x0 - x1, x0 - extent.x1, 0.0)
                dy = max(extent.y0 - y1, y0 - extent.y1, 0.0)
                nearest = min(nearest, math.hypot(dx, dy))
            found[text] = min(nearest, found.get(text, math.inf))
    return found


#: The layout rules a draft holds. A label or the title sits at least
#: :data:`CLEAR_PX` from every stroke and tint and at least :data:`INSIDE_PX`
#: inside the board, and no two labels come within :data:`APART_LABELS_PX`.
CLEAR_PX, INSIDE_PX, APART_LABELS_PX = 10, 5, 4

#: The colours that mark a scene as drawn in one set or the other. The ink and
#: the grey belong to both, so they decide nothing.
BRASS_COLOURS = frozenset(to_hex(c) for c in (BRASS, GREEN, RED, GREEN_TINT, RED_TINT))
NEW_COLOURS = frozenset(to_hex(c) for c in (ACCENT, GOOD, LOST, TINT, REJECT_TINT))


def boxes(scene: Scene, font: str | None = None) -> list:
    """Each visible text on the finished drawing, laid out in ``font``, with
    its box in pixels."""
    with matplotlib.rc_context(style(font)):
        canvas, artists = new_board(scene)
        show(scene, artists, Frame(len(scene.steps), 0, 0))
        canvas.draw()
        renderer = canvas.get_renderer()
        return [
            (artist.get_text(), artist.get_window_extent(renderer))
            for artist in artists
            if hasattr(artist, "get_text") and artist.get_visible()
        ]


def layout_problems(scene: Scene) -> list[str]:
    """Every layout rule the scene breaks, checked in each font the board can use.

    1. A label or the title sits at least :data:`CLEAR_PX` from every stroke
       and tint.
    2. A label or the title sits at least :data:`INSIDE_PX` inside the board.
    3. No two labels come within :data:`APART_LABELS_PX` of each other.
    4. Every label names a stroke or tint that comes before it in the steps.

    The first three run in :func:`hand_font` and again in the fallback, unless
    the fallback is the hand font, because a machine without the hand font
    lays every label out wider or narrower.
    """
    found = []
    seen: set[str] = set()
    for step in scene.steps:
        if isinstance(step, Label) and step.names not in seen:
            found.append(f"label {step.text!r} names {step.names!r}, which is not drawn before it")
        if isinstance(step, (Stroke, Fill)):
            seen.add(step.gid)
    fonts = [hand_font()]
    if fonts[0] != FALLBACK_FONT:
        fonts.append(FALLBACK_FONT)
    for font in fonts:
        for text, gap in clearances(scene, font).items():
            if gap < CLEAR_PX:
                found.append(f"{font}: {text!r} is {gap:.1f} px from a stroke")
        laid = boxes(scene, font)
        for text, extent in laid:
            inside = min(extent.x0, extent.y0, WIDTH - extent.x1, HEIGHT - extent.y1)
            if inside < INSIDE_PX:
                found.append(f"{font}: {text!r} is {inside:.1f} px inside the board")
        for k, (one, a) in enumerate(laid):
            for other, b in laid[k + 1 :]:
                gap = math.hypot(
                    max(a.x0 - b.x1, b.x0 - a.x1, 0.0), max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
                )
                if gap < APART_LABELS_PX:
                    found.append(f"{font}: {one!r} and {other!r} are {gap:.1f} px apart")
    return found


def frame_problems(scene: Scene, timeline_frames: int, gif: Path) -> list[str]:
    """The strokes that will merge frames, and whether the written GIF lost any.

    When two frames show the same picture, Pillow merges them, and the GIF's
    timing no longer matches the timeline. A stroke drawn over more than one
    frame needs more points than frames to avoid that, and a one-point dot
    needs ``frames=1``. ``gif`` is the file written from a timeline of
    ``timeline_frames`` frames.
    """
    found = [
        f"stroke {step.gid}: {step.frames} frames for {step.points} points"
        for step in scene.strokes
        if step.frames > 1 and step.frames >= step.points
    ]
    with Image.open(gif) as image:
        written = image.n_frames
    if written != timeline_frames:
        found.append(f"the GIF holds {written} frames where the timeline has {timeline_frames}")
    return found


def inks_for(scene: Scene, moved: bool = False) -> tuple[tuple[str, ...], list[str]]:
    """The inks a scene is drawn on, and the problem when it mixes the two sets.

    A scene draws in the colour-blind safe set when ``moved`` says its post
    has moved to that set, or when any step uses one of that set's own
    colours, and in the brass set otherwise. In the first case any brass-set
    colour is a problem, even in a scene with no other colour, because the
    renderer maps it onto the nearest new ink, which is the grey. A helper
    whose colour defaults to the brass set, such as :func:`dot`, leaves such a
    mark behind.
    """
    used = {to_hex(step.color) for step in scene.steps}
    brass = sorted(used & BRASS_COLOURS)
    if not (moved or used & NEW_COLOURS):
        return BRASS_INKS, []
    left = [
        f"keeps the brass set's {', '.join(brass)} in a scene drawn in the colour-blind "
        "safe set, which the renderer maps onto the nearest new ink"
    ]
    return INKS, left if brass else []


def draft(scene: Scene, out: Path, paced: bool = False, moved: bool = False) -> list[str]:
    """Write one scene's GIF and last frame into ``out``, and return every rule it breaks.

    The GIF is ``out / "<slug>.gif"``, timed by :func:`paced_timeline` when
    ``paced`` is set and by :func:`timeline` otherwise, and drawn on the inks
    :func:`inks_for` picks. Its last frame is ``out / "<slug>_last.png"``. The
    rules are :func:`inks_for`'s, then :func:`frame_problems`', then
    :func:`layout_problems`'. An empty list means the scene breaks none.
    """
    frames = paced_timeline(scene) if paced else timeline(scene)
    inks, mixed = inks_for(scene, moved)
    images = render(scene, frames, inks)
    gif = out / f"{scene.slug}.gif"
    images[0].save(
        gif,
        save_all=True,
        append_images=images[1:],
        duration=[f.ms for f in frames],
        loop=0,
        optimize=False,
    )
    images[-1].convert("RGB").save(out / f"{scene.slug}_last.png")
    return mixed + frame_problems(scene, len(frames), gif) + layout_problems(scene)
