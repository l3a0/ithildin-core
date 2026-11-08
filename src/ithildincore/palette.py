"""The colours the account's charts and whiteboard sketches draw in.

A colour that fails a reader fails silently. The figure still draws, and only
the reader who cannot tell two lines apart notices. So the colours are named
here once, and ``tests/test_palette.py`` measures them rather than trusting
the hex codes.

Two sets of three colours share one ink and one grey.

1. The colour-blind safe set, for new work. ``ACCENT``, a berry, marks the
   thing a picture is about. ``GOOD``, a blue, marks a profit or the side that
   wins. ``LOST``, a vermilion, marks a loss or a warning.
2. The brass set, ``BRASS``, ``GREEN`` and ``RED`` in the same three roles,
   which l3a0/quantitative-trading's posts drew in before the colour-blind set
   existed. Those posts move to the new set one at a time, and a GIF already
   published stays in the colours it was drawn in, so the brass set stays
   here for as long as any of them draws in it.

The reason for the first set is a measurement. Under deuteranopia the brass of
a post's subject and the red of a loss sit 3.0 apart, on a scale where 8 is
the least a reader can rely on. Each new colour clears 4.5 to 1 against the
sketches' white board and a chart's figure ground, :data:`SURFACE`, so a label
drawn in it reads as text. Each pair stays at least 8 apart under protanopia,
deuteranopia and tritanopia, and at least 15 apart in normal vision. The
measurements leave two gaps, and the tests pin both so a post that moves meets
them as numbers rather than by surprise.

1. Inside a chart's axes, whose ground is the darker :data:`GROUND`, ``GOOD``
   reaches only 4.37 to 1 and ``LOST`` 4.30. A label in either colour that
   sits on the axes falls short of the bar for text, so it goes on a white box
   or in ``INK``.
2. Under protanopia ``LOST`` sits 7.5 from the grey, just short of 8. The
   brass set's red sat 3.8 from it, so the new set narrows the gap without
   closing it. Where a loss and a note in grey sit side by side, a label or a
   sign tells them apart as well as the colour.

A sketch needs two more colours per set, the pale tints a band or a reject
zone is filled with. Each is its set's green or red laid on the white board at
a fifth of its strength, the lightest blend
:data:`ithildincore.whiteboard.BLENDS` holds, so a tint is a colour the GIF's
palette already carries and adds none. The blend is written out here rather
than imported, so this module needs only matplotlib's colour conversions and
not the whiteboard's renderer. ``tests/test_whiteboard.py`` holds that the two
agree.

**Where this came from.** l3a0/quantitative-trading at ``715d1262``. The new
set and its tints are ``src/chan/palette.py`` there. The ink, the grey, the
grounds, the rule colour and the brass set are the essay palette at the top of
``src/chan/regime_figure.py``, taken from the GLD/GDX essay's page. The brass
set's two tints are ``TINT`` and ``REJECT_TINT`` in
``src/chan/usdcad_tests_whiteboard.py``. Three things changed on the way over.

1. The brass set's colours were named ``ACCENT``, ``GOOD`` and ``LOST`` in
   ``chan.regime_figure``, the same names as the new set's. Here they are
   ``BRASS``, ``GREEN`` and ``RED``, so one module can hold both.
2. The brass set's tints were computed with the whiteboard's own blend. Here
   they use the same arithmetic as the new set's, and the tests hold that both
   give the same hex code.
3. The prose was rewritten for a reader outside that repository.
"""

from __future__ import annotations

from matplotlib.colors import to_hex, to_rgb

# The essay palette's neutrals, shared by both sets.
SURFACE = "#FEFDFA"  # a chart's figure ground
GROUND = "#F7F5EF"  # a chart's axes ground
INK = "#23201A"
MUTED = "#6B6353"
RULE = "#CFC8B4"

# The colour-blind safe set, for new work.
ACCENT = "#981E87"  # berry: the subject
GOOD = "#276EE1"  # blue: a profit, or the side that wins
LOST = "#D24011"  # vermilion: a loss, or a warning

# The brass set, in the same three roles.
BRASS = "#8B651A"  # the subject. It reads as text on the cream ground.
GREEN = "#4A7C59"  # a profit, or the side that wins
RED = "#A24C3C"  # a loss, or a warning

#: The inks a sketch drawn in the colour-blind safe set is mapped onto, in the
#: order :func:`ithildincore.whiteboard.palette` reads them.
INKS = (INK, MUTED, ACCENT, GOOD, LOST)
#: The same five for the brass set, which the whiteboard's renderer defaults to.
BRASS_INKS = (INK, MUTED, BRASS, GREEN, RED)


def _lightest(ink: str) -> str:
    """``ink`` laid on the white board at a fifth of its strength, the lightest
    blend :data:`ithildincore.whiteboard.BLENDS` holds."""
    return to_hex([0.2 * channel + 0.8 for channel in to_rgb(ink)])


#: A band's tint in the colour-blind safe set: ``GOOD`` at the lightest blend.
TINT = _lightest(GOOD)
#: A reject zone's tint in the colour-blind safe set: ``LOST`` at that blend.
REJECT_TINT = _lightest(LOST)
#: A band's tint in the brass set: ``GREEN`` at the lightest blend.
GREEN_TINT = _lightest(GREEN)
#: A reject zone's tint in the brass set: ``RED`` at the lightest blend.
RED_TINT = _lightest(RED)
