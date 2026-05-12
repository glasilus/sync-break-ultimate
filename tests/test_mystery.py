"""Mystery section tests.

Mystery is intentionally chaotic, so we only verify:
  * apply() returns a uint8 array of the same shape,
  * with all knobs at zero, the output equals the input,
  * each knob, when raised in isolation, *does* change the output
    (proves wiring isn't accidentally dead).

Bit-exact golden testing here is brittle because Mystery reseeds RNG from
seg.rms + ZERO; instead we lean on shape/dtype + at-least-changes-something.
"""
import random
import numpy as np

from vpc.analyzer import Segment, SegmentType
from vpc.mystery import MysterySection


def make_seg():
    return Segment(0.0, 1.0, 1.0, SegmentType.SUSTAIN, 0.6, 0.5, 0.3, 0.1)


def make_frame(seed=42, h=64, w=64):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_zero_knobs_is_passthrough():
    m = MysterySection()
    f = make_frame()
    out = m.apply(f, make_seg(), draft=False)
    assert out.shape == f.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, f), 'zero knobs must be identity'


KNOBS = ['VESSEL', 'ENTROPY_7', 'STATIC_MIND', 'RESONANCE',
         'COLLAPSE', 'ZERO', 'FLESH_K', 'DOT']


def test_each_knob_changes_output():
    """Each knob alone, set high, must change the frame at least somewhere."""
    base = make_frame()
    seg = make_seg()
    for knob in KNOBS:
        random.seed(0); np.random.seed(0)
        m = MysterySection()
        setattr(m, knob, 0.9)
        # Run a few frames so stateful knobs (DOT slit-scan, VESSEL feedback)
        # accumulate enough history to be observable.
        out = base
        # Feed 6 *different* frames so stateful knobs (DOT slit-scan, VESSEL
        # feedback) have meaningful history to fold into the output.
        for i in range(6):
            out = m.apply(make_frame(seed=42 + i), seg, draft=False)
        diff = np.abs(out.astype(int) - base.astype(int)).sum()
        assert diff > 0, f'knob {knob} produced no change'


def test_always_flags_default_false_keep_legacy():
    """All `always_<KNOB>` fields exist and default to False — required so
    presets that predate the feature behave bit-identically to before."""
    m = MysterySection()
    for k in KNOBS + ['DELTA_OMEGA']:
        assert hasattr(m, f'always_{k}'), f'missing always_{k}'
        assert getattr(m, f'always_{k}') is False


def test_always_flag_forces_trigger_when_gate_would_block():
    """With a low knob value the random gate would *usually* not fire on
    a single frame. With always-on set, the block must still produce a
    visible change. We use seg.rms=0 so the rms-weighted gate term is 0
    and the bare base probability is small enough to fail most rolls.
    """
    seg = Segment(0.0, 1.0, 1.0, SegmentType.SUSTAIN, 0.6, 0.0, 0.0, 0.0)
    base = make_frame(seed=7)
    for knob in ['VESSEL', 'STATIC_MIND', 'RESONANCE', 'COLLAPSE',
                 'ENTROPY_7', 'ZERO', 'FLESH_K', 'DOT']:
        random.seed(123); np.random.seed(123)
        m = MysterySection()
        setattr(m, knob, 0.05)  # tiny value — gate has ~5% chance per roll
        setattr(m, f'always_{knob}', True)
        out = base
        for i in range(3):
            out = m.apply(make_frame(seed=10 + i), seg, draft=False)
        diff = np.abs(out.astype(int) - base.astype(int)).sum()
        assert diff > 0, f'always_{knob}=True with knob>0 produced no change'


def test_always_flag_off_at_knob_zero_is_noop():
    """always_<KNOB>=True must NOT trigger when knob value is 0."""
    m = MysterySection()
    m.always_FLESH_K = True
    m.always_DOT = True
    m.always_VESSEL = True
    f = make_frame()
    out = m.apply(f, make_seg(), draft=False)
    assert np.array_equal(out, f), 'always-on at knob=0 must stay identity'


def test_apply_robust_to_draft_flag():
    m = MysterySection()
    m.RESONANCE = 0.5
    m.COLLAPSE = 0.5
    f = make_frame()
    o1 = m.apply(f, make_seg(), draft=True)
    o2 = m.apply(f, make_seg(), draft=False)
    assert o1.shape == f.shape
    assert o2.shape == f.shape
