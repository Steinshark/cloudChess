from __future__ import annotations


# Short, legal UCI opening prefixes. Arena games are paired with colors reversed.
DEFAULT_OPENINGS: tuple[tuple[str, ...], ...] = (
    (),
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("e2e4", "c7c5", "g1f3", "b8c6"),
    ("e2e4", "e7e6", "d2d4", "d7d5"),
    ("e2e4", "c7c6", "d2d4", "d7d5"),
    ("d2d4", "d7d5", "c2c4", "e7e6"),
    ("d2d4", "g8f6", "c2c4", "g7g6"),
    ("d2d4", "g8f6", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "b1c3", "g8f6"),
    ("c2c4", "c7c5", "g1f3", "g8f6"),
    ("g1f3", "d7d5", "g2g3", "g8f6"),
    ("e2e4", "e7e5", "f1c4", "g8f6"),
    ("e2e4", "e7e5", "g1f3", "g8f6"),
    ("e2e4", "d7d5", "e4d5", "d8d5"),
    ("e2e4", "g8f6", "e4e5", "f6d5"),
    ("d2d4", "f7f5", "g2g3", "g8f6"),
    ("b2b3", "e7e5", "c1b2", "b8c6"),
    ("g2g3", "d7d5", "f1g2", "e7e5"),
    ("b1c3", "d7d5", "e2e4", "d5d4"),
)
