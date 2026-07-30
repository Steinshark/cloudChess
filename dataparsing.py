from __future__ import annotations

import argparse
import bisect
import json
import random
import re
import sqlite3
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import bulletchess
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Paths and default filtering
# ---------------------------------------------------------------------------

DATA_ROOT = Path("E:/data/chess/parsed_data")
DATASET_ROOT = Path("D:/data/chess/lichess_db")

MIN_ELO = 2000
MIN_TIME_CONTROL = 180
MAX_ELO_DIFFERENCE = 400
MIN_PLIES = 24

# Twelve positions per game is a sensible first bootstrap dataset.
# Set to 0 on the command line to retain every position.
POSITIONS_PER_GAME = 12

SHARD_SIZE = 250_000
WRITE_BUFFER_SIZE = 4_096
RANDOM_SEED = 17


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

STATE_PLANES = 34
BOARD_HEIGHT = 8
BOARD_WIDTH = 8
POLICY_PLANES = 73
ACTION_SPACE_SIZE = 8 * 8 * POLICY_PLANES

PIECE_TYPES = (
    bulletchess.PAWN,
    bulletchess.KNIGHT,
    bulletchess.BISHOP,
    bulletchess.ROOK,
    bulletchess.QUEEN,
    bulletchess.KING,
)

OUTCOME_MAP = {
    "1-0": 1,
    "0-1": -1,
    "1/2-1/2": 0,
}


# Plane arrangement:
#
#   0-5    current player's pieces: P, N, B, R, Q, K
#   6-11   opponent's pieces:       P, N, B, R, Q, K
#   12-17  previous position, current player's pieces
#   18-23  previous position, opponent's pieces
#   24     current player can castle kingside
#   25     current player can castle queenside
#   26     opponent can castle kingside
#   27     opponent can castle queenside
#   28     en-passant target square
#   29     halfmove clock, normalized to [0, 255]
#   30     repetition count, normalized to [0, 255]
#   31     absolute side to move: white=255, black=0
#   32     fullmove number, normalized to [0, 255]
#   33     constant plane


class DatasetBuildError(Exception):
    """Base class for recoverable dataset-generation errors."""


class GameParseError(DatasetBuildError):
    """Raised when the source game record is malformed."""


class MoveParseError(DatasetBuildError):
    """Raised when a move cannot be resolved legally."""


@dataclass(slots=True)
class Datapoint:
    state: np.ndarray
    action: int
    value: int
    ply: int


@dataclass(slots=True)
class GameMetadata:
    game_id: int
    source_file: str
    source_line: int
    white_elo: int
    black_elo: int
    time_control: int
    outcome: int
    total_plies: int


@dataclass(slots=True)
class BuildConfig:
    data_root: Path
    dataset_root: Path
    min_elo: int = MIN_ELO
    min_time_control: int = MIN_TIME_CONTROL
    max_elo_difference: int = MAX_ELO_DIFFERENCE
    min_plies: int = MIN_PLIES
    positions_per_game: int = POSITIONS_PER_GAME
    shard_size: int = SHARD_SIZE
    write_buffer_size: int = WRITE_BUFFER_SIZE
    random_seed: int = RANDOM_SEED
    overwrite: bool = False


@dataclass(slots=True)
class BuildStats:
    files_seen: int = 0
    lines_seen: int = 0
    games_written: int = 0
    positions_written: int = 0

    filtered_low_elo: int = 0
    filtered_short_time_control: int = 0
    filtered_rating_difference: int = 0
    filtered_short_game: int = 0

    malformed_records: int = 0
    malformed_moves: int = 0
    skipped_files: int = 0

    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0

    started_at: float = 0.0
    completed_at: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        elapsed = (
            self.completed_at - self.started_at
            if self.completed_at and self.started_at
            else time.time() - self.started_at
        )

        return {
            "files_seen": self.files_seen,
            "lines_seen": self.lines_seen,
            "games_written": self.games_written,
            "positions_written": self.positions_written,
            "filtered_low_elo": self.filtered_low_elo,
            "filtered_short_time_control": self.filtered_short_time_control,
            "filtered_rating_difference": self.filtered_rating_difference,
            "filtered_short_game": self.filtered_short_game,
            "malformed_records": self.malformed_records,
            "malformed_moves": self.malformed_moves,
            "skipped_files": self.skipped_files,
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "draws": self.draws,
            "elapsed_seconds": elapsed,
        }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, value: object) -> None:
    """Write JSON using a temporary file followed by an atomic replacement."""
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)

    temp_path.replace(path)


def parse_time_control(value: str) -> int:
    """
    Return the base time in seconds.

    Supported examples:
        300
        300+0
        600+5
    """
    value = value.strip()

    if "+" in value:
        value = value.split("+", 1)[0]

    try:
        time_control = int(value)
    except ValueError as exc:
        raise GameParseError(f"Invalid time control: {value!r}") from exc

    if time_control < 0:
        raise GameParseError(f"Negative time control: {time_control}")

    return time_control


def repetition_key(board: bulletchess.Board) -> str:
    """
    Return the FEN portion relevant to position repetition.

    Halfmove and fullmove counters are deliberately excluded.
    """
    return " ".join(board.fen().split()[:4])


def bitboard_to_absolute_plane(bitboard: object) -> np.ndarray:
    """
    Convert a bulletchess Bitboard into a board plane.

    Output orientation before perspective adjustment:

        row 0 -> rank 8
        row 7 -> rank 1
        col 0 -> file A
        col 7 -> file H
    """
    value = int(bitboard)

    bytes_view = np.array([value], dtype="<u8").view(np.uint8)
    bits = np.unpackbits(bytes_view, bitorder="little")

    # Original bit order is A1, B1, ..., H8.
    rank_one_first = bits.reshape(8, 8)

    # CNN board orientation: rank 8 at the top.
    return rank_one_first[::-1].astype(np.uint8, copy=False)


def orient_plane(
    absolute_plane: np.ndarray,
    perspective: object,
) -> np.ndarray:
    """
    Rotate the board 180 degrees when Black is the side-to-move.

    This places the current player's home rank at the bottom of the tensor.
    """
    if perspective == bulletchess.WHITE:
        return absolute_plane

    return np.rot90(absolute_plane, 2)


def encode_piece_planes(
    board: bulletchess.Board | None,
    perspective: object,
) -> np.ndarray:
    """
    Encode twelve piece planes:

        0-5:  perspective player's P, N, B, R, Q, K
        6-11: opponent's P, N, B, R, Q, K
    """
    planes = np.zeros((12, 8, 8), dtype=np.uint8)

    if board is None:
        return planes

    opponent = perspective.opposite

    for piece_index, piece_type in enumerate(PIECE_TYPES):
        own_bitboard = board[(perspective, piece_type)]
        opponent_bitboard = board[(opponent, piece_type)]

        own_plane = bitboard_to_absolute_plane(own_bitboard)
        opponent_plane = bitboard_to_absolute_plane(opponent_bitboard)

        planes[piece_index] = orient_plane(
            own_plane,
            perspective,
        ) * 255

        planes[6 + piece_index] = orient_plane(
            opponent_plane,
            perspective,
        ) * 255

    return planes


def square_name_to_coordinates(square_name: str) -> tuple[int, int]:
    """Convert a square such as e4 to zero-based file and rank."""
    if len(square_name) != 2:
        raise ValueError(f"Invalid square: {square_name!r}")

    file_index = ord(square_name[0].lower()) - ord("a")
    rank_index = int(square_name[1]) - 1

    if not (0 <= file_index < 8 and 0 <= rank_index < 8):
        raise ValueError(f"Invalid square: {square_name!r}")

    return file_index, rank_index


def orient_coordinates(
    file_index: int,
    rank_index: int,
    perspective: object,
) -> tuple[int, int]:
    """Orient coordinates from the side-to-move perspective."""
    if perspective == bulletchess.WHITE:
        return file_index, rank_index

    return 7 - file_index, 7 - rank_index


def encode_en_passant_plane(
    board: bulletchess.Board,
    perspective: object,
) -> np.ndarray:
    plane = np.zeros((8, 8), dtype=np.uint8)
    square = board.en_passant_square

    if square is None:
        return plane

    square_name = str(square).lower()
    file_index, rank_index = square_name_to_coordinates(square_name)
    file_index, rank_index = orient_coordinates(
        file_index,
        rank_index,
        perspective,
    )

    tensor_row = 7 - rank_index
    plane[tensor_row, file_index] = 255
    return plane


def encode_board(
    board: bulletchess.Board,
    previous_board: bulletchess.Board | None,
    repetition_count: int,
) -> np.ndarray:
    """
    Generate the [34, 8, 8] uint8 representation.

    Binary planes use 0 and 255. Scalar planes also use [0, 255].
    During training, convert with:

        state = state.float() / 255.0
    """
    perspective = board.turn
    opponent = perspective.opposite

    state = np.zeros((STATE_PLANES, 8, 8), dtype=np.uint8)

    state[0:12] = encode_piece_planes(board, perspective)
    state[12:24] = encode_piece_planes(previous_board, perspective)

    castling = board.castling_rights

    if castling.kingside(perspective):
        state[24].fill(255)

    if castling.queenside(perspective):
        state[25].fill(255)

    if castling.kingside(opponent):
        state[26].fill(255)

    if castling.queenside(opponent):
        state[27].fill(255)

    state[28] = encode_en_passant_plane(board, perspective)

    halfmove_value = round(
        min(board.halfmove_clock, 100) / 100.0 * 255
    )
    state[29].fill(halfmove_value)

    # First occurrence -> 0
    # Second occurrence -> approximately half
    # Third or later occurrence -> 255
    repeated_occurrences = max(0, repetition_count - 1)
    repetition_value = round(
        min(repeated_occurrences, 2) / 2.0 * 255
    )
    state[30].fill(repetition_value)

    if board.turn == bulletchess.WHITE:
        state[31].fill(255)

    fullmove_value = round(
        min(board.fullmove_number, 200) / 200.0 * 255
    )
    state[32].fill(fullmove_value)

    state[33].fill(255)

    return state


# ---------------------------------------------------------------------------
# AlphaZero-style move encoding
# ---------------------------------------------------------------------------

# Queen-like directions are expressed as:
#
#     (delta_file_sign, delta_rank_sign)
#
# Rank increases in the current player's forward direction.
QUEEN_DIRECTIONS = (
    (0, 1),    # north
    (1, 1),    # northeast
    (1, 0),    # east
    (1, -1),   # southeast
    (0, -1),   # south
    (-1, -1),  # southwest
    (-1, 0),   # west
    (-1, 1),   # northwest
)

KNIGHT_DIRECTIONS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

UNDERPROMOTION_PIECES = (
    bulletchess.KNIGHT,
    bulletchess.BISHOP,
    bulletchess.ROOK,
)

UNDERPROMOTION_FILES = (-1, 0, 1)


def sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def encode_move(
    move: bulletchess.Move,
    perspective: object,
) -> int:
    """
    Encode a move as one integer in [0, 4671].

    Layout:

        action = origin_tensor_square * 73 + movement_plane

    Movement planes:

        0-55:  queen-like movement, 8 directions × 7 distances
        56-63: knight movement
        64-72: underpromotion, 3 directions × 3 piece types

    Queen promotions use the ordinary queen-like movement plane.
    """
    uci = move.uci()

    if len(uci) < 4:
        raise ValueError(f"Unexpected UCI move: {uci!r}")

    origin_file, origin_rank = square_name_to_coordinates(uci[0:2])
    target_file, target_rank = square_name_to_coordinates(uci[2:4])

    origin_file, origin_rank = orient_coordinates(
        origin_file,
        origin_rank,
        perspective,
    )
    target_file, target_rank = orient_coordinates(
        target_file,
        target_rank,
        perspective,
    )

    delta_file = target_file - origin_file
    delta_rank = target_rank - origin_rank

    # The state tensor has rank 8 at row 0 and the current player's
    # home rank at row 7.
    origin_row = 7 - origin_rank
    origin_square = origin_row * 8 + origin_file

    promotion = move.promotion

    if promotion in UNDERPROMOTION_PIECES:
        try:
            direction_index = UNDERPROMOTION_FILES.index(delta_file)
        except ValueError as exc:
            raise ValueError(
                f"Invalid underpromotion direction for {uci!r}"
            ) from exc

        piece_index = UNDERPROMOTION_PIECES.index(promotion)

        # Nine planes arranged as:
        #   left N/B/R, straight N/B/R, right N/B/R
        movement_plane = 64 + direction_index * 3 + piece_index

    elif (delta_file, delta_rank) in KNIGHT_DIRECTIONS:
        movement_plane = 56 + KNIGHT_DIRECTIONS.index(
            (delta_file, delta_rank)
        )

    else:
        file_sign = sign(delta_file)
        rank_sign = sign(delta_rank)

        if file_sign == 0 and rank_sign == 0:
            raise ValueError(f"Move has no displacement: {uci!r}")

        is_straight = delta_file == 0 or delta_rank == 0
        is_diagonal = abs(delta_file) == abs(delta_rank)

        if not (is_straight or is_diagonal):
            raise ValueError(
                f"Move cannot be represented in 8x8x73: {uci!r}"
            )

        direction = (file_sign, rank_sign)

        try:
            direction_index = QUEEN_DIRECTIONS.index(direction)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported movement direction for {uci!r}"
            ) from exc

        distance = max(abs(delta_file), abs(delta_rank))

        if not 1 <= distance <= 7:
            raise ValueError(
                f"Unsupported movement distance for {uci!r}: {distance}"
            )

        movement_plane = direction_index * 7 + distance - 1

    action = origin_square * POLICY_PLANES + movement_plane

    if not 0 <= action < ACTION_SPACE_SIZE:
        raise AssertionError(f"Action outside policy space: {action}")

    return action


# ---------------------------------------------------------------------------
# Move parsing
# ---------------------------------------------------------------------------

ANNOTATION_SUFFIX = re.compile(r"[!?]+$")

# Supports dataset tokens such as:
#
#     Rb4xb2
#     Ng1f3
#
# These are not standard SAN, but contain an explicit origin square.
EXPLICIT_ORIGIN_MOVE = re.compile(
    r"""
    ^
    (?P<piece>[KQRBN])
    (?P<origin>[a-h][1-8])
    x?
    (?P<target>[a-h][1-8])
    (?:=(?P<promotion>[QRBN]))?
    [+#]?
    $
    """,
    re.VERBOSE,
)

# Supports coordinate moves such as:
#
#     e2e4
#     e7e8q
#     e7e8Q
COORDINATE_MOVE = re.compile(
    r"^(?P<origin>[a-h][1-8])(?P<target>[a-h][1-8])"
    r"(?P<promotion>[qrbnQRBN])?[+#]?$"
)

PROMOTION_TO_UCI = {
    "Q": "q",
    "R": "r",
    "B": "b",
    "N": "n",
}


def clean_move_token(token: str) -> str:
    token = token.strip()

    # Normalize zero-based castling notation.
    token = token.replace("0-0-0", "O-O-O")
    token = token.replace("0-0", "O-O")

    # Remove common annotations but preserve check and mate markers.
    token = ANNOTATION_SUFFIX.sub("", token)

    # Some exports append an en-passant marker.
    token = token.replace("e.p.", "").replace("ep", "").strip()

    return token


def require_legal_move(
    board: bulletchess.Board,
    candidate: bulletchess.Move | None,
    original_token: str,
) -> bulletchess.Move:
    if candidate is None:
        raise MoveParseError(
            f"Null move is not allowed in game data: {original_token!r}"
        )

    legal_moves = board.legal_moves()

    if candidate not in legal_moves:
        raise MoveParseError(
            f"Resolved move is illegal: token={original_token!r}, "
            f"uci={candidate.uci()!r}, fen={board.fen()!r}"
        )

    return candidate


def parse_dataset_move(
    board: bulletchess.Board,
    token: str,
) -> bulletchess.Move:
    """
    Parse a move from the dataset.

    Resolution order:

        1. Standard SAN
        2. UCI/coordinate notation
        3. Explicit-origin dataset notation such as Rb4xb2
    """
    original_token = token
    token = clean_move_token(token)

    if not token:
        raise MoveParseError("Empty move token")

    # Normal path: proper standard algebraic notation.
    try:
        return bulletchess.Move.from_san(token, board)
    except ValueError:
        pass

    coordinate_match = COORDINATE_MOVE.fullmatch(token)

    if coordinate_match:
        uci = (
            coordinate_match.group("origin")
            + coordinate_match.group("target")
        )

        promotion = coordinate_match.group("promotion")
        if promotion:
            uci += promotion.lower()

        try:
            candidate = bulletchess.Move.from_uci(uci)
        except ValueError:
            candidate = None

        if candidate is not None:
            return require_legal_move(
                board,
                candidate,
                original_token,
            )

    explicit_match = EXPLICIT_ORIGIN_MOVE.fullmatch(token)

    if explicit_match:
        uci = (
            explicit_match.group("origin")
            + explicit_match.group("target")
        )

        promotion = explicit_match.group("promotion")
        if promotion:
            uci += PROMOTION_TO_UCI[promotion]

        try:
            candidate = bulletchess.Move.from_uci(uci)
        except ValueError:
            candidate = None

        if candidate is not None:
            return require_legal_move(
                board,
                candidate,
                original_token,
            )

    raise MoveParseError(
        f"Could not parse move {original_token!r} "
        f"from FEN {board.fen()!r}"
    )


# ---------------------------------------------------------------------------
# Game parsing and datapoint generation
# ---------------------------------------------------------------------------

class ChessGame:
    def __init__(self, game_line: str) -> None:
        line = game_line.strip()

        if not line:
            raise GameParseError("Empty game line")

        try:
            (
                move_text,
                white_elo,
                black_elo,
                time_control,
                outcome,
            ) = line.rsplit(".", 4)
        except ValueError as exc:
            raise GameParseError(
                "Expected moves.white_elo.black_elo.time_control.outcome"
            ) from exc

        try:
            self.white_elo = int(white_elo)
            self.black_elo = int(black_elo)
        except ValueError as exc:
            raise GameParseError(
                f"Invalid Elo values: {white_elo!r}, {black_elo!r}"
            ) from exc

        self.time_control = parse_time_control(time_control)

        try:
            self.outcome = OUTCOME_MAP[outcome.strip()]
        except KeyError as exc:
            raise GameParseError(
                f"Unsupported game outcome: {outcome!r}"
            ) from exc

        self.game_moves = [
            move.strip()
            for move in move_text.split(",")
            if move.strip()
        ]

        if not self.game_moves:
            raise GameParseError("Game contains no moves")

    @property
    def average_elo(self) -> float:
        return (self.white_elo + self.black_elo) / 2.0

    def passes_filters(
        self,
        config: BuildConfig,
        stats: BuildStats,
    ) -> bool:
        if (
            self.white_elo < config.min_elo
            or self.black_elo < config.min_elo
        ):
            stats.filtered_low_elo += 1
            return False

        if self.time_control < config.min_time_control:
            stats.filtered_short_time_control += 1
            return False

        if (
            config.max_elo_difference >= 0
            and abs(self.white_elo - self.black_elo)
            > config.max_elo_difference
        ):
            stats.filtered_rating_difference += 1
            return False

        if len(self.game_moves) < config.min_plies:
            stats.filtered_short_game += 1
            return False

        return True

    def selected_plies(
        self,
        positions_per_game: int,
        random_seed: int,
        game_identity: str,
    ) -> set[int]:
        """
        Select positions deterministically.

        Ply indexes refer to positions before the corresponding move.

        When positions_per_game is zero, every position is retained.
        """
        total_positions = len(self.game_moves)

        if (
            positions_per_game <= 0
            or positions_per_game >= total_positions
        ):
            return set(range(total_positions))

        # Avoid overtraining the first few near-identical opening positions.
        opening_start = min(6, total_positions - 1)

        candidates = list(range(opening_start, total_positions))

        if positions_per_game >= len(candidates):
            return set(candidates)

        stable_seed = hash((random_seed, game_identity)) & 0xFFFFFFFF
        rng = random.Random(stable_seed)

        # Phase-aware sampling:
        #   20% opening
        #   50% middlegame
        #   30% late game
        opening_end = min(20, total_positions)
        middle_end = min(60, total_positions)

        groups = [
            list(range(opening_start, opening_end)),
            list(range(opening_end, middle_end)),
            list(range(middle_end, total_positions)),
        ]

        targets = [
            round(positions_per_game * 0.20),
            round(positions_per_game * 0.50),
        ]
        targets.append(positions_per_game - sum(targets))

        selected: set[int] = set()

        for group, target in zip(groups, targets):
            if target <= 0 or not group:
                continue

            selected.update(
                rng.sample(group, min(target, len(group)))
            )

        # Fill any shortfall from all remaining candidate positions.
        if len(selected) < positions_per_game:
            remaining = [
                ply for ply in candidates if ply not in selected
            ]

            selected.update(
                rng.sample(
                    remaining,
                    min(
                        positions_per_game - len(selected),
                        len(remaining),
                    ),
                )
            )

        return selected

    def generate_datapoints(
        self,
        selected_plies: set[int],
    ) -> list[Datapoint]:
        datapoints: list[Datapoint] = []

        board = bulletchess.Board()
        previous_board: bulletchess.Board | None = None

        repetitions: Counter[str] = Counter()
        repetitions[repetition_key(board)] = 1

        for ply, move_token in enumerate(self.game_moves):
            move = parse_dataset_move(board, move_token)

            if ply in selected_plies:
                state = encode_board(
                    board=board,
                    previous_board=previous_board,
                    repetition_count=repetitions[
                        repetition_key(board)
                    ],
                )

                action = encode_move(
                    move=move,
                    perspective=board.turn,
                )

                # Source outcome is from White's perspective.
                # The value target must be from the player-to-move perspective.
                value = (
                    self.outcome
                    if board.turn == bulletchess.WHITE
                    else -self.outcome
                )

                datapoints.append(
                    Datapoint(
                        state=state,
                        action=action,
                        value=value,
                        ply=ply,
                    )
                )

            previous_board = board.copy()
            board.apply(move)

            repetitions[repetition_key(board)] += 1

        return datapoints


# ---------------------------------------------------------------------------
# Indexed HDF5 writer
# ---------------------------------------------------------------------------

class ShardedDatasetWriter:
    def __init__(
        self,
        root: Path,
        shard_size: int,
        write_buffer_size: int,
    ) -> None:
        self.root = root
        self.shards_root = root / "shards"
        self.shard_size = shard_size
        self.write_buffer_size = write_buffer_size

        self.shards_root.mkdir(parents=True, exist_ok=True)

        self.database = sqlite3.connect(root / "index.sqlite3")
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=NORMAL")
        self.database.execute("PRAGMA temp_store=MEMORY")

        self._create_database_schema()

        self.current_file: h5py.File | None = None
        self.current_datasets: dict[str, h5py.Dataset] = {}
        self.current_shard_id = -1
        self.current_shard_count = 0
        self.global_count = 0

        self.shard_manifest: list[dict[str, int | str]] = []

        self._open_next_shard()

    def _create_database_schema(self) -> None:
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
                game_id       INTEGER PRIMARY KEY,
                source_file   TEXT NOT NULL,
                source_line   INTEGER NOT NULL,
                white_elo     INTEGER NOT NULL,
                black_elo     INTEGER NOT NULL,
                time_control  INTEGER NOT NULL,
                outcome       INTEGER NOT NULL,
                total_plies   INTEGER NOT NULL,
                positions     INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS game_segments (
                game_id       INTEGER NOT NULL,
                shard_id      INTEGER NOT NULL,
                local_start   INTEGER NOT NULL,
                count         INTEGER NOT NULL,
                global_start  INTEGER NOT NULL,
                FOREIGN KEY(game_id) REFERENCES games(game_id)
            );

            CREATE INDEX IF NOT EXISTS idx_game_segments_game_id
            ON game_segments(game_id);

            CREATE INDEX IF NOT EXISTS idx_games_source
            ON games(source_file, source_line);

            CREATE INDEX IF NOT EXISTS idx_games_elos
            ON games(white_elo, black_elo);

            CREATE INDEX IF NOT EXISTS idx_games_time_control
            ON games(time_control);
            """
        )
        self.database.commit()

    def _open_next_shard(self) -> None:
        self._close_current_shard()

        self.current_shard_id += 1
        self.current_shard_count = 0

        filename = f"shard_{self.current_shard_id:05d}.h5"
        path = self.shards_root / filename

        self.current_file = h5py.File(path, "w")

        chunk_rows = min(512, self.shard_size)

        self.current_datasets = {
            "states": self.current_file.create_dataset(
                "states",
                shape=(self.shard_size, STATE_PLANES, 8, 8),
                maxshape=(self.shard_size, STATE_PLANES, 8, 8),
                dtype=np.uint8,
                chunks=(chunk_rows, STATE_PLANES, 8, 8),
                compression="lzf",
                shuffle=True,
            ),
            "actions": self.current_file.create_dataset(
                "actions",
                shape=(self.shard_size,),
                maxshape=(self.shard_size,),
                dtype=np.uint16,
                chunks=(min(8192, self.shard_size),),
                compression="lzf",
            ),
            "values": self.current_file.create_dataset(
                "values",
                shape=(self.shard_size,),
                maxshape=(self.shard_size,),
                dtype=np.int8,
                chunks=(min(8192, self.shard_size),),
                compression="lzf",
            ),
            "game_ids": self.current_file.create_dataset(
                "game_ids",
                shape=(self.shard_size,),
                maxshape=(self.shard_size,),
                dtype=np.uint64,
                chunks=(min(8192, self.shard_size),),
                compression="lzf",
            ),
            "plies": self.current_file.create_dataset(
                "plies",
                shape=(self.shard_size,),
                maxshape=(self.shard_size,),
                dtype=np.uint16,
                chunks=(min(8192, self.shard_size),),
                compression="lzf",
            ),
            "white_elos": self.current_file.create_dataset(
                "white_elos",
                shape=(self.shard_size,),
                maxshape=(self.shard_size,),
                dtype=np.uint16,
                chunks=(min(8192, self.shard_size),),
                compression="lzf",
            ),
            "black_elos": self.current_file.create_dataset(
                "black_elos",
                shape=(self.shard_size,),
                maxshape=(self.shard_size,),
                dtype=np.uint16,
                chunks=(min(8192, self.shard_size),),
                compression="lzf",
            ),
            "time_controls": self.current_file.create_dataset(
                "time_controls",
                shape=(self.shard_size,),
                maxshape=(self.shard_size,),
                dtype=np.uint32,
                chunks=(min(8192, self.shard_size),),
                compression="lzf",
            ),
        }

        self.current_file.attrs["state_planes"] = STATE_PLANES
        self.current_file.attrs["policy_planes"] = POLICY_PLANES
        self.current_file.attrs["action_space_size"] = ACTION_SPACE_SIZE
        self.current_file.attrs["state_dtype"] = "uint8_0_to_255"

    def _close_current_shard(self) -> None:
        if self.current_file is None:
            return

        used = self.current_shard_count

        for dataset in self.current_datasets.values():
            new_shape = (used, *dataset.shape[1:])
            dataset.resize(new_shape)

        filename = Path(self.current_file.filename).name

        self.current_file.attrs["positions"] = used
        self.current_file.flush()
        self.current_file.close()

        if used > 0:
            self.shard_manifest.append(
                {
                    "shard_id": self.current_shard_id,
                    "filename": f"shards/{filename}",
                    "global_start": self.global_count - used,
                    "count": used,
                }
            )
        else:
            # Do not retain an empty trailing shard.
            (self.shards_root / filename).unlink(missing_ok=True)

        self.current_file = None
        self.current_datasets = {}

    def _write_segment(
        self,
        metadata: GameMetadata,
        datapoints: Sequence[Datapoint],
    ) -> None:
        if not datapoints:
            return

        count = len(datapoints)
        local_start = self.current_shard_count
        local_end = local_start + count
        global_start = self.global_count

        states = np.stack(
            [point.state for point in datapoints],
            axis=0,
        )
        actions = np.fromiter(
            (point.action for point in datapoints),
            dtype=np.uint16,
            count=count,
        )
        values = np.fromiter(
            (point.value for point in datapoints),
            dtype=np.int8,
            count=count,
        )
        plies = np.fromiter(
            (point.ply for point in datapoints),
            dtype=np.uint16,
            count=count,
        )

        self.current_datasets["states"][local_start:local_end] = states
        self.current_datasets["actions"][local_start:local_end] = actions
        self.current_datasets["values"][local_start:local_end] = values
        self.current_datasets["plies"][local_start:local_end] = plies

        self.current_datasets["game_ids"][local_start:local_end] = (
            metadata.game_id
        )
        self.current_datasets["white_elos"][local_start:local_end] = (
            metadata.white_elo
        )
        self.current_datasets["black_elos"][local_start:local_end] = (
            metadata.black_elo
        )
        self.current_datasets["time_controls"][local_start:local_end] = (
            metadata.time_control
        )

        self.database.execute(
            """
            INSERT INTO game_segments (
                game_id,
                shard_id,
                local_start,
                count,
                global_start
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                metadata.game_id,
                self.current_shard_id,
                local_start,
                count,
                global_start,
            ),
        )

        self.current_shard_count += count
        self.global_count += count

    def add_game(
        self,
        metadata: GameMetadata,
        datapoints: Sequence[Datapoint],
    ) -> None:
        self.database.execute(
            """
            INSERT INTO games (
                game_id,
                source_file,
                source_line,
                white_elo,
                black_elo,
                time_control,
                outcome,
                total_plies,
                positions
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.game_id,
                metadata.source_file,
                metadata.source_line,
                metadata.white_elo,
                metadata.black_elo,
                metadata.time_control,
                metadata.outcome,
                metadata.total_plies,
                len(datapoints),
            ),
        )

        position = 0

        while position < len(datapoints):
            available = self.shard_size - self.current_shard_count

            if available == 0:
                self._open_next_shard()
                available = self.shard_size

            segment_size = min(
                available,
                len(datapoints) - position,
            )

            self._write_segment(
                metadata,
                datapoints[position : position + segment_size],
            )

            position += segment_size

    def commit(self) -> None:
        self.database.commit()

        if self.current_file is not None:
            self.current_file.flush()

    def close(self) -> None:
        self.commit()
        self._close_current_shard()
        self.database.close()

    def write_manifest(
        self,
        config: BuildConfig,
        stats: BuildStats,
    ) -> None:
        manifest = {
            "format_version": 1,
            "state_shape": [STATE_PLANES, 8, 8],
            "state_dtype": "uint8",
            "state_scale": 255,
            "action_dtype": "uint16",
            "action_space": "8x8x73",
            "action_space_size": ACTION_SPACE_SIZE,
            "value_dtype": "int8",
            "value_perspective": "side_to_move",
            "total_positions": self.global_count,
            "total_games": stats.games_written,
            "filters": {
                "minimum_elo": config.min_elo,
                "minimum_time_control": config.min_time_control,
                "maximum_elo_difference": config.max_elo_difference,
                "minimum_plies": config.min_plies,
                "positions_per_game": config.positions_per_game,
            },
            "shards": self.shard_manifest,
        }

        atomic_write_json(self.root / "manifest.json", manifest)


# ---------------------------------------------------------------------------
# Dataset reader
# ---------------------------------------------------------------------------

class ChessPositionDataset:
    """
    Random-access reader suitable for PyTorch Dataset wrapping.

    Examples:

        dataset = ChessPositionDataset(DATASET_ROOT)

        item = dataset[100_000]
        state = item["state"].astype(np.float32) / 255.0

        game_positions = dataset.get_game(42)
    """

    def __init__(
        self,
        root: str | Path,
        max_open_shards: int = 2,
    ) -> None:
        self.root = Path(root)

        with (self.root / "manifest.json").open(
            "r",
            encoding="utf-8",
        ) as file:
            self.manifest = json.load(file)

        self.shards = self.manifest["shards"]
        self.total_positions = int(
            self.manifest["total_positions"]
        )

        self.shard_starts = [
            int(shard["global_start"])
            for shard in self.shards
        ]

        self.max_open_shards = max_open_shards
        self.open_shards: OrderedDict[int, h5py.File] = OrderedDict()

        self.database = sqlite3.connect(
            self.root / "index.sqlite3"
        )
        self.database.row_factory = sqlite3.Row

    def __len__(self) -> int:
        return self.total_positions

    def _get_shard(self, shard_id: int) -> h5py.File:
        if shard_id in self.open_shards:
            handle = self.open_shards.pop(shard_id)
            self.open_shards[shard_id] = handle
            return handle

        shard_info = self.shards[shard_id]
        path = self.root / shard_info["filename"]

        handle = h5py.File(path, "r")
        self.open_shards[shard_id] = handle

        while len(self.open_shards) > self.max_open_shards:
            _, old_handle = self.open_shards.popitem(last=False)
            old_handle.close()

        return handle

    def _resolve_global_index(
        self,
        index: int,
    ) -> tuple[int, int]:
        if index < 0:
            index += self.total_positions

        if not 0 <= index < self.total_positions:
            raise IndexError(index)

        shard_id = bisect.bisect_right(
            self.shard_starts,
            index,
        ) - 1

        shard = self.shards[shard_id]
        local_index = index - int(shard["global_start"])

        return shard_id, local_index

    def _read_local_position(
        self,
        shard_id: int,
        local_index: int,
    ) -> dict[str, int | np.ndarray]:
        shard = self._get_shard(shard_id)

        return {
            "state": shard["states"][local_index],
            "action": int(shard["actions"][local_index]),
            "value": int(shard["values"][local_index]),
            "game_id": int(shard["game_ids"][local_index]),
            "ply": int(shard["plies"][local_index]),
            "white_elo": int(shard["white_elos"][local_index]),
            "black_elo": int(shard["black_elos"][local_index]),
            "time_control": int(
                shard["time_controls"][local_index]
            ),
        }

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, int | np.ndarray]:
        shard_id, local_index = self._resolve_global_index(index)
        return self._read_local_position(shard_id, local_index)

    def get_game_metadata(self, game_id: int) -> dict[str, object]:
        row = self.database.execute(
            "SELECT * FROM games WHERE game_id = ?",
            (game_id,),
        ).fetchone()

        if row is None:
            raise KeyError(f"Unknown game_id: {game_id}")

        return dict(row)

    def get_game(
        self,
        game_id: int,
    ) -> list[dict[str, int | np.ndarray]]:
        segments = self.database.execute(
            """
            SELECT shard_id, local_start, count
            FROM game_segments
            WHERE game_id = ?
            ORDER BY global_start
            """,
            (game_id,),
        ).fetchall()

        if not segments:
            raise KeyError(f"Unknown game_id: {game_id}")

        positions: list[dict[str, int | np.ndarray]] = []

        for segment in segments:
            shard_id = int(segment["shard_id"])
            local_start = int(segment["local_start"])
            count = int(segment["count"])

            for local_index in range(
                local_start,
                local_start + count,
            ):
                positions.append(
                    self._read_local_position(
                        shard_id,
                        local_index,
                    )
                )

        return positions

    def find_games(
        self,
        minimum_elo: int | None = None,
        minimum_time_control: int | None = None,
        outcome: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        conditions: list[str] = []
        values: list[int] = []

        if minimum_elo is not None:
            conditions.append(
                "white_elo >= ? AND black_elo >= ?"
            )
            values.extend([minimum_elo, minimum_elo])

        if minimum_time_control is not None:
            conditions.append("time_control >= ?")
            values.append(minimum_time_control)

        if outcome is not None:
            conditions.append("outcome = ?")
            values.append(outcome)

        where_clause = (
            " WHERE " + " AND ".join(conditions)
            if conditions
            else ""
        )

        query = (
            "SELECT * FROM games"
            + where_clause
            + " ORDER BY game_id LIMIT ?"
        )
        values.append(limit)

        rows = self.database.execute(
            query,
            values,
        ).fetchall()

        return [dict(row) for row in rows]

    def close(self) -> None:
        for handle in self.open_shards.values():
            handle.close()

        self.open_shards.clear()
        self.database.close()

    def __enter__(self) -> "ChessPositionDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def prepare_output_directory(config: BuildConfig) -> None:
    root = config.dataset_root

    if root.exists() and any(root.iterdir()):
        if not config.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {root}\n"
                "Pass --overwrite to replace the generated files."
            )

        for path in sorted(
            root.rglob("*"),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    root.mkdir(parents=True, exist_ok=True)


def input_files(data_root: Path) -> Iterator[Path]:
    for path in sorted(data_root.iterdir()):
        if path.is_file():
            yield path


def append_error(
    error_file: object,
    *,
    source_file: str,
    source_line: int,
    category: str,
    error: Exception,
    line: str,
) -> None:
    record = {
        "source_file": source_file,
        "source_line": source_line,
        "category": category,
        "error": str(error),
        "line_preview": line[:500],
    }

    error_file.write(json.dumps(record) + "\n")


def build_dataset(config: BuildConfig) -> BuildStats:
    if not config.data_root.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {config.data_root}"
        )

    prepare_output_directory(config)

    stats = BuildStats(started_at=time.time())

    writer = ShardedDatasetWriter(
        root=config.dataset_root,
        shard_size=config.shard_size,
        write_buffer_size=config.write_buffer_size,
    )

    errors_path = config.dataset_root / "errors.jsonl"

    next_game_id = 0
    games_since_commit = 0

    try:
        with errors_path.open("w", encoding="utf-8") as error_file:
            for game_file in input_files(config.data_root):
                stats.files_seen += 1

                try:
                    source = game_file.open(
                        "r",
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError:
                    stats.skipped_files += 1
                    continue

                with source:
                    for source_line, line in enumerate(source, start=1):
                        stats.lines_seen += 1

                        try:
                            game = ChessGame(line)
                        except GameParseError as exc:
                            stats.malformed_records += 1

                            append_error(
                                error_file,
                                source_file=game_file.name,
                                source_line=source_line,
                                category="record",
                                error=exc,
                                line=line,
                            )
                            continue

                        if not game.passes_filters(config, stats):
                            continue

                        identity = (
                            f"{game_file.name}:{source_line}:"
                            f"{game.white_elo}:{game.black_elo}"
                        )

                        selected_plies = game.selected_plies(
                            positions_per_game=config.positions_per_game,
                            random_seed=config.random_seed,
                            game_identity=identity,
                        )

                        try:
                            datapoints = game.generate_datapoints(
                                selected_plies
                            )
                        except (
                            MoveParseError,
                            ValueError,
                            AssertionError,
                        ) as exc:
                            stats.malformed_moves += 1

                            append_error(
                                error_file,
                                source_file=game_file.name,
                                source_line=source_line,
                                category="move",
                                error=exc,
                                line=line,
                            )
                            continue

                        if not datapoints:
                            continue

                        metadata = GameMetadata(
                            game_id=next_game_id,
                            source_file=game_file.name,
                            source_line=source_line,
                            white_elo=game.white_elo,
                            black_elo=game.black_elo,
                            time_control=game.time_control,
                            outcome=game.outcome,
                            total_plies=len(game.game_moves),
                        )

                        writer.add_game(metadata, datapoints)

                        next_game_id += 1
                        games_since_commit += 1

                        stats.games_written += 1
                        stats.positions_written += len(datapoints)

                        if game.outcome == 1:
                            stats.white_wins += 1
                        elif game.outcome == -1:
                            stats.black_wins += 1
                        else:
                            stats.draws += 1

                        if games_since_commit >= 1_000:
                            writer.commit()
                            games_since_commit = 0

                        if stats.games_written % 10_000 == 0:
                            elapsed = time.time() - stats.started_at
                            rate = (
                                stats.positions_written / elapsed
                                if elapsed > 0
                                else 0.0
                            )

                            print(
                                f"Games: {stats.games_written:,} | "
                                f"Positions: {stats.positions_written:,} | "
                                f"Lines: {stats.lines_seen:,} | "
                                f"Rate: {rate:,.0f} positions/s"
                            )

        stats.completed_at = time.time()

        writer.commit()
        writer.close()
        writer.write_manifest(config, stats)

        atomic_write_json(
            config.dataset_root / "stats.json",
            stats.as_dict(),
        )

        return stats

    except BaseException:
        writer.close()
        raise


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the sharded neural-chess dataset."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=MIN_ELO,
    )
    parser.add_argument(
        "--min-time-control",
        type=int,
        default=MIN_TIME_CONTROL,
    )
    parser.add_argument(
        "--max-elo-difference",
        type=int,
        default=MAX_ELO_DIFFERENCE,
        help="Use -1 to disable this filter.",
    )
    parser.add_argument(
        "--min-plies",
        type=int,
        default=MIN_PLIES,
    )
    parser.add_argument(
        "--positions-per-game",
        type=int,
        default=POSITIONS_PER_GAME,
        help="Use 0 to retain every position.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    config = BuildConfig(
        data_root=args.data_root,
        dataset_root=args.dataset_root,
        min_elo=args.min_elo,
        min_time_control=args.min_time_control,
        max_elo_difference=args.max_elo_difference,
        min_plies=args.min_plies,
        positions_per_game=args.positions_per_game,
        shard_size=args.shard_size,
        random_seed=args.seed,
        overwrite=args.overwrite,
    )

    stats = build_dataset(config)

    print("\nDataset generation complete.")
    print(json.dumps(stats.as_dict(), indent=2))


if __name__ == "__main__":
    main()