from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import bulletchess
import numpy as np


STATE_PLANES = 34
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

QUEEN_DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
QUEEN_DIRECTION_TO_INDEX = {
    direction: index for index, direction in enumerate(QUEEN_DIRECTIONS)
}

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
KNIGHT_DIRECTION_TO_INDEX = {
    direction: index for index, direction in enumerate(KNIGHT_DIRECTIONS)
}

UNDERPROMOTION_PIECES = (
    bulletchess.KNIGHT,
    bulletchess.BISHOP,
    bulletchess.ROOK,
)
UNDERPROMOTION_PIECE_TO_INDEX = {
    piece: index for index, piece in enumerate(UNDERPROMOTION_PIECES)
}
UNDERPROMOTION_FILES = (-1, 0, 1)
UNDERPROMOTION_FILE_TO_INDEX = {
    file_delta: index for index, file_delta in enumerate(UNDERPROMOTION_FILES)
}

# Bitboard square indexes use A1=0, B1=1, ..., H8=63.  Broadcasting this
# vector across all piece bitboards converts all 12 or 24 planes at once.
_BIT_SHIFTS = np.arange(64, dtype=np.uint64)
_UINT64_ONE = np.uint64(1)
_UINT8_FULL = np.uint8(255)


def opposite(color: object) -> object:
    return bulletchess.BLACK if color == bulletchess.WHITE else bulletchess.WHITE


def repetition_key(board: bulletchess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def initial_repetition_counts(board: bulletchess.Board) -> Counter[str]:
    """Reconstruct position occurrence counts once when a game/tree is built."""
    copy = board.copy()
    positions: list[str] = [repetition_key(copy)]
    while copy.history:
        copy.undo()
        positions.append(repetition_key(copy))
    return Counter(positions)


def _piece_bitboard_values(
    board: bulletchess.Board,
    perspective: object,
) -> np.ndarray:
    """Return own then opponent P/N/B/R/Q/K bitboards as uint64."""
    enemy = opposite(perspective)
    return np.fromiter(
        (
            *(
                int(board[perspective, piece_type])
                for piece_type in PIECE_TYPES
            ),
            *(int(board[enemy, piece_type]) for piece_type in PIECE_TYPES),
        ),
        dtype=np.uint64,
        count=12,
    )


def bitboards_to_oriented_planes(
    bitboards: Sequence[int] | np.ndarray,
    perspective: object,
) -> np.ndarray:
    """Vectorize any number of uint64 bitboards into [N, 8, 8] uint8 planes."""
    values = np.asarray(bitboards, dtype=np.uint64).reshape(-1)
    bits = ((values[:, None] >> _BIT_SHIFTS[None, :]) & _UINT64_ONE).astype(
        np.uint8,
        copy=False,
    )

    # Convert A1-first bit order into tensor rows rank 8 -> rank 1.
    planes = bits.reshape(-1, 8, 8)[:, ::-1, :]
    if perspective == bulletchess.BLACK:
        # Current player's home rank remains tensor row 7 for either color.
        planes = planes[:, ::-1, ::-1]
    return planes * _UINT8_FULL


def _assemble_state(
    board: bulletchess.Board,
    *,
    perspective: object,
    current_bitboards: np.ndarray,
    prior_bitboards: np.ndarray | None,
    repetition_count: int,
) -> np.ndarray:
    state = np.zeros((STATE_PLANES, 8, 8), dtype=np.uint8)

    if prior_bitboards is None:
        combined = current_bitboards
        converted = bitboards_to_oriented_planes(combined, perspective)
        state[0:12] = converted
    else:
        combined = np.concatenate((current_bitboards, prior_bitboards))
        converted = bitboards_to_oriented_planes(combined, perspective)
        state[0:24] = converted

    enemy = opposite(perspective)
    rights = board.castling_rights
    if rights.kingside(perspective):
        state[24].fill(255)
    if rights.queenside(perspective):
        state[25].fill(255)
    if rights.kingside(enemy):
        state[26].fill(255)
    if rights.queenside(enemy):
        state[27].fill(255)

    if board.en_passant_square is not None:
        square_index = board.en_passant_square.index()
        file_index = square_index & 7
        rank_index = square_index >> 3
        if perspective == bulletchess.BLACK:
            file_index = 7 - file_index
            rank_index = 7 - rank_index
        state[28, 7 - rank_index, file_index] = 255

    state[29].fill(round(min(board.halfmove_clock, 100) / 100.0 * 255))
    repeated = max(0, repetition_count - 1)
    state[30].fill(round(min(repeated, 2) / 2.0 * 255))
    if board.turn == bulletchess.WHITE:
        state[31].fill(255)
    state[32].fill(round(min(board.fullmove_number, 200) / 200.0 * 255))
    state[33].fill(255)
    return state


def encode_board(
    board: bulletchess.Board,
    prior_board: bulletchess.Board | None = None,
    repetition_count: int = 1,
) -> np.ndarray:
    """Encode a board, optionally using a separately supplied prior position.

    Search and self-play hot paths should use :func:`encode_board_with_history`,
    which obtains the previous position with one in-place undo/apply pair and
    never deep-copies the board or its move history.
    """
    perspective = board.turn
    current = _piece_bitboard_values(board, perspective)
    prior = (
        None
        if prior_board is None
        else _piece_bitboard_values(prior_board, perspective)
    )
    return _assemble_state(
        board,
        perspective=perspective,
        current_bitboards=current,
        prior_bitboards=prior,
        repetition_count=repetition_count,
    )


def encode_board_with_history(
    board: bulletchess.Board,
    repetition_count: int = 1,
) -> np.ndarray:
    """Encode current and previous piece planes without copying ``board``.

    The board is restored exactly before returning. Metadata planes always use
    the current position; only planes 12-23 are read from the previous position.
    """
    perspective = board.turn
    current = _piece_bitboard_values(board, perspective)
    prior: np.ndarray | None = None

    if board.history:
        previous_move = board.undo()
        try:
            prior = _piece_bitboard_values(board, perspective)
        finally:
            board.apply(previous_move)

    return _assemble_state(
        board,
        perspective=perspective,
        current_bitboards=current,
        prior_bitboards=prior,
        repetition_count=repetition_count,
    )


def orient_square_index(square_index: int, perspective: object) -> tuple[int, int]:
    file_index = square_index & 7
    rank_index = square_index >> 3
    if perspective == bulletchess.BLACK:
        return 7 - file_index, 7 - rank_index
    return file_index, rank_index


def sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def encode_move(move: bulletchess.Move, perspective: object) -> int:
    """Encode a legal move into the square-major 8x8x73 policy space."""
    origin_file, origin_rank = orient_square_index(
        move.origin.index(),
        perspective,
    )
    target_file, target_rank = orient_square_index(
        move.destination.index(),
        perspective,
    )

    delta_file = target_file - origin_file
    delta_rank = target_rank - origin_rank
    origin_square = (7 - origin_rank) * 8 + origin_file
    promotion = move.promotion

    if promotion in UNDERPROMOTION_PIECE_TO_INDEX:
        try:
            direction_index = UNDERPROMOTION_FILE_TO_INDEX[delta_file]
        except KeyError as exc:
            raise ValueError(
                f"Invalid underpromotion direction for {move.uci()!r}"
            ) from exc
        piece_index = UNDERPROMOTION_PIECE_TO_INDEX[promotion]
        movement_plane = 64 + direction_index * 3 + piece_index
    else:
        knight_index = KNIGHT_DIRECTION_TO_INDEX.get((delta_file, delta_rank))
        if knight_index is not None:
            movement_plane = 56 + knight_index
        else:
            straight = delta_file == 0 or delta_rank == 0
            diagonal = abs(delta_file) == abs(delta_rank)
            if not (straight or diagonal):
                raise ValueError(f"Unrepresentable move: {move.uci()}")

            direction = (sign(delta_file), sign(delta_rank))
            try:
                direction_index = QUEEN_DIRECTION_TO_INDEX[direction]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported movement direction: {move.uci()}"
                ) from exc

            distance = max(abs(delta_file), abs(delta_rank))
            if not 1 <= distance <= 7:
                raise ValueError(
                    f"Unsupported movement distance for {move.uci()}: {distance}"
                )
            movement_plane = direction_index * 7 + distance - 1

    action = origin_square * POLICY_PLANES + movement_plane
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise ValueError(f"Action outside policy space: {action}")
    return action


def legal_action_map(
    board: bulletchess.Board,
) -> tuple[np.ndarray, list[bulletchess.Move]]:
    moves = board.legal_moves()
    actions = np.fromiter(
        (encode_move(move, board.turn) for move in moves),
        dtype=np.int64,
        count=len(moves),
    )
    if np.unique(actions).size != actions.size:
        raise RuntimeError(f"Action collision in position: {board.fen()}")
    return actions, moves
