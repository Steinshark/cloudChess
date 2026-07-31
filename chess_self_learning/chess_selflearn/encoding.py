from __future__ import annotations

from collections import Counter

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


def opposite(color: object) -> object:
    return bulletchess.BLACK if color == bulletchess.WHITE else bulletchess.WHITE


def repetition_key(board: bulletchess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def initial_repetition_counts(board: bulletchess.Board) -> Counter[str]:
    """Reconstruct legal-position occurrence counts from the board history."""
    copy = board.copy()
    positions: list[str] = [repetition_key(copy)]
    while copy.history:
        copy.undo()
        positions.append(repetition_key(copy))
    return Counter(positions)


def previous_board(board: bulletchess.Board) -> bulletchess.Board | None:
    if not board.history:
        return None
    copy = board.copy()
    copy.undo()
    return copy


def bitboard_to_absolute_plane(bitboard: object) -> np.ndarray:
    value = int(bitboard)
    bytes_view = np.array([value], dtype="<u8").view(np.uint8)
    bits = np.unpackbits(bytes_view, bitorder="little")
    return bits.reshape(8, 8)[::-1].astype(np.uint8, copy=False)


def orient_plane(plane: np.ndarray, perspective: object) -> np.ndarray:
    if perspective == bulletchess.WHITE:
        return plane
    return np.rot90(plane, 2)


def encode_piece_planes(
    board: bulletchess.Board | None,
    perspective: object,
) -> np.ndarray:
    planes = np.zeros((12, 8, 8), dtype=np.uint8)
    if board is None:
        return planes

    enemy = opposite(perspective)
    for index, piece_type in enumerate(PIECE_TYPES):
        own = bitboard_to_absolute_plane(board[perspective, piece_type])
        other = bitboard_to_absolute_plane(board[enemy, piece_type])
        planes[index] = orient_plane(own, perspective) * 255
        planes[6 + index] = orient_plane(other, perspective) * 255
    return planes


def square_name_to_coordinates(name: str) -> tuple[int, int]:
    if len(name) != 2:
        raise ValueError(f"Invalid square: {name!r}")
    file_index = ord(name[0].lower()) - ord("a")
    rank_index = int(name[1]) - 1
    if not (0 <= file_index < 8 and 0 <= rank_index < 8):
        raise ValueError(f"Invalid square: {name!r}")
    return file_index, rank_index


def orient_coordinates(
    file_index: int,
    rank_index: int,
    perspective: object,
) -> tuple[int, int]:
    if perspective == bulletchess.WHITE:
        return file_index, rank_index
    return 7 - file_index, 7 - rank_index


def encode_en_passant(
    board: bulletchess.Board,
    perspective: object,
) -> np.ndarray:
    plane = np.zeros((8, 8), dtype=np.uint8)
    if board.en_passant_square is None:
        return plane
    file_index, rank_index = square_name_to_coordinates(
        str(board.en_passant_square).lower()
    )
    file_index, rank_index = orient_coordinates(
        file_index,
        rank_index,
        perspective,
    )
    plane[7 - rank_index, file_index] = 255
    return plane


def encode_board(
    board: bulletchess.Board,
    prior_board: bulletchess.Board | None = None,
    repetition_count: int = 1,
) -> np.ndarray:
    perspective = board.turn
    enemy = opposite(perspective)
    state = np.zeros((STATE_PLANES, 8, 8), dtype=np.uint8)

    state[0:12] = encode_piece_planes(board, perspective)
    state[12:24] = encode_piece_planes(prior_board, perspective)

    rights = board.castling_rights
    if rights.kingside(perspective):
        state[24].fill(255)
    if rights.queenside(perspective):
        state[25].fill(255)
    if rights.kingside(enemy):
        state[26].fill(255)
    if rights.queenside(enemy):
        state[27].fill(255)

    state[28] = encode_en_passant(board, perspective)
    state[29].fill(round(min(board.halfmove_clock, 100) / 100.0 * 255))
    repeated = max(0, repetition_count - 1)
    state[30].fill(round(min(repeated, 2) / 2.0 * 255))
    if board.turn == bulletchess.WHITE:
        state[31].fill(255)
    state[32].fill(round(min(board.fullmove_number, 200) / 200.0 * 255))
    state[33].fill(255)
    return state


def sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def encode_move(move: bulletchess.Move, perspective: object) -> int:
    uci = move.uci()
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
    origin_square = (7 - origin_rank) * 8 + origin_file
    promotion = move.promotion

    if promotion in UNDERPROMOTION_PIECES:
        direction_index = UNDERPROMOTION_FILES.index(delta_file)
        piece_index = UNDERPROMOTION_PIECES.index(promotion)
        movement_plane = 64 + direction_index * 3 + piece_index
    elif (delta_file, delta_rank) in KNIGHT_DIRECTIONS:
        movement_plane = 56 + KNIGHT_DIRECTIONS.index((delta_file, delta_rank))
    else:
        straight = delta_file == 0 or delta_rank == 0
        diagonal = abs(delta_file) == abs(delta_rank)
        if not (straight or diagonal):
            raise ValueError(f"Unrepresentable move: {uci}")
        direction = (sign(delta_file), sign(delta_rank))
        distance = max(abs(delta_file), abs(delta_rank))
        movement_plane = QUEEN_DIRECTIONS.index(direction) * 7 + distance - 1

    action = origin_square * POLICY_PLANES + movement_plane
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise ValueError(f"Action outside policy space: {action}")
    return action


def legal_action_map(
    board: bulletchess.Board,
) -> tuple[list[int], list[bulletchess.Move]]:
    moves = board.legal_moves()
    actions = [encode_move(move, board.turn) for move in moves]
    if len(set(actions)) != len(actions):
        raise RuntimeError(f"Action collision in position: {board.fen()}")
    return actions, moves
