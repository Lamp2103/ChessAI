import chess.pgn
import numpy as np
import tensorflow as tf

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_vector(board):
    """Chuyển bàn cờ (chess.Board) thành tensor shape (8, 8, 12)"""
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            index = PIECE_TO_INDEX[piece.symbol()]
            tensor[row, col, index] = 1.0
    return tensor

def move_to_index(move):
    """Chuyển nước đi thành nhãn số hóa"""
    # Gán index cho tất cả các nước đi hợp lệ (4096 khả năng từ 64x64)
    return move.from_square * 64 + move.to_square

def extract_data_from_pgn(pgn_file_path, max_games=500):
    games_data = []
    with open(pgn_file_path, 'r', encoding='utf-8') as pgn:
        for line in pgn:
            try:
                board = Board()
                # Giả sử dòng này chứa các nước đi
                moves = line.strip().split()
                for move in moves:
                    if move in board.legal_moves:  # Kiểm tra xem nước đi có hợp lệ không
                        board.push_san(move)
                games_data.append(board)
            except Exception as e:
                print(f"Đã xảy ra lỗi với ván cờ: {e}")
                continue  # Tiếp tục với ván cờ khác nếu gặp lỗi
            if len(games_data) >= max_games:
                break
    return games_data

def create_tf_dataset(X_data, y_data, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    X_data, y_data = extract_data_from_pgn("D:\Chess\chess-ai\chess\data\games.csv", max_games=500)
    np.save("X_data.npy", X_data)
    np.save("y_data.npy", y_data)
    print(f"Saved dataset: {X_data.shape[0]} samples")
