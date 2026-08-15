from .board import EightPuzzleBoard
def main() -> None:
    print("Hello from 8puzzle!")

    board = EightPuzzleBoard.from_matrix([
        [3, 2, 8],
        [4, 1, 7],
        [5, 0, 6]
    ])

    print(board.__repr__())

    for n in board.neighbours():
        print(n)

if __name__ == "__main__":
    main()
