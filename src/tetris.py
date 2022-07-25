import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random

style.use("ggplot")


WIDTH = 10   ## ancho en bloques del tetris ##
HEIGHT = 20  ## anlto en bloques del tetris ##
BLOCK_SIZE = 30  ## tamaño del bloque ##
TEXT_SIZE = BLOCK_SIZE / 2
TEXT_COLOR = (200, 20, 220)

GAMEOVER_REWARD = 2

#todo switcher
shape_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)]

#todo que se vean las rotaciones posibles
shapes = [
    [[1, 1],
     [1, 1]],

    [[2, 2, 2, 2]],

    [[0, 3, 3],
     [3, 3, 0]],

    [[4, 4, 0],
     [0, 4, 4]],

    [[5, 5, 5],
     [0, 5, 0]],

    [[6, 0],
     [6, 0],
     [6, 6]],

    [[0, 7],
     [0, 7],
     [7, 7]]
]


class Tetris:
    def __init__(self):
        self.extra_board = np.ones((HEIGHT * BLOCK_SIZE, WIDTH * int(BLOCK_SIZE / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        self.reset()

    '''
    #todo
    '''
    def reset(self):
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.score = 0
        self.shapes_set = 0
        self.lines_destroyed = 0
        self.bag = list(range(len(shapes)))
        random.shuffle(self.bag)
        self.shape_id = self.bag.pop()
        self.shape = [row[:] for row in shapes[self.shape_id]]
        self.current_pos = {"x": WIDTH // 2 - len(self.shape[0]) // 2, "y": 0}
        self.gameover = False
        return self.get_state_properties(self.board)

    '''
    método que da vuelta a las piezas
    recibe: una pieza
    retorna: la pieza rotada
    '''
    def rotate(self, piece):
        return [list(row) for row in zip(*reversed(piece))]


    '''
    #todo
    '''
    def get_state_properties(self, board):
        deleted_lines, board = self.check_full_rows(board)
        holes = self.get_holes(board)
        bumpiness, sum_heights = self.get_bumpiness_height(board)
        return torch.FloatTensor([deleted_lines, holes, bumpiness, sum_heights])

    '''
    método que devuelve todos los huecos encontrados en el tablero
    recibe: el tablero
    retorna: total de huecos encontrados
    '''
    def get_holes(self, board):
        total_holes = 0
        for col in zip(*board):
            row = 0
            while col[row] and row < HEIGHT == 0:
                row += 1
            listed_holes = [x for x in col[row + 1:] if x == 0]
            total_holes += len(listed_holes)
        return total_holes

    '''
    mask: tablero con False donde hay 0 y True donde no
    dist_to_roof: lista de distancias del box lleno más alto al techo del tablero
    filled_heights: lista de alturas de cuadros llenos por columna
    sum_heights: suma de todas las alturas
    diffs_next_col: diferencia de altura entre una columna y la siguiente]
    sum_bumpiness: suma de las diferencias de alturas de una columna y la siguiente
    '''
    def get_bumpiness_height(self, board):
        board = np.array(board)
        mask = 0 != board
        dist_to_roof = np.where(mask.any(axis=0), np.argmax(mask, axis=0), HEIGHT)
        filled_heights = HEIGHT - dist_to_roof
        sum_heights = np.sum(filled_heights)
        diffs_next_col = np.abs(filled_heights[:-1] - filled_heights[1:])
        sum_bumpiness = np.sum(diffs_next_col )
        return sum_bumpiness, sum_heights

    '''
    #todo
    '''
    def get_next_states(self):
        states = {}
        piece_id = self.shape_id
        curr_piece = [row[:] for row in self.shape]
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = WIDTH - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.shape)):
            for x in range(len(self.shape[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.shape[y][x]
        return board

    '''
    método que coloca una pieza en el tablero
    '''
    def set_new_piece(self):
        if not len(self.bag):
            self.bag = list(range(len(shapes)))
            random.shuffle(self.bag)
        self.shape_id = self.bag.pop()
        self.shape = [row[:] for row in shapes[self.shape_id]]
        self.current_pos = {"x": WIDTH // 2 - len(self.shape[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.shape, self.current_pos):
            self.gameover = True

    '''
    método que avisa si una pieza colisiona con techo o no
    recibe: una pieza y su posición
    retorna: True si hay colisión o False si no
    ejemplo de valor de pos {'x': 5, 'y':0}
    pieza cae en 'y' y se mueve izq o der en 'x'
    '''
    def check_collision(self, shape, pos):
        colision = False
        future_row = pos["y"]+1
        for y in range(len(shape)):
            for x in range(len(shape[y])):
                if future_row + y > HEIGHT-1 or self.board[y+future_row][pos["x"]+x] and shape[y][x]:
                    colision = True
        return colision

    '''
    #todo
    '''
    def truncate(self, shape, pos):
        gameover = False
        last_collision_row = -1
        for y in range(len(shape)):
            for x in range(len(shape[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and shape[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(shape) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(shape) > 1:
                gameover = True
                last_collision_row = -1
                del shape[0]
                for y in range(len(shape)):
                    for x in range(len(shape[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and shape[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    '''
    #todo
    '''
    def store(self, shape, pos):
        board = [x[:] for x in self.board]
        for y in range(len(shape)):
            for x in range(len(shape[y])):
                if shape[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = shape[y][x]
        return board

    '''
    método que chequea si hay filas llenas y de ser así llama a remove_row
    con la lista de índices de filas a eliminar
    recibe: el tablero
    retorna: cantidad de filas eliminadas, el tablero
    '''
    def check_full_rows(self, board):
        to_delete = []
        for i, row in enumerate(board):
            if 0 not in row:
                to_delete.append(i)
        if to_delete != []:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    '''
    método que elimina filas del tablero
    recibe: el tablero y los índices de filas a eliminar
    retorna: el tablero con las filas eliminadas
    '''
    def remove_row(self, board, indices):
        for i in indices:
            board.pop(i)
            board.insert(0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return board

    '''
    método que ejecuta una accion, cambia estado actual del tablero y recibe recompensa
    recibe: accion (posicion eje x, numero de rotaciones)
    retorna: recompensa y bool para saber si el juego ya termino
    '''
    def perform_action(self, action):
        #La accion representa la posicion en x y el num de rotaciones
        x_pos, num_rotations = action
        self.current_pos = {"x": x_pos, "y": 0}
        for _ in range(num_rotations):
            self.shape = self.rotate(self.shape)
        #Bajo la figura (eje y) hasta que colisione con el suelo
        while not self.check_collision(self.shape, self.current_pos):
            self.current_pos["y"] += 1
            #Visualizo la nueva posicion de la figura
            self.render()
        #Revisa que no se haya desbordado del tablero
        roof_overflow = self.truncate(self.shape, self.current_pos)
        if roof_overflow:
            self.gameover = True
        # Guarda la figura en el tablero   
        self.board = self.store(self.shape, self.current_pos)
         # Obtiene las lineas que fueron eliminidas
        lines_destroyed, self.board = self.check_full_rows(self.board)
        # Obtiene el score (reward): 1 + (lineas destruidas)^2 * width
        score = 1 + (lines_destroyed**2)*WIDTH
        self.score += score
        self.shapes_set += 1
        self.lines_destroyed += lines_destroyed
        if not self.gameover:
            self.set_new_piece()
        if self.gameover:
            # Si termina el juego el castigo es de -2
            self.score -= GAMEOVER_REWARD
        #Retorna al score y si el juego esta acabado
        return score, self.gameover

    '''
    #todo
    '''
    def render(self):
        if not self.gameover:
            img = [shape_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            img = [shape_colors[p] for row in self.board for p in row]
        img = np.array(img).reshape((HEIGHT, WIDTH, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize((WIDTH * BLOCK_SIZE, HEIGHT * BLOCK_SIZE))
        img = np.array(img)
        img[[i * BLOCK_SIZE for i in range(HEIGHT)], :, :] = 0
        img[:, [i * BLOCK_SIZE for i in range(WIDTH)], :] = 0

        img = np.concatenate((img, self.extra_board), axis=1)


        cv2.putText(img, "Score:", (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)
        cv2.putText(img, str(self.score),
                    (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), 2 * BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)

        cv2.putText(img, "Pieces:", (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), 4 * BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)
        cv2.putText(img, str(self.shapes_set),
                    (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), 5 * BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)

        cv2.putText(img, "Lines:", (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), 7 * BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)
        cv2.putText(img, str(self.lines_destroyed),
                    (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), 8 * BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)

        

        cv2.imshow("Deep Q-Learning Tetris", img)
        cv2.waitKey(1)
