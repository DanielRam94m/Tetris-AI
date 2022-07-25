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
TEXT_COLOR = (0, 0, 0)

GAMEOVER_REWARD = 2

shape_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)]

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
        self.background_board = np.ones((HEIGHT * BLOCK_SIZE, WIDTH * int(BLOCK_SIZE / 2), 3),
                                   dtype=np.uint8) * np.array([224, 224, 224], dtype=np.uint8)
        self.reset()

    '''
    método que reincia al ambiente: crea un nuevo tablero e inialiaza variables
    devuelve el estado inicial del tablero
    '''
    def reset(self):
        #Creamos el tablero lleno de ceros
        self.board = [[0] * WIDTH for _ in range(HEIGHT)] 
        self.score = 0 # para llevar el conteo del puntaje
        self.shapes_set = 0 #para llevar el conteo de figuras que envian  
        self.lines_destroyed = 0 # para llevar el conteo de las lineas destruidas
        self.list_shapes = list(range(len(shapes))) #lista con los IDs de las figuras
        random.shuffle(self.list_shapes) # desordena la lista para escoger una figura al azar
        self.shape_id = self.list_shapes.pop() #id figura escogida y lo quita de la lista para no volver a escogerla
        self.shape = [row[:] for row in shapes[self.shape_id]] #Lista de lista para guardar figura escogida
        self.current_pos = {"x": WIDTH // 2 - len(self.shape[0]) // 2, "y": 0} #posicion inicial de la figura en (x, y)
        self.gameover = False # para llevar el control de cuando se acabe la simulacion
        return self.get_state(self.board)

    '''
    método que da vuelta a las piezas
    recibe: una pieza
    retorna: la pieza rotada
    '''
    def rotate(self, piece):
        return [list(row) for row in zip(*reversed(piece))]

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
    método que devuelve el estado acutal del tablero
    retorna: un tensor con los siguientes valores 
    deleted_lines: cantidad de filas eliminadas en el tablero
    holes: cantidad de huecos encontrados en el tablero
    bumpiness: suma de las diferencias de alturas de una columna y la siguiente
    sum_heights: suma de todas las alturas
    '''
    def get_state(self, board):
        deleted_lines, board = self.check_full_rows(board)
        holes = self.get_holes(board)
        bumpiness, sum_heights = self.get_bumpiness_height(board)
        return torch.FloatTensor([deleted_lines, holes, bumpiness, sum_heights])

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
    método que obtiene todos los posibles estados del tablero para la figura actual
    retorna: lista de tensores con una uno de los posibles estados del tablero
    tomando en cuenta la rotacion de la figura actual y su posicion en el eje x 
    '''
    def get_next_states(self):
        states = {}
        shape_id = self.shape_id # Id figura actual
        actual_shape = [row[:] for row in self.shape] # Lista con figura actual
        '''Definimos el numero de rotaciones dependiendo de la figura'''
        if shape_id == 0:  # figura cuadrado
            num_rotations = 1
        elif shape_id == 2 or shape_id == 3 or shape_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4
        #Para cada rotacion
        for i in range(num_rotations):
            valid_xs = WIDTH - len(actual_shape[0]) # define el limite hasta donde se puede mover en el eje x
            for x in range(valid_xs + 1): # para cada posible punto en el eje x
                shape = [row[:] for row in actual_shape] # obtiene la figura despues de la rotacion 
                pos = {"x": x, "y": 0}  #guarda posicion
                while not self.is_collision(shape, pos): # mientra no haya tocada la base
                    pos["y"] += 1 #mueve la figura hacia abajo en el eje y
                self.truncate(shape, pos) #revisa desbordamiento
                board = self.store(shape, pos) #almacena la figura en el tablero
                states[(x, i)] = self.get_state(board) # obtiene el estado del tablero para cada posible figura generada 
            actual_shape = self.rotate(actual_shape) #Realiza una rotacion
        return states

    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.shape)):
            for x in range(len(self.shape[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.shape[y][x]
        return board

    '''
    método que obtiene una nueva figura
    '''
    def get_new_piece(self):
        if not len(self.list_shapes): #Si ya utilizamos todas las figuras
            self.list_shapes = list(range(len(shapes))) #genera de nuevo la lista de las opciones
            random.shuffle(self.list_shapes) #despordena la lista para escoger una al azar
        self.shape_id = self.list_shapes.pop() #saca la primera figura de la lista 
        self.shape = [row[:] for row in shapes[self.shape_id]] #Lista para guardar figura escogida
        self.current_pos = {"x": WIDTH // 2 - len(self.shape[0]) // 2,"y": 0} #posicion inicial de la figura en (x, y)
        if self.is_collision(self.shape, self.current_pos): #Revisa que la figura escogida no genera una colision 
            self.gameover = True

    '''
    método que avisa si una pieza colisiona con techo o no
    recibe: una pieza y su posición
    retorna: True si hay colisión o False si no
    ejemplo de valor de pos {'x': 5, 'y':0}
    pieza cae en 'y' y se mueve izq o der en 'x'
    '''
    def is_collision(self, shape, pos):
        colision = False
        future_row = pos["y"]+1
        for y in range(len(shape)):
            for x in range(len(shape[y])):
                if future_row + y > HEIGHT-1 or self.board[y+future_row][pos["x"]+x] and shape[y][x]:
                    colision = True
        return colision

    '''
    método que determina si la figura se desbordó del techo del tablero
    recibe figura y posicion en el tablero
    retorno bool true en caso que de el juego este terminado
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
    método que guarda una figura en el tablero
    recibe una figura y su posicion
    retorna el tablero resultante
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
            # si 0 no está en la línea, se agrega índice a to_delete
            if 0 not in row:
                to_delete.append(i)
        # si la lista a eliminar no está vacía, llama a remove_row con la lista a eliminar
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
        while not self.is_collision(self.shape, self.current_pos):
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
        #Si el el juego no se ha acabado, obtiene la siguiente figura
        if not self.gameover:
            self.get_new_piece()
        if self.gameover:
            self.score -= GAMEOVER_REWARD
        #Retorna al score y si el juego esta acabado
        return score, self.gameover

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

        img = np.concatenate((img, self.background_board), axis=1)


        cv2.putText(img, "Score:", (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)
        cv2.putText(img, str(self.score),
                    (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), 2 * BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)


        cv2.putText(img, "Lines:", (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), 7 * BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)
        cv2.putText(img, str(self.lines_destroyed),
                    (WIDTH * BLOCK_SIZE + int(TEXT_SIZE), 8 * BLOCK_SIZE),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=TEXT_COLOR)

        

        cv2.imshow("Deep Q-Learning Tetris", img)
        cv2.waitKey(1)
