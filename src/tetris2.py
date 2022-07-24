'''

'''
#from random import random
import random
import pygame

'''
Variables glogales
'''
game_width =  800
game_height = 600
block_size = 30
#width = 10
#height = 20
cols = 10
rows = 20


grid_width = block_size*cols
grid_height = block_size*rows
top_left_x = (game_width - (grid_width)) // 2
top_left_y = game_height - (grid_height)

'''
#TODO: Class Shapes
Formas posibles
'''
O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]
 
Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]
 
T = [['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '..0..',
      '.000.',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....']]

L = [['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.0...',
      '.000.',
      '.....',
      '.....']]

J = [['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....'],
     ['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....']]

shapes = [O, I, S, Z, T, L, J]

'''
#TODO: cambiar números (R,G,B)
'''

'''
recibe:   ídice del shape
devuelve: color en RGB
'''
def get_color(shape_index):
    switcher = {
        0: (0, 255, 0),
        1: (255, 0, 0),
        2: (0, 255, 255),
        3: (255, 255, 0),
        4: (255, 165, 0),
        5: (0, 0, 255),
        6: (128, 0, 128),
    }
    return switcher.get(shape_index, "nothing")

class Piece(object):  #TODO: cambiar nombre
  '''
  cols = width = 10
  rows = height = 20
  '''
  def __init__(self, cols, rows, shape):
    self.cols = cols #TODO: cambiar nombre
    self.rows = rows #TODO: cambiar nombre
    self.shape = shape
    shape_index = shapes.index(self.shape)
    self.color = get_color(shape_index)
    self.rotation = 0


'''
Crea el tablero
'''
def create_grig(locked_pos = {}): #TODO: cambiar nombre
  'se inicializa el grid vacío, usamos _ porque no usaremos el iterador'
  grid = [[(0,0,0) for _ in range(cols)] for _ in range(rows)]
  'se chequea por bloqueados'
  for row in range(grid):
    for col in range(len(grid[row])):
      if (col, row) in locked_pos:
        c = locked_pos[(col, row)]
        grid[row][col] = c
  return grid


def convert_shape_format(): #TODO: cambiar nombre
  pass


def valid_space(): #TODO: cambiar nombre
  pass


def check_lost(): #TODO: cambiar nombre
  pass

def get_shape(): #TODO: cambiar nombre
  return random.choice(shapes)

def draw_text_middle():
  pass

def draw_grid(surface):
  surface.fill((0,0,0)) #todo: (255,255,255)
  pygame.font.init()
  #todo cambiar comicsans font
  font = pygame.font.SysFont('comicsans', 60)
  label = font.render('Tetris', 1, (255,255, 255)) #todo: (0,0,0)
  'colocamos el label en el medio'
  surface.blit(label, (top_left_x + grid_width)/2 - (label.get_width()/2), 30)

def clear_rows():
  pass

def draw_next_shape():
  pass

def draw_window():
  pass

def main():
  pass