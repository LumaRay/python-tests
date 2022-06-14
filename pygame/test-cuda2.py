import pygame
import numpy
import math
from numba import cuda

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
TEXTCOLOR = (0, 0, 0)
(width, height) = (800, 600)

running = True

background_color = TEXTCOLOR

# CUDA kernel
@cuda.jit
def matmul(A):
    body_idx = cuda.grid(1)
    if body_idx >= A.shape[0]:
        return
    x, y = 0., 0.
    for k in range(A.shape[0]):
        x += A[k][0]
        y += A[k][1]
    A[body_idx] = x, y

def main():
    data = numpy.asarray([[20, 30], [40, 50]])
    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(data.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(data.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    data_global_mem = cuda.to_device(data)

    global running, screen

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("123")
    screen.fill(background_color)
    pygame.display.update()

    while running:
        ev = pygame.event.get()

        for event in ev:

            if event.type == pygame.MOUSEBUTTONUP:
                draw_circle()
                pygame.display.update()

            if event.type == pygame.QUIT:
                running = False

        step_time(blockspergrid, threadsperblock, data_global_mem)
        pygame.display.update()

def draw_circle():
    pos = pygame.mouse.get_pos()
    pygame.draw.circle(screen, BLUE, pos, 20)

def step_time(blockspergrid, threadsperblock, data_global_mem):
    matmul[blockspergrid, threadsperblock](data_global_mem)
    data = data_global_mem.copy_to_host()
    pass

if __name__ == '__main__':
    main()
