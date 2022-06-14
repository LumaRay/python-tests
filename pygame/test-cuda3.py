import pygame
import numpy
import math
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
TEXTCOLOR = (0, 0, 0)
(width, height) = (800, 600)

running = True

background_color = TEXTCOLOR

step_divisor = 100

# CUDA kernel
@cuda.jit
def point_step(A, rng_states):
    thread_id = cuda.threadIdx.x
    row, col = cuda.grid(2)
    if row >= A.shape[0] or col >= A.shape[1]:
        return
    row_rand = xoroshiro128p_uniform_float32(rng_states, thread_id) * A.shape[0]
    col_rand = xoroshiro128p_uniform_float32(rng_states, thread_id) * A.shape[1]
    row_delta = max(1, (row_rand - row) / step_divisor)
    col_delta = max(1, (col_rand - col) / step_divisor)
    row_new = min(A.shape[0], int(row + row_delta))
    col_new = min(A.shape[1], int(col + col_delta))
    tmpR = A[row_new, col_new, 0]
    tmpG = A[row_new, col_new, 1]
    tmpB = A[row_new, col_new, 2]
    A[row_new, col_new, 0] = A[row, col, 1]
    A[row_new, col_new, 1] = A[row, col, 2]
    A[row_new, col_new, 2] = A[row, col, 3]
    A[row, col, 0] = tmpR
    A[row, col, 1] = tmpG
    A[row, col, 2] = tmpB

def main():
    # data = (numpy.random.rand(height, width, 4) * 255).astype(numpy.uint8)
    data = (numpy.random.rand(height, width, 3) * 255).astype(numpy.uint8)
    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(data.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(data.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    data_global_mem = cuda.to_device(data)
    rng_states = create_xoroshiro128p_states(threadsperblock[0] * threadsperblock[1] * blockspergrid_x * blockspergrid_y, seed=1)

    global running, screen

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    # pygame.display.set_caption("123")
    # screen.fill(background_color)
    pygame.display.update()

    while running:
        ev = pygame.event.get()

        for event in ev:

            if event.type == pygame.MOUSEBUTTONUP:
                draw_circle()
                pygame.display.update()

            if event.type == pygame.QUIT:
                running = False
        pygame.surfarray.blit_array(screen, data.swapaxes(0, 1))

        data = step_time(blockspergrid, threadsperblock, data_global_mem, rng_states)

        pygame.display.update()

def draw_circle():
    pos = pygame.mouse.get_pos()
    pygame.draw.circle(screen, BLUE, pos, 20)

def step_time(blockspergrid, threadsperblock, data_global_mem, rng_states):
    point_step[blockspergrid, threadsperblock](data_global_mem, rng_states)
    res = data_global_mem.copy_to_host()
    return res

if __name__ == '__main__':
    main()
