import pygame
import numpy
import math
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import cv2

img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
img_data = cv2.imread(img_path)

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
TEXTCOLOR = (0, 0, 0)
(width, height) = (800, 600)

running = True

background_color = TEXTCOLOR

# step_divisor = 100
step_divisor = 1000

# max_difference = 12
max_difference = 120

calc_steps = 1
# calc_steps = 10
# calc_steps = 100
# calc_steps = 1000

# CUDA kernel
@cuda.jit
def point_step(A, busy, rng_states):
    thread_id = cuda.threadIdx.x
    row, col = cuda.grid(2)
    if row >= A.shape[0] or col >= A.shape[1]:
        return
    if busy[row, col]:
        return
    busy[row, col] = True
    row_compared = int(min(A.shape[0] - 1, xoroshiro128p_uniform_float32(rng_states, thread_id) * A.shape[0]))
    col_compared = int(min(A.shape[1] - 1, xoroshiro128p_uniform_float32(rng_states, thread_id) * A.shape[1]))
    if abs(A[row_compared, col_compared, 0] - A[row, col, 0]) > max_difference \
            or abs(A[row_compared, col_compared, 1] - A[row, col, 1]) > max_difference \
            or abs(A[row_compared, col_compared, 2] - A[row, col, 2]) > max_difference:
        busy[row, col] = False
        return
    row_delta, col_delta = 0, 0
    if row_compared < row:
        row_delta = -1
    if row_compared > row:
        row_delta = 1
    if col_compared < col:
        col_delta = -1
    if col_compared > col:
        col_delta = 1
    if col_delta == 0 and row_delta == 0:
        busy[row, col] = False
        return
    # row_delta = (row_compared - row) / step_divisor
    # col_delta = (col_compared - col) / step_divisor
    # if 0 < row_delta < col_delta < 1:
    #     row_delta, col_delta = 0, 1
    # elif 0 < col_delta < row_delta < 1:
    #     col_delta, row_delta = 0, 1
    # elif -1 < row_delta < col_delta < 0:
    #     row_delta, col_delta = -1, 0
    # elif -1 < col_delta < row_delta < 0:
    #     col_delta, row_delta = -1, 0
    # if 0 < row_delta < 1:
    #     row_delta = 1
    # if -1 < row_delta < 0:
    #     row_delta = -1
    # if 0 < col_delta < 1:
    #     col_delta = 1
    # if -1 < col_delta < 0:
    #     col_delta = -1
    row_new = max(0, min(A.shape[0] - 1, int(round(row + row_delta))))
    col_new = max(0, min(A.shape[1] - 1, int(round(col + col_delta))))
    if busy[row_new, col_new]:
        busy[row, col] = False
        return
    busy[row_new, col_new] = True

    tmpR = A[row_new, col_new, 0]
    A[row_new, col_new, 0] = A[row, col, 0]
    A[row, col, 0] = tmpR
    tmpG = A[row_new, col_new, 1]
    A[row_new, col_new, 1] = A[row, col, 1]
    A[row, col, 1] = tmpG
    tmpB = A[row_new, col_new, 2]
    A[row_new, col_new, 2] = A[row, col, 2]
    A[row, col, 2] = tmpB

    busy[row_new, col_new] = False
    busy[row, col] = False

def main():
    # data = (numpy.random.rand(height, width, 4) * 255).astype(numpy.uint8)
    # data = (numpy.random.rand(height, width, 3) * 255).astype(numpy.uint8)
    data = img_data
    data = cv2.resize(data, (width, height))
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    # threadsperblock = (16, 16)
    # threadsperblock = (32, 32)
    threadsperblock = (15, 20)
    blockspergrid_x = int(math.ceil(data.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(data.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    data_global_mem = cuda.to_device(data)
    rng_states = create_xoroshiro128p_states(threadsperblock[0] * threadsperblock[1] * blockspergrid_x * blockspergrid_y, seed=1)
    busy_global_mem = cuda.device_array((height, width), dtype=bool)

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

        data = step_time(blockspergrid, threadsperblock, data_global_mem, busy_global_mem, rng_states)

        pygame.surfarray.blit_array(screen, data.swapaxes(0, 1))

        pygame.display.update()

def draw_circle():
    pos = pygame.mouse.get_pos()
    pygame.draw.circle(screen, BLUE, pos, 20)

def step_time(blockspergrid, threadsperblock, data_global_mem, busy_global_mem, rng_states):
    for _ in range(calc_steps):
        point_step[blockspergrid, threadsperblock](data_global_mem, busy_global_mem, rng_states)
    res = data_global_mem.copy_to_host()
    return res

if __name__ == '__main__':
    main()
