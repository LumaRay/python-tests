import os
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import pygame
import math
import numpy as np
from numba import cuda

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
TEXTCOLOR = (0, 0, 0)
(width, height) = (800, 600)

# MAX_PARTICLES = 1
# MAX_PARTICLES = 2
# MAX_PARTICLES = 100
MAX_PARTICLES = 1000
# MAX_PARTICLES = 10000
# MAX_PARTICLES = 100000

# INITIAL_PARTICLES = 1
# INITIAL_PARTICLES = 2
# INITIAL_PARTICLES = 10
# INITIAL_PARTICLES = 100
INITIAL_PARTICLES = 1000
# INITIAL_PARTICLES = 10000
# INITIAL_PARTICLES = 100000

G = 6.67
k1 = .0000001
k2 = 1

MIN_DISTANCE = 0.1

calc_steps = 1
# calc_steps = 10
# calc_steps = 100
# calc_steps = 1000

running = True

# CUDA kernel
@cuda.jit
def particles_step1(mgm_particles_count, gm_m, gm_y, gm_x, gm_fy, gm_fx):
    particle_idx, ref_particle_idx = cuda.grid(2)
    if particle_idx == ref_particle_idx:
        return
    if particle_idx >= gm_y.shape[0] or particle_idx >= mgm_particles_count[0]:
        return
    if ref_particle_idx >= gm_y.shape[0] or ref_particle_idx >= mgm_particles_count[0]:
        return
    m = gm_m[particle_idx]
    y, x = gm_y[particle_idx], gm_x[particle_idx]
    rm = gm_m[ref_particle_idx]
    ry, rx = gm_y[ref_particle_idx], gm_x[ref_particle_idx]
    rdy, rdx = ry - y, rx - x
    l = (rdy**2 + rdx**2) ** 0.5
    if l < MIN_DISTANCE:
        l = MIN_DISTANCE
    f = k1 * G * m * rm / (l**2)
    gm_fy[particle_idx] += f * rdy / l
    gm_fx[particle_idx] += f * rdx / l

# CUDA kernel
@cuda.jit
def particles_step2(mgm_particles_count, gm_m, gm_y, gm_x, gm_vy, gm_vx, gm_fy, gm_fx):
    particle_idx = cuda.grid(1)
    if particle_idx >= gm_y.shape[0] or particle_idx >= mgm_particles_count[0]:
        return
    m = gm_m[particle_idx]
    fy_sum, fx_sum = gm_fy[particle_idx], gm_fx[particle_idx]
    vy, vx = gm_vy[particle_idx], gm_vx[particle_idx]
    y, x = gm_y[particle_idx], gm_x[particle_idx]
    ay = fy_sum / m if fy_sum != 0 and m != 0 else 0
    ax = fx_sum / m if fx_sum != 0 and m != 0 else 0
    nvy, nvx = vy + ay, vx + ax
    dy, dx = k2 * nvy, k2 * nvx
    ny, nx = y + dy, x + dx
    if ny < 0 or ny > 1:
        dy = -dy
        nvy = -nvy
        ny = y + dy
    if nx < 0 or nx > 1:
        dx = -dx
        nvx = -nvx
        nx = x + dx
    gm_y[particle_idx], gm_x[particle_idx] = ny, nx
    gm_vy[particle_idx], gm_vx[particle_idx] = nvy, nvx
    gm_fy[particle_idx], gm_fx[particle_idx] = 0, 0

def main():
    global mgm_particles_count, gm_m, gm_y, gm_x, gm_vy, gm_vx, gm_fy, gm_fx
    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(MAX_PARTICLES / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(MAX_PARTICLES / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    mgm_particles_count = cuda.mapped_array((1), dtype=np.uint64)
    mgm_particles_count[0] = INITIAL_PARTICLES
    # gm_m = cuda.device_array(MAX_PARTICLES, dtype=np.float32)
    # gm_y = cuda.device_array(MAX_PARTICLES, dtype=np.float32)
    # gm_x = cuda.device_array(MAX_PARTICLES, dtype=np.float32)
    # gm_vy = cuda.device_array(MAX_PARTICLES, dtype=np.float32)
    # gm_vx = cuda.device_array(MAX_PARTICLES, dtype=np.float32)
    arr_m = np.zeros(MAX_PARTICLES, dtype=np.float32)
    arr_y = np.zeros(MAX_PARTICLES, dtype=np.float32)
    arr_x = np.zeros(MAX_PARTICLES, dtype=np.float32)
    arr_vy = np.zeros(MAX_PARTICLES, dtype=np.float32)
    arr_vx = np.zeros(MAX_PARTICLES, dtype=np.float32)
    arr_fy = np.zeros(MAX_PARTICLES, dtype=np.float32)
    arr_fx = np.zeros(MAX_PARTICLES, dtype=np.float32)
    for particle_idx in range(INITIAL_PARTICLES):
        arr_m[particle_idx] = np.random.random()
        arr_y[particle_idx] = np.random.random()
        arr_x[particle_idx] = np.random.random()
        arr_vy[particle_idx] = (np.random.random() * 2 - 1) * .001
        arr_vx[particle_idx] = (np.random.random() * 2 - 1) * .001
    gm_m = cuda.to_device(arr_m)
    gm_y = cuda.to_device(arr_y)
    gm_x = cuda.to_device(arr_x)
    gm_vy = cuda.to_device(arr_vy)
    gm_vx = cuda.to_device(arr_vx)
    gm_fy = cuda.to_device(arr_fy)
    gm_fx = cuda.to_device(arr_fx)

    global running, screen

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.update()

    while running:
        screen.fill((0, 0, 0))
        ev = pygame.event.get()
        for event in ev:

            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                draw_circle(pos)
                pygame.display.update()

            if event.type == pygame.QUIT:
                running = False

        data = step_time(blockspergrid, threadsperblock)

        # pygame.display.update()
        pygame.display.flip()

def draw_circle(pos):
    pygame.draw.circle(screen, BLUE, pos, 20)

def step_time(blockspergrid, threadsperblock):
    global mgm_particles_count, gm_m, gm_y, gm_x, gm_vy, gm_vx, gm_fy, gm_fx
    for _ in range(calc_steps):
        particles_step1[blockspergrid, threadsperblock](
            mgm_particles_count, gm_m, gm_y, gm_x, gm_fy, gm_fx)
        particles_step2[blockspergrid[0], threadsperblock[0]](
            mgm_particles_count, gm_m, gm_y, gm_x, gm_vy, gm_vx, gm_fy, gm_fx)
    res = None  # kernel_args[2].copy_to_host(), kernel_args[3].copy_to_host()

    arr_m = gm_m.copy_to_host()
    arr_x = gm_x.copy_to_host()
    arr_y = gm_y.copy_to_host()
    for particle_idx in range(mgm_particles_count[0]):
        pygame.draw.circle(screen,
                           (arr_m[particle_idx] * 255, arr_m[particle_idx] * 105 + 50, 0),
                           (arr_x[particle_idx] * width, arr_y[particle_idx] * height),
                            arr_m[particle_idx] * 10)
    return res

if __name__ == '__main__':
    mgm_particles_count = None
    gm_m = None
    gm_y = None
    gm_x = None
    gm_vy = None
    gm_vx = None
    gm_fy = None
    gm_fx = None

    main()

