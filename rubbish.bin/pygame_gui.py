import time
import pygame
import taichi as ti
import numpy as np
from pygame.locals import *





def mainloop(res, title, img, render):
    dat = ti.Vector.field(3, ti.u8, res[::-1])


    @ti.kernel
    def export():
        for i, j in dat:
            dat[i, j] = min(255, max(0, int(img[j, res[1] - 1 - i] * 255)))


    pygame.init()

    screen = pygame.display.set_mode(res, DOUBLEBUF | HWSURFACE | FULLSCREEN)
    pygame.display.set_caption(title)
    fpsclk = pygame.time.Clock()
    last = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                exit()

        render()
        export()
        data = dat.to_numpy()
        data = pygame.image.frombuffer(data.tobytes('C'), res, 'RGB')

        screen.blit(data, (0, 0))
        pygame.display.flip()
        fpsclk.tick(60)
        t = time.time()
        dt = t - last
        print(f'({1 / dt:.2f} FPS)')
        #pygame.display.set_caption(f'{title} ({1 / dt:.2f} FPS)')
        last = t


if __name__ == '__main__':
    res = 1920, 1080
    img = ti.Vector.field(3, float, res)

    @ti.kernel
    def render():
        for i, j in img:
            img[i, j] = [i / res[0], j / res[1], 0]

    mainloop(res, 'THREE', img, render)
