import os
import sys
import random
import pygame
import numpy as np
from sympy.physics.units import speed
import zipfile
import torch
from sympy.printing.pytorch import torch
import matplotlib.pyplot as plt
from world.flora.simple_grass import Grass
from world.fauna.worm import Worm
import time
import logging


GLOBAL_W = 1920
GLOBAL_H = 1080

GLOBAL_W_HALF = int(GLOBAL_W / 2)
GLOBAL_H_HALF = int(GLOBAL_H / 2)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    filename="world.log",
    # encoding="utf-8",
)
logger = logging.getLogger(__name__)

logger.debug("Starting")
pygame.init()
screen = pygame.display.set_mode((GLOBAL_W, GLOBAL_H))
screen_array = np.zeros((GLOBAL_W, GLOBAL_H, 3), dtype=np.uint8)
pygame.display.set_caption("simworld")
clock = pygame.time.Clock()

def point_collider(ax, ay, bx, by, cx, cy, x, y):
    cp1 = (ax-bx)*(y-ay)-(by-ay)*(x-ax)
    cp2 = (ax-cx)*(y-ay)-(cy-ay)*(x-ax)

    if cp1 >= 0 and cp2 >= 0:
        return True
    else:
        return False

logger.debug("Making grass")
grass = Grass(random.randint(200, 400), random.randint(200, 400), screen, screen_array)

test_worm = Worm([400, 300], follow=False, length=100)
worms = [
#test_worm,
#Worm([500, 200], follow=False, length=100, color=[255, 0, 0]),
#Worm([300, 400], follow=False, length=100, color=[0, 0, 255]),
#Worm([200, 500], follow=False, length=100, color=[255, 0, 255]),
Worm(
        [
            GLOBAL_W_HALF + random.randint(-GLOBAL_W_HALF, GLOBAL_W_HALF),
            GLOBAL_H_HALF + random.randint(-GLOBAL_H_HALF, GLOBAL_H_HALF)
        ],
        follow=False, length=100,
        color=[
            50 + random.randint(-50, 50),
            155 + random.randint(-50, 50),
            50 + random.randint(-50, 50)],
        id_=i
    )

for i in range(15)
]
logger.debug("Worms was created")

logger.debug("Configuration archive...")
archive = f"./models/model_{str(int(time.time()))}.zip"
with zipfile.ZipFile(archive, mode="a") as zf:
    zf.writestr("models/", "")
    for i in range(1, len(worms) + 1):
        zf.writestr(f"models/{i}/", "")
logger.debug("Ready to start...")


alive_worms = list(worms)
#alive_worms.append(test_worm)
steps = 0
epoch = 0

running = True
while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #screen_array[::3, ::3, 0] = 255
    #screen_array[::6, ::6, 1] = 255
    #screen_array[::9, ::9, 2] = 255

    surface = pygame.surfarray.make_surface(screen_array)
    screen.blit(surface, (0, 0))

    rgb_arr = np.zeros((GLOBAL_W, GLOBAL_H, 3), dtype=np.uint8)
    a = grass.update(steps)
    grass_points = grass.get_points()
    fovs = []


    rgb_arr[a == 1] = [0, 255, 0]
    surface = pygame.surfarray.make_surface(rgb_arr)
    screen.blit(surface, (0, 0))

    for w in alive_worms:
        aw = list(alive_worms)
        aw.remove(w)
        w.get_worm(screen, grass_points, aw)
        for pnt in w.get_deleted():
            try:
                grass.grass[pnt[0], pnt[1]] = 0
            except:
                pass
        if w.hp <= 0:
            alive_worms.remove(w)

    if len(alive_worms) == 0 or steps == 4500: #4500
        print(len(alive_worms) == 0, steps == 4500)
        logger.debug(f"Epoch {epoch} finished")
        for index, w in enumerate(worms):
            if len(alive_worms) == 0:
                w.reward -= 10

            logger.debug(f"Executing worm {index}")
            with zipfile.ZipFile(archive, mode="a") as zf:
                log = w.get_log()
                x, y = w.get_reward()

                plt.plot(x, y)
                plt.xlabel("step")
                plt.ylabel("reward")
                plt.savefig("./cache/rewards.png", dpi=300, bbox_inches="tight")
                plt.close()
                with open("./cache/log.txt", "w") as f:
                    f.write(str(log).replace(", ", ", \n"))

                torch.save(w.get_model(), "./cache/model.pth")

                zf.write(f"./cache/rewards.png", arcname=f"models/{index+1}/rewards/rewards-{epoch}.png")
                zf.write(f"./cache/log.txt", arcname=f"models/{index+1}/log/log-{epoch}.txt")
                zf.write(f"./cache/model.pth", arcname=f"models/{index+1}/model/model-{epoch}.pth")

            logger.debug("Training...")
            w.train()
            logger.debug("Reset")
            w.hunger = 100
            w.length = 100
            w.hp = 100
            w.pos = [
                GLOBAL_W_HALF + random.randint(-GLOBAL_W_HALF, GLOBAL_W_HALF),
                GLOBAL_H_HALF + random.randint(-GLOBAL_H_HALF, GLOBAL_H_HALF)
            ]
            w.pos_log = []
            w.log_rewards = []
            w.log_probs = []
            w.reward = 0
        epoch += 1
        grass = Grass(random.randint(200, 400), random.randint(200, 400), screen, screen_array)
        alive_worms = list(worms)
        steps = 0
    steps += 1
    clock.tick(60)
    pygame.display.flip()
    if (steps % 600) == 0:
        worms_stat = "\n".join(
            f"W{i + 1}({w.hp:.0f}hp,{w.hunger:.0f}hg,{w.reward:+.0f}r)"
            for i, w in enumerate(worms)
        )
        print(
            f"[{steps // 60}0s|Ep {epoch}] FPS: {clock.get_fps():.0f} | Grass: {len(grass_points)} | Alive: {len(alive_worms)}/{len(worms)}"
        )
        print(f"  -> {worms_stat}")
        logger.info(f"[{steps // 60}0s|Ep {epoch}] FPS: {clock.get_fps():.0f} | Grass: {len(grass_points)} | Alive: {len(alive_worms)}/{len(worms)}")
        logger.info(f"  -> {worms_stat}")

pygame.quit()
sys.exit()