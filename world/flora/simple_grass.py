import random
import numpy as np
import pygame.surface


class Grass:
    def __init__(self, x: int, y: int, screen: pygame.surface.Surface, mtx=None):
        self.w = screen.get_width()
        self.h = screen.get_height()

        self.grass = np.zeros((self.w, self.h), dtype=np.uint8)
        self.total_cells = self.w * self.h

        # Параметры баланса кустов
        self.spawn_interval = 150  # Новый кустик спавнится раз в 45 кадров
        self.cluster_min = 30  # Мин. травинок в одном кусте
        self.cluster_max = 80  # Макс. травинок в одном кусте
        self.cluster_radius = 50  # Радиус разброса внутри куста (в пикселях)
        self.max_total_grass = 8000  # Жесткий общий лимит на всю карту

        # Спавним первый кустик вокруг стартовой точки (x, y)
        self._spawn_cluster(x, y)
        self.grass_count = int(np.sum(self.grass))

    def _spawn_cluster(self, center_x: int, center_y: int):
        """Создает компактный кустик из 5-10 травинок вокруг центральной точки."""
        count = random.randint(self.cluster_min, self.cluster_max)
        for _ in range(count):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(0, self.cluster_radius)

            gx = int(center_x + np.cos(angle) * dist)
            gy = int(center_y + np.sin(angle) * dist)

            if 0 <= gx < self.w and 0 <= gy < self.h:
                self.grass[gx, gy] = 1

    def update(self, s):
        # 1. Спавн нового случайного куста раз в self.spawn_interval кадров
        if isinstance(s, int) and s % self.spawn_interval == 0:
            if self.grass_count < self.max_total_grass:
                rand_x = random.randint(30, self.w - 30)
                rand_y = random.randint(30, self.h - 30)
                self._spawn_cluster(rand_x, rand_y)

        # 2. Медленное увядание: раз в 60 кадров удаляем небольшую порцию старой травы
        if isinstance(s, int) and s % 60 == 0:
            pts = np.argwhere(self.grass == 1)
            if len(pts) > 30:  # не трогаем, если травы совсем мало
                # Удаляем случайные 4% существующих точек
                die_mask = np.random.rand(len(pts)) < 0.04
                dying_pts = pts[die_mask]
                if len(dying_pts) > 0:
                    self.grass[dying_pts[:, 0], dying_pts[:, 1]] = 0

        self.grass_count = int(np.sum(self.grass))
        return self.grass

    def get_points(self):
        """Возвращает массив [[x, y], ...] для лучей зрения червя."""
        return np.argwhere(self.grass == 1)