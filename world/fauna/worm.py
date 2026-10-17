import math
import random
from math import atan2
import heapq
import numpy as np
import pygame
import torch
from world.fauna.brains import green
from torch import optim


class Worm:
    def __init__(
            self,
            pos: list[int] = None,
            x: int = None,
            y: int = None,
            length: int = 0,
            color: list[int] = None,
            follow: bool = False,
            speed: int = 5,
            id_: int = random.randint(1000, 9999)
    ):
        if color is None:
            color = [0, 255, 0]

        if pos is None and (x is None and y is None):
            raise Exception("Pos or x and y must be set!")

        elif pos is not None:
            self.pos = pos

        elif x is not None and y is not None:
            self.pos = [x, y]

        self.pos_log = []
        self.speed = speed
        self.spin_speed = 0.02
        self.follow = follow
        self.length = length
        self.default_length = length
        self.color = color
        self.fov = 45
        self.dov = 300
        self.brain = green.Green()
        self.angle = 0

        self.seq_len = 1
        self.hp = 100
        self.hunger = 100
        self.ray_stack = [0]*20
        self.noise = 0.2

        self.log_probs = []
        self.log_rewards = []
        self.reward = 0
        self.step_reward = 0
        self.fov = math.radians(self.fov)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)
        self.gamma = 0.99

        self.id = id_
        self.font = pygame.font.Font(None, 16)
        self.text_surf = self.font.render(f"Worm {self.id}", True, (255, 0, 0))

    def get_pos(self):
        fov = [
            [0, 0],
            [0, 0],
            [0, 0],
        ]

        if len(self.pos_log) > self.length:
            self.pos_log.pop(0)

        if self.follow:
            p = pygame.mouse.get_pos()
            normx = p[0] - self.pos[0]
            normy = p[1] - self.pos[1]
            d = math.sqrt(normx ** 2 + normy ** 2)
            x_n = self.pos[0] + self.speed * (normx / d)
            y_n = self.pos[1] + self.speed * (normy / d)
            angle = math.atan2(normy, normx)
            left = angle - (self.fov / 2)
            right = angle + (self.fov / 2)

            left_x = x_n + math.cos(left) * self.dov
            left_y = y_n + math.sin(left) * self.dov

            right_x = x_n + math.cos(right) * self.dov
            right_y = y_n + math.sin(right) * self.dov
            fov = [
                [x_n, y_n],
                [left_x, left_y],
                [right_x, right_y],
            ]
            self.pos = [x_n, y_n]
            self.pos_log.append(self.pos)

        else:
            rays = self.ray_stack
            inp = torch.tensor(self.ray_stack+[self.hp/100, self.hunger/100])
            mu = self.brain(inp)

            std = torch.tensor([self.noise, self.noise])
            dist = torch.distributions.Normal(mu, std)
            raw_action = dist.sample()
            action = torch.clamp(raw_action, -1.0, 1.0)
            result = action.detach().numpy()

            self.angle += result[0]*math.pi*self.spin_speed
            x_n, y_n = self.pos
            x_n += int(math.cos(self.angle)*self.speed*result[1])
            y_n += int(math.sin(self.angle)*self.speed*result[1])
            self.pos = [x_n, y_n]
            self.pos_log.append(self.pos)

            if self.hunger > 0:
                self.hunger -= 0.2
            else:
                self.hp -= 0.5

            self.step_reward = self.reward
            self.reward = 0
            if self.hunger < 60:
                self.step_reward -= 0.5
            elif self.hunger < 30:
                self.step_reward -= 2
            elif self.hunger > 80:
                self.step_reward += 0.02

            if self.length >= self.default_length:
                self.step_reward += 0.1*(self.length-self.default_length)+0.1

            rays_dist = rays[1::4]
            self.step_reward -= (100-self.hp)/10
            self.step_reward -= 0.01
            #self.step_reward += max(rays_dist)*3
            #self.step_reward += sum(rays_dist)



            self.log_rewards.append(self.step_reward)
            self.log_probs.append(dist.log_prob(raw_action).sum())

    def train(self):
        if len(self.log_rewards) == 0:
            return

        discounted_rewards = []
        running_add = 0.0
        for r in reversed(self.log_rewards):
            running_add = r + self.gamma * running_add
            discounted_rewards.insert(0, running_add)

        returns = torch.tensor(discounted_rewards, dtype=torch.float32)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.log_rewards.clear()

    def get_worm(self, screen, points, worms):
        if self.length <= 1:
            self.hp = 0
            return  [0, 0]
        for i in self.pos_log:
           pygame.draw.circle(screen, self.color, i, 5)
        stack_, p = self.get_look(points, worms, screen)
        self.get_pos()
        text_rect = self.text_surf.get_rect(center=(p[0][0], p[0][1]-20))
        screen.blit(self.text_surf, text_rect)
        pygame.draw.polygon(screen, self.color, p, 1)
        return p

    def get_look(self, points, worms, screen):
        a = list(map(int, self.pos))
        bx = int(self.pos[0] + math.cos(self.angle - (self.fov / 2)) * self.dov)
        by = int(self.pos[1] + math.sin(self.angle - (self.fov / 2)) * self.dov)
        cx = int(self.pos[0] + math.cos(self.angle + (self.fov / 2)) * self.dov)
        cy = int(self.pos[1] + math.sin(self.angle + (self.fov / 2)) * self.dov)
        p = [a, [bx, by], [cx, cy]]

        self.deleted = []
        tmp = []
        alive_worms = []
        for w in worms:
            for index_, position in enumerate(w.pos_log):
                alive_worms.append([position, w, index_])
        self.ray_stack = []
        for point in points:
            if self.point_collider(
                    p[0][0],
                    p[0][1],
                    p[1][0],
                    p[1][1],
                    p[2][0],
                    p[2][1],
                    point[0],
                    point[1]
            ):
                t = math.sqrt(((p[0][0] - point[0])/self.dov)**2+((p[0][1] - point[1])/self.dov)**2)
                if t < 0.1:
                    self.deleted.append(point)
                    self.reward += 10
                    self.hp += 10
                    self.hunger += 5
                    self.length += 1
                target_angle = math.atan2(point[1] - self.pos[1], point[0] - self.pos[0])
                diff_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
                norm_angle = diff_angle / (self.fov * 0.5)
                tmp.append([norm_angle, 1-t, 0, 1, [point[0], point[1]]])

        t_ = []
        for point, wrm, index_ in alive_worms:
            if wrm in t_:
                continue
            if self.point_collider(
                    p[0][0],
                    p[0][1],
                    p[1][0],
                    p[1][1],
                    p[2][0],
                    p[2][1],
                    point[0],
                    point[1]
            ):
                t = math.sqrt(((p[0][0] - point[0])/self.dov)**2+((p[0][1] - point[1])/self.dov)**2)
                if t < 0.05:
                    self.deleted.append(point)
                    self.reward += 15
                    self.hp += 12
                    self.hunger += 10
                    self.length += 3
                    wrm.cut(index_)
                    t_.append(wrm)
                target_angle = math.atan2(point[1] - self.pos[1], point[0] - self.pos[0])
                diff_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
                norm_angle = diff_angle / (self.fov * 0.5)
                tmp.append([norm_angle, 1-t, 1, 0, [point[0], point[1]]])
        tmp = heapq.nlargest(5, tmp, key=lambda x: x[1])
        for i in tmp:
            self.ray_stack.append(i[0])
            self.ray_stack.append(i[1])
            self.ray_stack.append(i[2])
            self.ray_stack.append(i[3])
            pygame.draw.line(screen, [255, 0, 0], self.pos, i[4])
        self.ray_stack += [0]*(20-len(self.ray_stack))
        return self.ray_stack, p

    def damage(self, d):
        self.hp -= d

    def cut(self, to):
        self.length -= to
        self.pos_log = self.pos_log[to::]
        self.hp -= to
        self.reward -= to

    def point_collider(self, ax, ay, bx, by, cx, cy, x, y):
        d1 = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        d2 = (cx - bx) * (y - by) - (cy - by) * (x - bx)
        d3 = (ax - cx) * (y - cy) - (ay - cy) * (x - cx)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def get_deleted(self):
        return  self.deleted

    def get_model(self):
        return self.brain.state_dict()

    def get_log(self):
        return {
            "alive": self.hp > 0,
            "hp": self.hp,
            "hunger": self.hunger,
            "pos": [self.pos[0], self.pos[1]],
            "angle_rad": round(self.angle, 3),
            "angle_deg": round(math.degrees(self.angle) % 360, 1),
            "body_length": len(self.pos_log),
            "step_reward": round(self.reward, 4),
            "accumulated_rewards_count": len(self.log_rewards),
            "total_pending_reward": round(sum(self.log_rewards), 4) if self.log_rewards else 0.0,
            "detected_points_count": len(self.ray_stack) // 3 if hasattr(self, "ray_stack") else 0,
            "eaten_last_step": len(self.deleted) if hasattr(self, "deleted") else 0,
            "follow_mouse": self.follow,
            "speed": self.speed,
        }

    def get_reward(self):
        steps = list(range(len(self.log_rewards)))
        return steps, self.log_rewards