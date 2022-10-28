import pygame
import numpy as np

from nn import NeuralNetwork
from config import CONFIG

NEAR_BOX = 3


class Player():

    def __init__(self, mode, control=False):

        # if True, playing mode is activated. else, AI mode.
        self.control = control
        self.pos = [100, 275]   # position of the agent
        self.direction = -1     # if 1, goes upwards. else, goes downwards.
        self.v = 0              # vertical velocity
        self.g = 9.8            # gravity constant
        self.mode = mode        # game mode

        # neural network architecture (AI mode)
        layer_sizes = self.init_network(mode)

        self.nn = NeuralNetwork(layer_sizes)
        self.fitness = 0  # fitness of agent

    def move(self, box_lists, camera, events=None):

        if len(box_lists) != 0:
            if box_lists[0].x - camera + 60 < self.pos[0]:
                box_lists.pop(0)

        mode = self.mode

        # manual control
        if self.control:
            self.get_keyboard_input(mode, events)

        # AI control
        else:
            agent_position = [camera + self.pos[0], self.pos[1]]
            self.direction = self.think(
                mode, box_lists, agent_position, self.v)

        # game physics
        if mode == 'gravity' or mode == 'helicopter':
            self.v -= self.g * self.direction * (1 / 60)
            self.pos[1] += self.v

        elif mode == 'thrust':
            self.v -= 6 * self.direction
            self.pos[1] += self.v * (1 / 40)

        # collision detection
        is_collided = self.collision_detection(mode, box_lists, camera)

        return is_collided

    # reset agent parameters
    def reset_values(self):
        self.pos = [100, 275]
        self.direction = -1
        self.v = 0

    def get_keyboard_input(self, mode, events=None):

        if events is None:
            events = pygame.event.get()

        if mode == 'helicopter':
            self.direction = -1
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.direction = 1

        elif mode == 'thrust':
            self.direction = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.direction = 1
            elif keys[pygame.K_DOWN]:
                self.direction = -1

        for event in events:
            if event.type == pygame.KEYDOWN:

                if mode == 'gravity' and event.key == pygame.K_SPACE:
                    self.direction *= -1

    def init_network(self, mode):

        # you can change the parameters below

        layer_sizes = None
        if mode == 'gravity':
            layer_sizes = [2 + 3 * NEAR_BOX, 64, 2]
        elif mode == 'helicopter':
            layer_sizes = [2 + 3 * NEAR_BOX, 64, 2]
        elif mode == 'thrust':
            layer_sizes = [2 + 3 * NEAR_BOX, 64, 3]
        return layer_sizes

    def think(self, mode, box_lists, agent_position, velocity):

        features = [agent_position[1] / CONFIG['HEIGHT'], velocity / 10]
        box_list_len = NEAR_BOX if len(
            box_lists) > NEAR_BOX else len(box_lists)

        for box in box_lists[:box_list_len]:
            features.extend([
                (box.x - agent_position[0]) / CONFIG['WIDTH'], (agent_position[1] - box.gap_mid) / CONFIG['HEIGHT'], self.get_distance(agent_position, box)])

        for _ in range(NEAR_BOX - box_list_len):
            distance = pow(pow((CONFIG['WIDTH'] - agent_position[0]) / CONFIG['WIDTH'], 2) + pow(
                ((CONFIG['HEIGHT'] / 2) - agent_position[1]) / CONFIG['HEIGHT'], 2), 1/2)
            sign = 1 if agent_position[1] < (CONFIG['HEIGHT'] / 2) else -1
            features.extend(
                [(CONFIG['WIDTH'] - agent_position[0]) / CONFIG['WIDTH'], (agent_position[1] - (CONFIG['HEIGHT'] / 2)) / CONFIG['HEIGHT'], distance * sign])

        input = np.array(features)
        output = self.nn.forward(input)

        if(mode == 'helicopter' or mode == 'gravity'):
            return 1 if np.argmax(output) == 0 else -1
        else:
            return np.argmax(output) - 1

    def collision_detection(self, mode, box_lists, camera):
        if mode == 'helicopter':
            rect = pygame.Rect(self.pos[0], self.pos[1], 100, 50)
        elif mode == 'gravity':
            rect = pygame.Rect(self.pos[0], self.pos[1], 70, 70)
        elif mode == 'thrust':
            rect = pygame.Rect(self.pos[0], self.pos[1], 110, 70)
        else:
            rect = pygame.Rect(self.pos[0], self.pos[1], 50, 50)
        is_collided = False

        if self.pos[1] < -60 or self.pos[1] > CONFIG['HEIGHT']:
            is_collided = True

        if len(box_lists) != 0:
            box_list = box_lists[0]
            for box in box_list.boxes:
                box_rect = pygame.Rect(box[0] - camera, box[1], 60, 60)
                if box_rect.colliderect(rect):
                    is_collided = True

        return is_collided

    def get_distance(self, position, box):
        distance = pow(pow((box.x - position[0]) / CONFIG['WIDTH'], 2) + pow(
            (box.gap_mid - position[1]) / CONFIG['HEIGHT'], 2), 1/2)
        sign = 1 if position[1] < box.gap_mid else -1
        return distance * sign