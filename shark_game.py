import sys, getopt
import pygame
import random
import math
from keras_example import get_trained_model
import numpy as np


black = (0, 0, 0)
blue = (0, 0, 100)
red = (255, 0, 0)
white = (255, 255, 255)
display_height = 800
display_width = 600

class driver(object):
    def __init__(self,character):
        self.character = character

    def drive(self,target):
        pass

class dummyDriver(driver):
    def drive(self, target):
        if((self.character.x <= self.character.speed)
           or (self.character.y <= self.character.speed)
           or (self.character.x >= self.character.maxX - self.character.speed)
           or (self.character.y >= self.character.maxY - self.character.speed)):
            self.character.direction = random.uniform(-1,1)

class terminatorDriver(driver):
    def drive(self, target):
        diffx = target.x - self.character.x
        diffy = target.y - self.character.y
        angle = math.atan2(diffy,diffx)
        self.character.direction = angle / math.pi

class kerasDriver(driver):
    def __init__(self,character):
        super(kerasDriver, self).__init__(character)
        self.model = get_trained_model(10000000)

    def drive(self, target):
        diffx = (target.x - self.character.x) / display_width
        diffy = (target.y - self.character.y) / display_height

        x = np.array([[diffy, diffx]])
        angle = math.atan2(diffy,diffx) / math.pi

        self.character.direction = self.model.predict(x)[0][0]
        print ("%.5f %0.5f" % (self.character.direction,angle))


class Character(object):

    def __init__(self,width,height,color,maxX,maxY):
        self.x = random.randint(0, maxX)
        self.y = random.randint(0, maxY)
        self.maxX = maxX
        self.maxY = maxY
        self.width = width
        self.height = height
        self.color = color
        self.direction = random.uniform(-1,1)
        self.speed = 3;
        self.driver = dummyDriver

    def setEnnemy(self, ennemy):
        self.ennemy = ennemy

    def setDriver(self, driver):
        self.driver = driver

    def draw(self,display):
        pygame.draw.rect(display,
                         self.color,
                         (self.x-self.width/2 , self.y-self.height/2, self.width, self.height), 0)

    def move(self):
        self.driver.drive(self.ennemy)
        angle = self.direction * math.pi;
        self.x = self.x + (math.cos(angle) * self.speed)
        self.y = self.y + (math.sin(angle) * self.speed)
        self.appliedLimit()

    def appliedLimit(self):
        if(self.x <= 0):
            self.x = 0
        if(self.y <= 0):
            self.y = 0
        if(self.x >= self.maxX):
            self.x = self.maxX
        if(self.y >= self.maxY):
            self.y = self.maxY


def getCharacters():
    shark = Character(20,
                      20,
                      blue,
                      display_height, display_width)

    fish = Character(10,
                      10,
                      red,
                      display_height, display_width)
    fish.setEnnemy(shark);
    fish.setDriver(dummyDriver(fish))

    shark.setEnnemy(fish);
    shark.setDriver(kerasDriver(shark))

    return [shark,fish]


def main(argv):
    pygame.init()
    folder = './'
    try:
        opts, args = getopt.getopt(argv, "f:a:")
    except getopt.GetoptError:
        print('main.py -f <folder> -a <action>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in "-f":
            folder = arg

    display = pygame.display.set_mode((display_height,display_width))


    clock = pygame.time.Clock()

    characters = getCharacters()

    fini = False

    while not fini:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                fini = True

        pygame.display.update()
        clock.tick(60)
        display.fill(white)
        for character in characters:
            character.move();
            character.draw(display)

    pygame.quit()
    quit()


if __name__ == "__main__":
    main(sys.argv[1:])
