import pygame
import cv2
import os

from ConvertToImage import ConvertToImage

pygame.init() 

WIDTH = 28
FPS = 30
clock = pygame.time.Clock()
convertation = ConvertToImage()

# CREATING CANVAS 
screen = pygame.display.set_mode((500, 500))
figure = []

# TITLE OF CANVAS 
pygame.display.set_caption("My Board") 
exit = False
f1 = pygame.font.Font(None, 36)

bg_color = (255, 255, 255)
while not exit:
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT: 
            exit = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                with open("coord.txt", "a") as file:
                    file.write(str(figure) + "\n")
                print("Сохранено")
            if event.key == pygame.K_c:
                figure.clear()
                print("Отчищено")

            if event.key == pygame.K_1:
                image = convertation.convert_to_image(figure)
                cv2.imwrite(f"Dataset/Circle/{len(os.listdir('Dataset/Circle/'))}.jpg", image)
                print("Сохранен круг")
            if event.key == pygame.K_2:
                image = convertation.convert_to_image(figure)
                cv2.imwrite(f"Dataset/Square/{len(os.listdir('Dataset/Square/'))}.jpg", image)
                print("Сохранен квадрат")
            if event.key == pygame.K_3:
                image = convertation.convert_to_image(figure)
                cv2.imwrite(f"Dataset/Triangle/{len(os.listdir('Dataset/Triangle/'))}.jpg", image)
                print("Сохранен треугольник")
            
    text = f1.render('1 - Circle 2 - Square 3 - Triangle', 1, (0, 0, 0))
    screen.blit(text, (10, 10))
            
    for i in range(len(figure) - 1):
        pygame.draw.line(screen, (0, 0, 0), figure[i], figure[i+1], 3)

    if pygame.mouse.get_pressed()[0]:
        figure.append(pygame.mouse.get_pos())

    clock.tick(FPS)
    pygame.display.update()
    screen.fill(bg_color)
    