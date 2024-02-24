import pygame
import cv2
import os
import keras
import numpy as np

from ConvertToImage import ConvertToImage

pygame.init() 
def save_to_dataset(figure, label_name):
    image = convertation.convert_to_image(figure)
    cv2.imwrite(f"Dataset/{label_name}/{len(os.listdir(f'Dataset/{label_name}/'))}.jpg", image)
    print(f"Сохранен {label_name}")

WIDTH = 28
FPS = 30
clock = pygame.time.Clock()
convertation = ConvertToImage()
model = keras.models.load_model('model/first.keras')
labels = ["Circle", "Nothing", "Square", "Triangle"]

# CREATING CANVAS 
screen = pygame.display.set_mode((500, 500))
figure = []

# TITLE OF CANVAS 
pygame.display.set_caption("My Board") 
exit = False
f1 = pygame.font.Font(None, 36)
label = f1.render(" ", 1, (0, 0, 0))

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
            elif event.key == pygame.K_c:
                figure.clear()
                label = f1.render(" ", 1, (0, 0, 0))
                print("Отчищено")
            elif event.key == pygame.K_p:
                image = convertation.convert_to_image(figure)
                image = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1]
                pred = model.predict(np.array([image]))
                print(pred)
                threshold = pred[0][np.argmax(pred)]
                label = f1.render(f"{labels[np.argmax(pred)]}: {threshold:0.2f}", 1, (0, 0, 0))
                print("Predicted: ", labels[np.argmax(pred)])

            elif event.key == pygame.K_1:
                save_to_dataset(figure, "Circle")
            elif event.key == pygame.K_2:
                save_to_dataset(figure, "Square")
            elif event.key == pygame.K_3:
                save_to_dataset(figure, "Triangle")
            elif event.key == pygame.K_0:
                save_to_dataset(figure, "Nothing")

    screen.blit(label, (10, 450))      
    text = f1.render('0 - Nothing 1 - Circle 2 - Square 3 - Triangle', 1, (0, 0, 0))
    screen.blit(text, (10, 10))
            
    for i in range(len(figure) - 1):
        pygame.draw.line(screen, (0, 0, 0), figure[i], figure[i+1], 3)

    if pygame.mouse.get_pressed()[0]:
        figure.append(pygame.mouse.get_pos())

    clock.tick(FPS)
    pygame.display.update()
    screen.fill(bg_color)
