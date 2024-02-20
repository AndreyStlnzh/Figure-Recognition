import pygame 
pygame.init() 

FPS = 20
clock = pygame.time.Clock()


# CREATING CANVAS 
screen = pygame.display.set_mode((500, 500))
figure = []

# TITLE OF CANVAS 
pygame.display.set_caption("My Board") 
exit = False

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
            
    for i in range(len(figure) - 1):
        pygame.draw.line(screen, (0, 0, 0), figure[i], figure[i+1], 3)
    

    if pygame.mouse.get_pressed()[0]:
        figure.append(pygame.mouse.get_pos())

    clock.tick(FPS)
    pygame.display.update()
    screen.fill(bg_color)
    