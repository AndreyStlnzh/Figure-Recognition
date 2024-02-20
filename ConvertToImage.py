import cv2
import numpy as np

WIDTH = 28

class ConvertToImage():
    def __init__(self) -> None:
        pass
        # self.figure = figure
        # self.x = [i[0] for i in figure]
        # self.y = [i[1] for i in figure]
        # self.center = (np.mean(self.x), np.mean(self.y))

        
    def normalization(self, figure, center=None, integer=False):
        res_figure = [None] * len(figure)
        min_x = min([i[0] for i in figure])
        min_y = min([i[1] for i in figure])

        for i in range(len(figure)):
            res_figure[i] = (figure[i][0] - min_x, figure[i][1] - min_y)

        koef_x = WIDTH / max([i[0] for i in res_figure])
        koef_y = WIDTH / max([i[1] for i in res_figure])

        koef = koef_x if koef_x <= koef_y else koef_y

        for i in range(len(figure)):
            if integer:
                res_figure[i] = (int(res_figure[i][0] * koef), int(res_figure[i][1] * koef))
            else:
                res_figure[i] = (res_figure[i][0] * koef, res_figure[i][1] * koef)
        
        if center:
            center = ((center[0] - min_x) * koef, (center[1] - min_y) * koef)

        return res_figure, center
    

    def convert_to_image(self, figure):
        res_figure, _ = self.normalization(figure, integer=True)
        blank_image = np.zeros((WIDTH, WIDTH), np.uint8)
        blank_image[:, :] = 255

        for i in range(len(res_figure) - 1):
            cv2.line(blank_image, res_figure[i], res_figure[i+1], 0, 2)

        return blank_image
    
if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    figure = [(233, 149), (201, 198), (139, 265), (112, 297), (112, 298), (116, 298), (168, 306), (256, 315), (303, 319), (314, 316), (316, 312), (314, 301), (303, 272), (284, 226), (260, 185), (245, 154), (240, 144), (240, 144), (240, 144)]
    c = ConvertToImage()
    blank_image = c.convert_to_image(figure)
    # plt.imshow(blank_image)
    # plt.show()