import random
from pathlib import Path

import rootutils
import numpy as np
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)


def seed_everything(seed):
    global random_state

    random_state = np.random.RandomState(seed)
    random.seed(seed)


# функция генерирующая синтетическое изображения с линиями
# и вычисляет точки пересечения линий
def draw_lines(image, nb_lines=10):
    # TODO!
    return image, points


def draw_polygon(image, max_sides=8):
    # TODO!
    return image, points


def draw_multiple_polygons(
    image, max_sides=8, nb_polygons=30, kernel_boundaries=(50, 100)
):
    # TODO!
    return image, points


def draw_ellipses(image, nb_ellipses=20):
    # TODO!
    return image, np.empty((0, 2), dtype=int)


def draw_star(image, nb_branches=6):
    # TODO!
    return image, points


def draw_checkerboard(image, max_rows=7, max_cols=7, transform_params=(0.05, 0.15)):
    # TODO!
    return image, points


def draw_stripes(
    image, max_nb_cols=13, min_width_ratio=0.04, transform_params=(0.1, 0.1)
):
    # TODO!
    return image, points


def draw_cube(
    image, min_size_ratio=0.2, scale_interval=(0.4, 0.6), trans_interval=(0.5, 0.2)
):
    # TODO!
    return image, points


def gaussian_noise(image):
    cv2.randu(image, 0, 255)
    return image, np.empty((0, 2), dtype=int)


# функция для генерации фонового изображения
def generate_background(background_size):
    # TODO
    return image


# функция, которая генерирует картинку, точки и сохраняет их в out_dir
def generate_data(n, out_dir, draw_fn, background_size, image_size, blur_size):
    #
    file = out_dir / str(n).zfill(6)

    # сгенерировать изображение с фоном
    image = generate_background(background_size)

    # на фоне нарисовать примитивы
    image, points = draw_fn(image)

    # здесь скорей всего нужно будет сгладить 
    # и отресайзить выходное изображение в image_size
    
    file = out_dir / str(n).zfill(6)
    
    cv2.imwrite(file.with_suffix(".png"), image) # сохраняем grayscale синтетическое изображение
    np.save(file.with_suffix(".npy"), points) # сохраняем ключевые точки, размер N x 2 (если точек нет то N = 0)
    

def main(cfg):
    # зададим seed для воспроизводимости результатов генерации
    seed_everything(cfg.seed)

    # создать директорию cfg.data_dir
    # если она уже создана - удалить вместе с ее содержимым

    # словарь, который мапит название функции на ее реализацию
    draw_fn = {
            # функция генерирующая синтетическое изображение с линиями и вычисляет точки пересечения линий
            "draw_lines": draw_lines, 
            # другие функции для генерации разных видов синтетических изображений и координат ключевых точек
            "draw_polygon": draw_polygon,
            "draw_multiple_polygons": draw_multiple_polygons,
            "draw_ellipses": draw_ellipses,
            "draw_star": draw_star,
            "draw_checkerboard": draw_checkerboard,
            "draw_stripes": draw_stripes,
            "draw_cube": draw_cube,
            "gaussian_noise": gaussian_noise,
    }[name]

    # перебираем все примитивы из конфига
    for name, size in cfg.primitives.items():
        # создать директорию 
        out_dir = Path(cdg.data_dir) / name

        # генерируем size штук синтетических изображений типа name
        # в дальнейшем этот цикл надо распараллелить с помощью joblib 
        # для увеличения скорости генерации данных
        for n in range(size):
            generate_data(n, out_dir, draw_fn, cfg.background_size, cfg.image_size, cfg.blur_size)


if __name__ == "__main__":
    params = Path(__file__).stem
    cfg = OmegaConf.load("params.yaml")[params]

    main(cfg)