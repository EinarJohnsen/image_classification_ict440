from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join


def loop_images(source, dest):
    onlyfiles = [f for f in listdir(source) if isfile(join(source, f))]
    for file in onlyfiles:
        if (file == ".DS_Store"):
            continue
        print("reading file: ", file)
        img = Image.open(source + file)
        blurred = blur_image(img)
        blurred.save(dest + "blurred_" + file)
        img.save(dest + file)


def blur_image(img):
    return img.filter(ImageFilter.GaussianBlur(5))


loop_images("images_rotated_32/", "images_rotated_blurred_32/")