from PIL import Image
from os import listdir
from os.path import isfile, join

new_width, new_height = 32, 32


def loop_images(source, dest):
    onlyfiles = [f for f in listdir(source) if isfile(join(source, f))]
    for file in onlyfiles:
        if (file == ".DS_Store"):
            continue
        print("reading file: ", file)
        resized_image = resize_image(source + file)
        resized_image.save(dest + "dummy_" + file)


def resize_image(image_path):
    return Image.open(image_path).resize((new_width, new_height),
                                         Image.ANTIALIAS)


loop_images("test_images_v2/", "test_images_resized_32//")