from PIL import Image
from os import listdir
from os.path import isfile, join

angle = 90


def loop_images(source, dest):
    onlyfiles = [f for f in listdir(source) if isfile(join(source, f))]
    for file in onlyfiles:
        if (file == ".DS_Store"):
            continue
        print("reading file: ", file)
        img = Image.open(source + file)
        rotated_image1 = rotate_image(img, angle * 1)
        rotated_image2 = rotate_image(img, angle * 2)
        rotated_image3 = rotate_image(img, angle * 3)
        #rotated_image4 = rotate_image(img, angle * 4)
        #rotated_image5 = rotate_image(img, angle * 5)
        #rotated_image6 = rotate_image(img, angle * 6)
        #rotated_image7 = rotate_image(img, angle * 7)
        new_name0 = "rotated" + str(angle * 0) + "_" + file
        new_name1 = "rotated" + str(angle * 1) + "_" + file
        new_name2 = "rotated" + str(angle * 2) + "_" + file
        new_name3 = "rotated" + str(angle * 3) + "_" + file
        #new_name4 = "rotated" + str(angle * 4) + "_" + file
        #new_name5 = "rotated" + str(angle * 5) + "_" + file
        #new_name6 = "rotated" + str(angle * 6) + "_" + file
        #new_name7 = "rotated" + str(angle * 7) + "_" + file
        img.save(dest + new_name0)
        rotated_image1.save(dest + new_name1)
        rotated_image2.save(dest + new_name2)
        rotated_image3.save(dest + new_name3)
        #rotated_image4.save(dest + new_name4)
        #rotated_image5.save(dest + new_name5)
        #rotated_image6.save(dest + new_name6)
        #rotated_image7.save(dest + new_name7)


def rotate_image(img, ang):
    return img.rotate(ang)


loop_images("images_resized_32/", "images_rotated90_32/")