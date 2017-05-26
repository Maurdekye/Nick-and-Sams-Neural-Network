import numpy as np



def format_for_network(image: list, label: bytes) -> list:
    output = [[]]
    output[label] = 1
    input = [image.GetLength(0) * image.GetLength(1)]

    index = 0

    for x in image[0]:
        for y in image[1]:
            input[index] = image[x, y] / 255.0
            index += 1

    return (input, output)



