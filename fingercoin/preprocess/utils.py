from PIL import Image, ImageDraw
import math
import copy

def apply_kernel_at(get_value, kernel, i, j):
    #kernel = list(kernel)
    kernel_size = len(kernel)
    result = 0
    for k in range(0, kernel_size):
        for l in range(0, kernel_size):
            pixel = get_value(i + k - kernel_size // 2, j + l - kernel_size // 2)
            result += pixel * kernel[k][l]
    return result

def apply_to_each_pixel(pixels, f):
    for i in range(0, len(pixels)):
        for j in range(0, len(pixels[i])):
            pixels[i][j] = f(pixels[i][j])

def calculate_angles(image, block):
    (x, y) = image.size
    image_load = image.load()
    get_pixel = lambda x, y: image_load[x, y]

    f = lambda x, y: 2 * x * y
    g = lambda x, y: x ** 2 - y ** 2

    ySobel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    xSobel = transpose(ySobel)

    result = [[] for i in range(1, x, block)]

    for i in range(1, x, block):
        for j in range(1, y, block):
            nominator = 0
            denominator = 0
            for k in range(i, min(i + block , x - 1)):
                for l in range(j, min(j + block, y - 1)):
                    Gx = apply_kernel_at(get_pixel, xSobel, k, l)
                    Gy = apply_kernel_at(get_pixel, ySobel, k, l)
                    nominator += f(Gx, Gy)
                    denominator += g(Gx, Gy)
            angle = (math.pi + math.atan2(nominator, denominator)) / 2
            result[(i - 1) // block].append(angle)

    return result

def flatten(ls):
    return reduce(lambda x, y: x + y, ls, [])

def transpose(ls):
    return map(list, zip(*ls))

def gauss(x, y):
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))

def kernel_from_function(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel

def gauss_kernel(size):
    return kernel_from_function(size, gauss)

def apply_kernel(pixels, kernel):
    apply_kernel_with_f(pixels, kernel, lambda old, new: new)

def apply_kernel_with_f(pixels, kernel, f):
    size = len(kernel)
    for i in range(size // 2, len(pixels) - size // 2):
        for j in range(size // 2, len(pixels[i]) - size // 2):
            pixels[i][j] = f(pixels[i][j], apply_kernel_at(lambda x, y: pixels[x][y], kernel, i, j))

def smooth_angles(angles):
    cos_angles = copy.deepcopy(angles)
    sin_angles = copy.deepcopy(angles)
    apply_to_each_pixel(cos_angles, lambda x: math.cos(2 * x))
    apply_to_each_pixel(sin_angles, lambda x: math.sin(2 * x))

    kernel = gauss_kernel(5)
    apply_kernel(cos_angles, kernel)
    apply_kernel(sin_angles, kernel)

    for i in range(0, len(cos_angles)):
        for j in range(0, len(cos_angles[i])):
            cos_angles[i][j] = (math.atan2(sin_angles[i][j], cos_angles[i][j])) / 2

    return cos_angles

def load_image(image):
    (x, y) = image.size
    im_load = image.load()

    result = []
    for i in range(0, x):
        result.append([])
        for j in range(0, y):
            result[i].append(im_load[i, j])

    return result

def load_pixels(image, pixels):
    (x, y) = image.size
    im_load = image.load()

    for i in range(0, x):
        for j in range(0, y):
            im_load[i, j] = int(pixels[i][j])

def draw_lines(image, angles, block):
    (x, y) = image.size
    result = image.convert("RGB")

    draw = ImageDraw.Draw(result)

    for i in range(1, x, block):
        for j in range(1, y, block):
            tang = math.tan(angles[(i - 1) / block][(j - 1) / block])

            (begin, end) = get_line_ends(i, j, block, tang)
            draw.line([begin, end], fill=150)
    del draw

    return result

def get_line_ends(i, j, block, tang):
    if -1 <= tang and tang <= 1:
        begin = (i, (-block/2) * tang + j + block/2)
        end = (i + block, (block/2) * tang + j + block/2)
    else:
        begin = (i + block/2 + block/(2 * tang), j + block/2)
        end = (i + block/2 - block/(2 * tang), j - block/2)
    return (begin, end)

##Frequency calculation
def block_frequency(i, j, block, angle, im_load):
    tang = math.tan(angle)
    ortho_tang = -1 / tang

    (x_norm, y_norm, step) = vec_and_step(tang, block)
    (x_corner, y_corner) = (0 if x_norm >= 0 else block, 0 if y_norm >= 0 else block)

    grey_levels = []

    for k in range(0, block):
        line = lambda x: (x - x_norm * k * step - x_corner) * ortho_tang + y_norm * k * step + y_corner
        points = points_on_line(line, block)
        level = 0
        for point in points:
            level += im_load[point[0] + i * block, point[1] + j * block]
        grey_levels.append(level)

    treshold = 100
    upward = False
    last_level = 0
    last_bottom = 0
    count = 0.0
    spaces = len(grey_levels)
    for level in grey_levels:
        if level < last_bottom:
            last_bottom = level
        if upward and level < last_level:
            upward = False
            if last_bottom + treshold < last_level:
                count += 1
                last_bottom = last_level
        if level > last_level:
            upward = True
        last_level = level

    return count / spaces if spaces > 0 else 0

def points_on_line(line, block):
    im = Image.new("L", (block, 3 * block), 100)
    draw = ImageDraw.Draw(im)
    draw.line([(0, line(0) + block), (block, line(block) + block)], fill=10)
    im_load = im.load()

    points = []
    for x in range(0, block):
        for y in range(0, 3 * block):
            if im_load[x, y] == 10:
                points.append((x, y - block))

    del draw
    del im

    #dist = lambda (x, y): (x - block / 2) ** 2 + (y - block / 2) ** 2
    dist = lambda x: sum((n - block / 2) ** 2 for n in x)

    #return sorted(points, cmp = lambda x, y: dist(x) < dist(y))[:block]
    return sorted(points, key = lambda x: dist(x))[:block]

def vec_and_step(tang, block):
    (begin, end) = get_line_ends(0, 0, block, tang)
    (x_vec, y_vec) = (end[0] - begin[0], end[1] - begin[1])
    length = math.hypot(x_vec, y_vec)
    (x_norm, y_norm) = (x_vec / length, y_vec / length)
    step = length / block

    return (x_norm, y_norm, step)

def freq(image, block, angles):
    (x, y) = image.size
    im_load = image.load()
    freqs = [[0] for i in range(0, x // block)]

    for i in range(1, x // block - 1):
        for j in range(1, y // block - 1):
            freq = block_frequency(i, j, block, angles[i][j], im_load)
            freqs[i].append(freq)
        freqs[i].append(0)

    freqs[0] = freqs[-1] = [0 for i in range(0, y // block)]

    return freqs


##Kernel for Gabor filter
def gabor_kernel(block, angle, freq):
    cos = math.cos(angle)
    sin = math.sin(angle)

    yangle = lambda x, y: x * cos + y * sin
    xangle = lambda x, y: -x * sin + y * cos

    xsigma = ysigma = 4

    return kernel_from_function(block, lambda x, y:
            math.exp(-(
                (xangle(x, y) ** 2) / (xsigma ** 2) +
                (yangle(x, y) ** 2) / (ysigma ** 2)) / 2) *
            math.cos(2 * math.pi * freq * xangle(x, y)))

#############

def reverse(ls):
    cpy = ls[:]
    cpy.reverse()
    return cpy
