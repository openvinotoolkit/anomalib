import torch as th
from matplotlib import pyplot as plt


def interp(t):
    # return 3 * t**2 - 2 * t ** 3
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def perlin(width, height, scale=10, device=None):
    gx, gy = th.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = th.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = th.linspace(0, 1, scale + 1)[None, :-1].to(device)

    wx = 1 - interp(xs)
    wy = 1 - interp(ys)

    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))

    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


# def my_perlin(width, height, scale=10):


def perlin_ms(octaves=[1, 1, 1, 1], width=2, height=2, device=None):
    scale = 2 ** len(octaves)
    out = 0
    for oct in octaves:
        p = perlin(width, height, scale, device)
        out += p * oct
        scale //= 2
        width *= 2
        height *= 2
    return out


if __name__ == "__main__":
    perlin = perlin(224, 224, 2)
    plt.figure(figsize=(12, 12))
    plt.imshow(perlin)
    plt.show()

    plt.figure(figsize=(12, 12))
    for idx, rho in enumerate([1, 2, 4, 8]):
        plt.subplot(2, 2, idx + 1)
        out = perlin_ms([rho**-i for i in range(4)], 6, 6).cpu().numpy()
        # out = perlin(6, 6, 2**rho).cpu().numpy()
        plt.imshow(out)
        plt.title(f"Decay for finer grids as {rho} ** -scale")
    plt.show()
