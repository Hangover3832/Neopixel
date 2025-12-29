import numpy as np


def temperature_to_RGB(temperature:float) -> np.ndarray:
    """Convert a temperature value normalized to [0..1] to an RGB color value."""

    R = (1.0, 1.0, 1.0, 0.0)
    G = (0.0, 1.0, 0.8, 0.0) # avoid the visual appearance of green in the spectrum
    B = (0.0, 0.0, 1.0, 1.0) 
    RGB = np.array([R, G, B])

    if temperature <= 1.0/3:
        interval = np.array((0.0, 1.0/3))
        rgb = RGB[:, :2]
    elif temperature >= 2.0/3:
        interval = np.array((2.0/3, 1.0))
        rgb = RGB[:, 2:]
    else:
        interval = np.array((1.0/3, 2.0/3))
        rgb = RGB[:, 1:3]

    r = np.interp(temperature, interval, rgb[0])
    g = np.interp(temperature, interval, rgb[1])
    b = np.interp(temperature, interval, rgb[2])

    return np.array([r, g, b])


# From https://www.rapidtables.com/convert/color/:

def rgb_to_hsv(rgb: np.ndarray | list | tuple) -> np.ndarray:
    """
    Convert an RGB numpy array to an HSV numpy array.

    :param rgb: Input array of shape (..., 3) with RGB values in range [0, 1].
    :type rgb: numpy.ndarray

    :return: Output array of shape (..., 3) with HSV values in ranges:
        H: [0, 1], S: [0, 1], V: [0, 1]
    :rtype: numpy.ndarray
    """

    # Ensure the input is a numpy array and in the correct shape
    rgb = np.asarray(rgb, dtype=np.float32)[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    np_err = np.seterr(invalid='ignore')
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    s = np.where(maxc == 0.0, 0.0, (maxc - minc) / maxc)

    # Compute Hue (H)
    rc = np.where(maxc == minc, 0.0, (maxc - r) / (maxc - minc))
    gc = np.where(maxc == minc, 0.0, (maxc - g) / (maxc - minc))
    bc = np.where(maxc == minc, 0.0, (maxc - b) / (maxc - minc))

    h = np.where(maxc == minc, 0,
                 np.where(r == maxc, bc - gc,
                 np.where(g == maxc, 2.0 + rc - bc,
                 4.0 + gc - rc)))

    h = (h / 6.0) % 1.0
    np.seterr(**np_err)

    # Stack HSV components
    hsv = np.stack([h, s, v], axis=-1)

    return hsv


def hsv_to_rgb(hsv: np.ndarray | tuple | list) -> np.ndarray:

    hsv = np.asarray(hsv, dtype=np.float32)[..., :3]
    h, s, v= 360 * hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v * s

    # x = c * (1 - |(h/60) % 2 - 1|)
    x = c * (1.0 - np.abs((h / 60.0) % 2 - 1.0))
    m = v - c

    # prepare arrays for r', g', b'
    rp = np.zeros_like(h)
    gp = np.zeros_like(h)
    bp = np.zeros_like(h)

    # create masks for each region
    mask0 = (h >= 0) & (h < 60)
    mask1 = (h >= 60) & (h < 120)
    mask2 = (h >= 120) & (h < 180)
    mask3 = (h >= 180) & (h < 240)
    mask4 = (h >= 240) & (h < 300)
    mask5 = (h >= 300) & (h <= 360)

    rp[mask0] = c[mask0]; gp[mask0] = x[mask0]; bp[mask0] = 0.0
    rp[mask1] = x[mask1]; gp[mask1] = c[mask1]; bp[mask1] = 0.0
    rp[mask2] = 0.0;      gp[mask2] = c[mask2]; bp[mask2] = x[mask2]
    rp[mask3] = 0.0;      gp[mask3] = x[mask3]; bp[mask3] = c[mask3]
    rp[mask4] = x[mask4]; gp[mask4] = 0.0;      bp[mask4] = c[mask4]
    rp[mask5] = c[mask5]; gp[mask5] = 0.0;      bp[mask5] = x[mask5]

    # add m to match value and stack
    r = rp + m
    g = gp + m
    b = bp + m

    result = np.stack([r, g, b], axis=-1)
    return result

def rgb_to_hls(rgb: np.ndarray) -> np.ndarray:
    raise NotImplementedError

def hls_to_rgb(hls: np.ndarray) -> np.ndarray:
    raise NotImplementedError

def rgb_to_yiq(rgb: np.ndarray) -> np.ndarray:
    raise NotImplementedError

def yiq_to_rgb(yiq: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def run_test():
    from colors import SOME_COLORS

    np.set_printoptions(precision=3)
    hsv = rgb_to_hsv(list(SOME_COLORS['other'])[:3])
    print(hsv)
    print(hsv_to_rgb(hsv))

    #testing on single color vectors:
    for i, (name, color) in enumerate(SOME_COLORS.items()):
        hsv = rgb_to_hsv(color[:3])
        rgb = hsv_to_rgb(hsv)
        print(f"[{i:2}] {name:<10} {np.round(color, 3)}")
        print(f"[{i:2}] {'HSV':<10} {np.round(hsv, 3)}")
        print(f"[{i:2}] {'RGB':<10} {np.round(rgb, 3)}")
        print()

    #testing on color buffer array:
    buffer = np.vstack(list(SOME_COLORS.values()))[..., :3]
    print(f"Color buffer:\n{np.round(buffer, 3)}")
    buffer = rgb_to_hsv(buffer)
    print(f"to HSV:\n{np.round(buffer, 3)}")
    buffer = hsv_to_rgb(buffer)
    print(f"back to RGB:\n{np.round(buffer, 3)}")
    print()

    print(f"should be RGB red   [1,0,0]: {hsv_to_rgb((0.0, 1.0, 1.0))}, {hsv_to_rgb((1.0, 1.0, 1.0))}")
    print(f"should be RGB green [0,1,0]: {hsv_to_rgb((1.0/3, 1.0, 1.0))}")
    print(f"should be RGB blue  [0,0,1]: {hsv_to_rgb((2.0/3, 1.0, 1.0))}")
    print(f"should be HSV white [x,0,1]: {rgb_to_hsv((1.0, 1.0, 1.0))}")

    print(f"Temperature to RGB  at 0.0 : {temperature_to_RGB(0.0)}")
    print(f"Temperature to RGB  at 1/3 : {temperature_to_RGB(1.0/3)}")
    print(f"Temperature to RGB  at 2/3 : {temperature_to_RGB(2.0/3)}")
    print(f"Temperature to RGB  at 1.0 : {temperature_to_RGB(1.0)}")

    np.set_printoptions()

if __name__ == '__main__':
    run_test()