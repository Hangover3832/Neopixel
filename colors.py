from typing import Callable, Tuple
import numpy as np
from numpy.polynomial import Polynomial as Poly
from enum import Enum, auto
from color_conversions import rgb_to_hsv, hsv_to_rgb, yiq_to_rgb, temperature_to_RGB


SOME_COLORS = {
    'red':     np.array([1., 0., 0., 0.]),
    'green':   np.array([0., 1., 0., 0.]),
    'blue':    np.array([0. ,0., 1., 0.]),
    'yellow':  np.array([1., 1., 0., 0.]),
    'aqua':    np.array([0., 1., 1., 0.]),
    'pink':    np.array([1., 0., 1., 0.]),
    'white':   np.array([1., 1., 1., 0.]),
    'other':   np.array([0.7, 0.5, 0.2, 0.]),
}


def _create_gamma_function(values_out:np.ndarray) -> np.poly1d:
    """Calculate a polynomial that fits the given output values **exactly**."""
    return np.poly1d(np.linalg.solve(np.vander(np.linspace(0.0, 1.0, n:=len(values_out)), N=n), values_out))


def create_gamma_function(values_out:np.ndarray, values_in:np.ndarray | None = None) -> Poly:
    """Apply a (n-1)th degree polynomial least quare fit using n(values_out) distinct data points.

    :param values_out: Array of output values for the polynomial fit. For the input values, a linear space is calculated.
    :type values_out: np.ndarray
    :returns: A polynomial function that maps input values in [0, 1] to the specified output values [0, 1].
    :rtype: numpy.polynomial.Polynomial
    """
    assert (n := len(values_out)) > 1, "Gamma polynomial must have more than 1 data points"
    if values_in is None:
        values_in = np.linspace(0.0, 1.0, n) # create the linear input space from 0..1
    return Poly.fit(values_in, values_out, deg=n-1, window=(0.0, 1.0), domain=(0.0, 1.0))


class ColorMode(Enum):
    RGB = auto()
    HSV = auto()
    # YIQ = auto() # implement yiq_to_rgb() and rgb_to_yiq() first
    # HLS = auto() # implement hls_to_rgb() and rgb_to_hls() first


    @classmethod
    def kelvin_to_rgb(cls, kelvin: float) -> np.ndarray:
        return np.array(temperature_to_RGB(kelvin))

    def convert_to(self, value: np.ndarray, to_mode:'ColorMode') -> np.ndarray:
        """
        Convert color value from this color mode to the specified color mode.
        Note that the color conversion function mode_to_mode() must be globally avilable.

        :param value: The color value to be converted.
        :type value: np.ndarray
        :param to_mode: The target color mode.
        :type to_mode: ColorMode
        :return: The converted color value.
        :rtype: np.ndarray"""

        if self == to_mode:
            return value[..., :3]

        convert_function = f"{self.name.lower()}_to_{to_mode.name.lower()}"
        function = globals().get(convert_function)
        if function and callable(function):
            return function(value) # type: ignore

        raise NotImplementedError(f"{self.name} to {to_mode.name} is not implemented: '{convert_function}()'")


class PixelOrder(Enum):
    """Any combination of R, G, B and W is possible"""

    RGB     = auto()
    GRB     = auto()
    RGBW    = auto()
    GRBW    = auto()

    @property
    def num(self) -> int:
        """Return the number of LED per pixel (3 or 4)"""
        return len(self.name)

    @property
    def blank(self) -> np.ndarray:
        """Return black color value appropriate for the pixel type (RGB or RGBW)"""
        return np.array([0., 0., 0., 0.]) if self.num > 3 else np.array([0., 0., 0.])


"""
The gamma system uses a polynomial regression to map the input value to the output value.
https://en.wikipedia.org/wiki/Polynomial_regression
"""

CUSTOM_GAMMA = np.array([
    0.0,    # value for 0% brightness
    0.25,   # value for 25% brightness
    0.5,   # value for 50% brightness
    0.75,   # value for 75% brightness
    1.0     # value for 100% brightness
]) # insert more values in between if desired

LINEAR_GAMMA = np.array([0.0, 1.0])
INVERSE_GAMMA = np.array([1.0, 0.0])
SQUARE_GAMMA = np.array([0.0, 0.25, 1.0])

SIMPLE_GAMMA = np.array([0.0, 0.214, 1.0]) # sRGB 21.4% middle grey
"""This is quite close to the square gamma where the middle grey is 25%"""

DEFAULT_GAMMA = np.array([0.0, 0.11, 0.18, 0.35, 1.0]) # 18% middle grey + my own magic
""" 
I created this masterpiece of gamma based on the 18% middle grey principle
(Munsell, Sloan & Godlove, see https://en.wikipedia.org/wiki/Middle_gray) and just by my own subjective
perception of brightness on a particular neopixel stripe.
"""

SRGB_GAMMA = np.array([0.0, 0.15, 0.214, 0.37, 1.0])
# like default gamma but shifted towards sRGB

NO_DARK_GAMMA = np.array([0.02, 1.0])
# no more dark pixels :-()

CRAZY_GAMMA = np.array([1.0, 0.0, 1.0]) 
# dark pixels are bright, bright pixels are bright and middle pixels are dark ;-)


class G(Enum):
    """Gamma functions enum class."""

    @classmethod
    def plot(cls, functions: list[Callable] | None = None) -> None:
        """
        Gamma function plotter.
        
        :param function: The function to be plotted along the built-in functions.
        :type function: list[Callable] | None
        """

        import matplotlib.pyplot as plt
        from matplotlib.axes._axes import Axes
        from matplotlib.figure import Figure

        x: np.ndarray = np.linspace(0.0, 1.0, 25)
        splt = plt.subplots(2)
        fig: Figure = splt[0]
        ax: Axes = splt[1] # type: ignore
        ax[1].set(xlim=(0.0, 1.0), ylim=(0.0, 1.0)) # type: ignore
        ax[0].set(xlim=(0.0, 1.0), ylim=(-0.1, 1.1)) # type: ignore
        ax[0].set_title("Custom Gamma Function") # type: ignore
        ax[1].set_title("Built-in Gamma Functions") # type: ignore
        ax[1].set_xlabel("Input Value") # type: ignore
        ax[1].set_ylabel("LED Brightness") # type: ignore

        # Plot all built-in gamma functions
        for f in G:
            ax[1].plot(x, f.value(x), linewidth=2.0, label=f.name) # type: ignore

        if functions:
            for f in functions:
                # Plot custom gamma functions
                ax[0].plot(x, f(x), linewidth=2.0) # type: ignore

        ax[1].legend() # type: ignore
        plt.show()


    default = create_gamma_function(DEFAULT_GAMMA)
    srgb    = create_gamma_function(SRGB_GAMMA)
    simple  = create_gamma_function(SIMPLE_GAMMA)
    no_dark = create_gamma_function(NO_DARK_GAMMA)
    crazy   = create_gamma_function(CRAZY_GAMMA)
    square  = create_gamma_function(SQUARE_GAMMA)
    linear  = create_gamma_function(LINEAR_GAMMA)
    inverse = create_gamma_function(INVERSE_GAMMA)
    custom  = create_gamma_function(CUSTOM_GAMMA)


def main():
    test_gamma = lambda x: .5 + np.sin(x * 2 * np.pi) / 2
    G.plot([test_gamma])


if __name__ == '__main__':
    main()
