from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from colors import ColorMode, PixelOrder, G


PixelIndex = int | list[int] | tuple[int, ...] | slice
PixelValue = np.ndarray | list[float] | tuple[float, ...] | float | int

#-----------------------------------------------------------
class NeopixelDevice(ABC):
    def __init__(self, *args, pixel_order:PixelOrder = PixelOrder.GRB, **kwargs) -> None:
        super().__init__()
        self._neopixel: Neopixel | None = None
        self._pixel_order: PixelOrder = pixel_order
        self.is_simulated: bool = False
        self.args: tuple = args
        self.kwargs: dict = kwargs
        self._is_open: bool = False


    @abstractmethod
    def write_to_device(self, buffer:NDArray[np.float32]) -> Any:
        raise NotImplementedError


    def _write_buffer(self, buffer:NDArray[np.float32]) -> Any:

        # rearange the rgb_buffer to the correct PixelOrder
        # Here, we allow every possible pixel order with R,G,B and optional W
        buffer = buffer[:, [self.pixel_order.name.index(c) for c in 'RGBW' if c in self.pixel_order.name]]
        self.write_to_device(buffer)


    def _to_uint8(self, buffer:NDArray[np.float32]) -> NDArray[np.uint8]:
        # scale to [0, 255], and convert to uint8:
        return np.clip(np.round(255 * buffer), 0, 255).astype(np.uint8)


    def open_(self) -> bool:
        result = not self._is_open
        self._is_open = True
        return result


    def close_(self) -> bool:
        result = self._is_open
        self._is_open = False
        return result


    def __del__(self):
        if self._is_open:
            self.close_()

    def set_neopixel(self, neopixel: Neopixel) -> None:
        self._neopixel = neopixel

    @property
    def pixel_order(self) -> PixelOrder:
        return self._pixel_order
    
    @property
    def neopixel(self) -> Neopixel:
        if self._neopixel is None:
            raise ValueError("`neopixel` attribute not set.")
        return self._neopixel

    @neopixel.setter
    def neopixel(self, neopixel:Neopixel) -> None:
        self.set_neopixel(neopixel)


#-----------------------------------------------------------
class Neopixel:

    SPI_HIGH_BIT        = 0xC0
    SPI_LOW_BIT         = 0x80
    SPI_HIGH_BIT2       = 0x0C
    SPI_LOW_BIT2        = 0x08

    def __init__(self, 
            num_pixels:int,
            *,
            gamma_func: Callable = G.default.value,
            color_mode: ColorMode = ColorMode.HSV,
            brightness: float = 1.0, 
            auto_write: bool = False,
            max_power: float = 0.0
                ) -> None:

        # Public attributes
        self.reversed: bool = False
        self.auto_write = auto_write
        self.watts_per_led: np.ndarray = np.array([0.081, 0.081, 0.08, 0.09])

        # Private attributes
        # self._pixel_order: PixelOrder = pixel_order
        self._color_mode: ColorMode = color_mode
        self._current_power: float = 0.0
        self._max_power: float = max_power
        self._index: int = 0
        self._brightness: float = float(np.clip(brightness, 0., 1.))
        self._gamma_func: Callable = gamma_func
        self._update_counter: int = 0
        self._num_pixels = num_pixels
        self._mini_screens: list[np.ndarray] = []
        self._device: NeopixelDevice | None = None
        self._pixel_buffer: np.ndarray | None = None


    def to(self, device: NeopixelDevice) -> 'Neopixel':
        is_new = self._pixel_buffer is None
        self._device = device
        if is_new:
            self._pixel_buffer = np.zeros((self._num_pixels, device.pixel_order.num), dtype=np.float32)
        self._device.neopixel = self
        self._device.open_()
        return self if is_new else self.auto_show()


    def close_(self):
        if self._device is not None and self._device._is_open:
            self._device.close_()


    def begin_update(self) -> 'Neopixel':
        """In auto write mode, **no** update will occour until end_update() and the counter is 0"""
        self._update_counter += 1
        return self


    def end_update(self, reset:bool=False) -> 'Neopixel':
        self._update_counter -= 1

        if reset or self._update_counter < 0:
            self._update_counter = 0

        if self._update_counter <= 0:
            return self.auto_show()

        return self


    def add_virtual_screen(self, config: np.ndarray) -> int:
        """
        Add a virtual 2 dimensional screen area to the Neopixel stripe.
        
        :param config: A 2-dimensional array that contains all the pixel indices that bild up the screen,
        mapping from the top left to the bottom right on the virtual screen.
        :type config: np.ndarray[[int, ...], ...]
        :returns: The index number of the newly created screen.
        :rtype: int
        """

        self._mini_screens.append(config.astype(np.int16))
        return len(self._mini_screens)-1


    def virtual_screen_data(self, index: int, data: np.ndarray, color_mode: ColorMode  | None = None) -> 'Neopixel':
        """
        Put pixel data onto a virtual screen.

        :param index: The index of the virtuel screen created and returned by `add_virtual_screen()`.
        :type index: int
        :param data: A 3-dimensional array that contains RGB(W) Pixel values to put on the virtual screen.
        Note that the shape of the `data` array must match the shape of the virtual screen.
        :type data: np.ndarray[[[float, ...]]]
        :returns: self
        :rtype: Neopixel
        """

        assert data.shape[:2] == self._mini_screens[index].shape
        indices = self._mini_screens[index].flatten()
        data = data.reshape(-1, data.shape[2]).squeeze()

        self.begin_update()
        # no auto write in this loop
        for d, i in enumerate(indices):
            self.set_value(i, data[d], color_mode=color_mode)

        return self.end_update().auto_show()


    def _write_buffer(self) -> None:
        """Write pixel data to the Neopixels device"""

        rgb_buffer = self.pixel_buffer.copy()

        # Apply brightness and gamma correction
        rgb_buffer = np.clip(self._gamma_func(rgb_buffer * self._brightness), 0.0, 1.0)

        # calculate power consumption
        watts = self.watts_per_led if self.has_W else self.watts_per_led[:3]
        self._current_power = np.sum(watts * rgb_buffer)  

        # Power consumption limiter
        if (self._max_power > 1e-6) and (self._current_power > self._max_power):
            rgb_buffer *= self._max_power/self._current_power
            self._current_power = self._max_power

        if self.reversed:
            rgb_buffer = rgb_buffer[::-1]

        # Send data to device:
        self.device._write_buffer(rgb_buffer)


    def __setitem__(self, index: int | slice, value: PixelValue) -> None:
        """Indexed or sliced Neopixel access"""
        self.set_value(index, value)


    def __getitem__(self, index: int | slice) -> np.ndarray:
        rgb = self.pixel_buffer[index]
        result = ColorMode.RGB.convert_to(rgb, self._color_mode)

        if rgb.shape[-1] > 3 and self.has_W:
            # Put back the white channel to the last dimension if available
            result = np.concatenate([result, rgb[..., 3][..., np.newaxis]], axis=-1)

        return result


    def __len__(self) -> int:
        """Get the number of pixels in the strip."""
        return self.pixel_buffer.shape[0]
    

    def set_temperature(self, index:PixelIndex, temperature:float, brightness:float = 1.0) -> 'Neopixel':
        """Set pixel value at index using an approximation for the black body heat radiation.
        The temperature ranges from [0.0 .. 1.0]"""

        self.pixel_buffer[index, :3] = np.clip(brightness * self._color_mode.kelvin_to_rgb(temperature), 0.0, 1.0)
        return self.auto_show()



    def set_value(self, index: PixelIndex, value: PixelValue, color_mode: ColorMode | None = None) -> 'Neopixel':
        """
        This is the core routine that writes pixel value at index to self.pixel_buffer:
        if value is a single number, it affects the White LED only in a RGBW stripe. A non RGBW stripe
        will throw an exception in this case.
        """
        value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)

        if value.ndim == 0:# a single number applies to the white LED only
            if not self.has_W:
                #ignore if no white LED can be set
                return self

            self.pixel_buffer[index, 3] = float(np.clip(value, 0.0, 1.0))
            return self.auto_show()

        if (value.shape[-1] == 2) or (value.shape[-1] > 4):
            raise ValueError("Wrong number of values")

        if (value.ndim == 2) and (isinstance(index, int)):
            index = slice(index, index+value.shape[0])
        
        if value.shape[-1] == 1: # an array of single numbers is boradcasted to the white pixels only:
            if not self.has_W:
                #ignore if no white LED can be set
                return self
            
            self.pixel_buffer[index, -1] = value[:, 0]
            return self.auto_show()

        rgb = (color_mode or self._color_mode).convert_to(value, ColorMode.RGB)

        if (value.shape[-1] > 3) and self.has_W:
            # Put back the white channel to the last dimension if RGBW
            rgb = np.concatenate([rgb, value[..., 3][..., np.newaxis]], axis=-1)
            self.pixel_buffer[index] = rgb
        else:
            self.pixel_buffer[index, :3] = rgb
    
        return self.auto_show()


    def next_value(self, value: PixelValue, color_mode: ColorMode | None = None) -> 'Neopixel':
        """Set the value for the next pixel in the iteration"""
        return self.set_value(result := next(self), value=value, color_mode=color_mode)

    def fill(self, value: PixelValue, color_mode: ColorMode | None = None) -> 'Neopixel':
        """Fill all pixels with a given value"""
        return self.set_value(slice(None), value=value, color_mode=color_mode)


    def clear(self) -> 'Neopixel':
        """Clear all pixels by setting them to black."""
        return self.fill(self.blank, color_mode=ColorMode.RGB)

    def auto_show(self) -> 'Neopixel':
        return self.show() if self.auto_write and self._update_counter <= 0 else self

    def show(self) -> 'Neopixel':
        """Update the NeoPixels with the current pixel buffer."""
        self._write_buffer()
        return self


    def roll(self, shift: int = 1, value: PixelValue | None = None) -> 'Neopixel':
        """
        Roll the pixel buffer by the specified shift amount.
        
        :param shift: Number of positions to shift. Positive values shift right, negative values shift left. Defaults to 1.
        :type shift: int
        :param value: If `value` is None, pixels that roll off one end will reappear at the other end. If a `value` is provided, 
            pixels that roll in will be set to this value. If `value` is a single number, only the white LED is affected in a RGBW stripe.
        :type value: PixelValue | None
        :returns: The current instance of Neopixel.
        :rtype: Neopixel
        """
        if shift == 0:
            return self
        
        if value is None:
            self._pixel_buffer = np.roll(self.pixel_buffer, shift, axis=0)
            return self.auto_show()
        else:
            if not isinstance(value, (float, int)):
                value = np.asarray(value)

            if shift > 0:
                self.pixel_buffer[shift:] = self.pixel_buffer[:-shift]
                return self.set_value(slice(None,shift), value)
            else:
                self.pixel_buffer[:shift] = self.pixel_buffer[-shift:]
                return self.set_value(slice(shift, None), value)


    def create_gradient(self, 
                        from_value:PixelValue, 
                        to_value:PixelValue, 
                        index:int=0, count:int=0, 
                        color_mode:ColorMode | None = None) -> 'Neopixel':
        """Create a color gradient. If count=0, the whole pixel array is used"""

        from_value = np.asarray(from_value, dtype=np.float32)
        to_value = np.asarray(to_value, dtype=np.float32)

        if from_value.shape != to_value.shape:
            raise ValueError("from_value and to_value mut have the same shape")

        if count <= 0:
            count = self.num_pixels

        if from_value.shape: # RGB(W) value array
            rgb = [
                np.linspace(from_value[0], to_value[0], count),
                np.linspace(from_value[1], to_value[1], count),
                np.linspace(from_value[2], to_value[2], count)]

            if self.has_W:
                if from_value.shape[0] > 3:
                    rgb.append(np.linspace(from_value[3], to_value[3], count))

            gradient = np.stack(rgb, axis=1, dtype=np.float32)
        else: # W only value array
            gradient = np.linspace(from_value, to_value, count)[:, np.newaxis].astype(np.float32)
    
        return self.set_value(index, gradient[:self.num_pixels-index], color_mode=color_mode)


    def __call__(self, index: PixelIndex | None = None, value: PixelValue | None = None) -> 'Neopixel':
        """
        Calls `show()` that updates the stripe if no index or value is provided. If index is provided but no value,
        the pixel(s) at index gets cleared. If both index and value are provided, the pixel
        at index is set to the specified value.

        :param index: Pixel index number(s)
        :type index: int or slice
        :param value: Pixel value. 
            If value is a number, it affects the White LED only in a RGBW stripe. A non RGBW stripe
            will throw an exception in this case.
        :type value: number or array like
        """

        if index is None:
            return self.show()

        if value is None:
            return self.set_value(index, self.blank)

        return self.set_value(index, value)


    # __dunder__ methods =============================

    def __del__(self):
        if self._device is not None:
            self._device.close_()

    def __iter__(self) -> 'Neopixel':
        return self

    def __next__(self) -> int:
        """ Iterate over the indices of the pixels. """
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        self._index += 1
        return self._index - 1

    def __iadd__(self, value: np.ndarray | float) -> 'Neopixel':
        """Add value to the pixel buffer in RGB space, e.g. pixels += 0.1"""
        self._pixel_buffer =  np.clip(self.pixel_buffer + value, 0., 1.)
        return self.auto_show()

    def __imul__(self, value: np.ndarray | float) -> Neopixel:
        """Multiply value with the pixel buffer in RGB space, e.g. neo *= 0.9"""
        self._pixel_buffer = np.clip(self.pixel_buffer * value, 0., 1.)
        return self.auto_show()

    def __ilshift__(self, amount: int) -> 'Neopixel':
        """roll to the left by amount, e.g. `pixels <<= 1`"""
        return self.roll(-int(abs(amount)))
    
    def __irshift__(self, amount: int) -> 'Neopixel':
        """roll to the right by amount, e.g. `pixels >>= 1`"""
        return self.roll(int(abs(amount)))

    def __invert__(self)-> 'Neopixel':
        """Invert all colors of all pixels, e.g. `~pixels` """
        self._pixel_buffer = 1.0 - self.pixel_buffer
        return self.show() if self.auto_write else self


    # properties =============================

    @property
    def blank(self) -> np.ndarray:
        """Get a black color value appropriate for the pixel type (RGB or RGBW)."""
        return self.device.pixel_order.blank
    
    @property
    def has_W(self) -> bool:
        """Check if the pixel type has a white channel."""
        return self.pixel_buffer.shape[-1] > 3

    @property
    def gamma_func(self) -> Callable[[float], float]:
        """Get the gamma function used for brightness correction."""
        return self._gamma_func

    @gamma_func.setter
    def gamma_func(self, new_gamma: Callable[[float], float]) -> None:
        """Set a new gamma function for color correction."""
        self._gamma_func = new_gamma
        self.auto_show()

    @property 
    def color_mode(self) -> ColorMode:
        """Get the current color mode."""
        return self._color_mode

    @color_mode.setter
    def color_mode(self, new_mode: ColorMode) -> None:
        """Set a new color mode."""
        self._color_mode = new_mode

    @property
    def brightness(self) -> float:
        """Get the current brightness level."""
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        """Set a new brightness level."""
        self._brightness = float(np.clip(value, 0.0, 1.0))
        self.auto_show()
    
    @property
    def num_pixels(self) -> int:
        """Get the number of pixels in the strip."""
        return len(self)

    @property
    def power_consumption(self) -> float:
        """Returns the total power consumption [0..1]"""
        return self._current_power
    
    @property
    def max_power(self) -> float | None:
        return self._max_power

    @max_power.setter
    def max_power(self, max_power: float) -> None:
        self._max_power = max_power
        self.auto_show()

    @property
    def pixel_buffer(self) -> np.ndarray:
        assert self._pixel_buffer is not None
        return self._pixel_buffer

    @property
    def device(self) -> NeopixelDevice:
        if self._device is None:
            raise ValueError("Output device is not set. Use the method `to()` to set the output device")

        return self._device
