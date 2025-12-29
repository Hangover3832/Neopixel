from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray
from colors import ColorMode, PixelOrder, G


PixelIndex = int | list[int] | tuple[int, ...] | slice
PixelValue = np.ndarray | list[float] | tuple[float, ...] | float | int

class Spi_Clock(Enum): # SPI clock rates
    CLOCK_400KHZ  = 1_625_000
    CLOCK_800KHZ  = 3_250_000
    CLOCK_1200KHZ = 6_500_000

class Spi_Bit_Encoding(Enum):
    # for SPI devices:
    SPI_HIGH_BIT   = 0xC0
    SPI_LOW_BIT    = 0x80
    SPI_HIGH_BIT2  = 0x0C
    SPI_LOW_BIT2   = 0x08

#-----------------------------------------------------------
class OutputDevice:
    """Common abstract base class for digital switching output, used for custom chip select"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def enable(self):
        raise NotImplementedError
    
    @abstractmethod
    def disable(self):
        raise NotImplementedError

    def set_neopixel(self, neopixel: Neopixel) -> None:
        self._neopixel = neopixel

    @property
    @abstractmethod
    def enabled(self) -> bool:
        raise NotImplementedError

    @property
    def neopixel(self) -> Neopixel | None:
        return self._neopixel


#-----------------------------------------------------------
class NoOutputDevice(OutputDevice):
    """Dummy output device when custom chip select is not used"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._enabled: bool = False

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def is_active(self) -> bool:
        return self._enabled


#-----------------------------------------------------------
class NeopixelDevice(ABC):
    def __init__(self, *, pixel_order:PixelOrder, custom_cs:OutputDevice | None = None, **kwargs) -> None:
        super().__init__()
        self._neopixel: Neopixel | None = None
        self._pixel_order: PixelOrder = pixel_order
        self.is_simulated: bool = False
        self._cs = custom_cs
    
    @abstractmethod
    def write_to_device(self, buffer:NDArray[np.uint8]):
        raise NotImplementedError

    def write_bytes(self, buffer:NDArray[np.uint8]) -> None:
        if self._cs is not None:
            self._cs.enable()

        # rearange the rgb_buffer to the correct PixelOrder
        # Here, we allow every possible pixel order with R,G,B and optional W
        buffer = buffer[:, [self.pixel_order.name.index(c) for c in 'RGBW' if c in self.pixel_order.name]]
        self.write_to_device(buffer)

        if self._cs is not None:
            self._cs.disable()

    def close(self) -> None:
        pass


    def set_neopixel(self, neopixel: Neopixel) -> None:
        self._neopixel = neopixel


    @property
    def pixel_order(self) -> PixelOrder:
        return self._pixel_order
    
    @property
    def neopixel(self) -> Neopixel:
        if self._neopixel is None:
            raise ValueError("`neopixel` attribute not set. Use the method `set_neopixel()`")

        return self._neopixel
 

#-----------------------------------------------------------
class SPIDevice(NeopixelDevice):

    def __init__(self, *, pixel_order:PixelOrder, custom_cs:OutputDevice | None = None, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=custom_cs, **kwargs)
        self._spi_buffer: NDArray[np.uint8] | None = None


    def set_neopixel(self, neopixel: Neopixel):
        super().set_neopixel(neopixel)
        # assert self.neopixel is not None
        
        if self._pixel_order.num == 4:
            self._double_bits_per_pixel = 16
            self._msb_mask = 0x80000000
            self._c_mask = 0xFFFFFFFF
        else:
            self._double_bits_per_pixel = 12
            self._msb_mask = 0x800000
            self._c_mask = 0xFFFFFF

        # Pre-allocate buffer for the encoded bits
        self._spi_buffer = self._spi_buffer = np.zeros([self._double_bits_per_pixel, self.neopixel.num_pixels], dtype=np.uint8)


    def write_to_device(self, buffer: NDArray[np.uint8]):
        if self.neopixel is None or self._spi_buffer is None:
            raise ValueError("Attribute `neopixel` is not set")

        if buffer.ndim != 2:
            raise ValueError("buffer must be 2D [n, [r,g,b,(w)]]")

        rgb_buffer = buffer.copy()

        # Convert [r, g, b, (w)] to uint32:
        if self.neopixel.has_W:
            rgb_buffer = rgb_buffer * np.array([0x1_00_00_00, 0x1_00_00, 0x1_00, 1], dtype=np.uint32)
        else:
            rgb_buffer = rgb_buffer * np.array([0x1_00_00, 0x1_00, 1], dtype=np.uint32)

        # reduce the array to a single uint32 per pixel:
        rgb_buffer = np.bitwise_or.reduce(rgb_buffer, axis=1, dtype=np.uint32)

        # shift out 2 bits of each pixel and encode them to a byte for SPI transmission:
        for i in range(self._double_bits_per_pixel):
            bit1 = (rgb_buffer & self._msb_mask).astype(bool)
            rgb_buffer = ((rgb_buffer << 1) & self._c_mask).astype(np.uint32)
            bit2 = (rgb_buffer & self._msb_mask).astype(bool) 
            rgb_buffer = ((rgb_buffer << 1) & self._c_mask).astype(np.uint32)
            # encode 2 pixel bits into 1 SPI byte:
            self._spi_buffer[i] = (np.where(
                        bit1, Spi_Bit_Encoding.SPI_HIGH_BIT.value,  Spi_Bit_Encoding.SPI_LOW_BIT.value
                        ) | np.where(
                        bit2, Spi_Bit_Encoding.SPI_HIGH_BIT2.value, Spi_Bit_Encoding.SPI_LOW_BIT2.value)
                    ).astype(np.uint8)
            
        # self._buffer is now ready to be used by the child class


#-----------------------------------------------------------
class ConsoleSimulationDevice(NeopixelDevice):

    LED_CHAR = "\u25CF"

    def __init__(self, *, pixel_order:PixelOrder, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=None, **kwargs)
        self.is_simulated = True

    def write_to_device(self, buffer:NDArray[np.uint8]):
        print('', end='\r')
        for value in buffer:
            g, r, b = value[:3]
            w = value[3] if len(value) > 3 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
            print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='', flush=True) # print the LEDs


#-----------------------------------------------------------
class ConsoleSimulationSPIDevice(SPIDevice):

    LED_CHAR = "\u25CF"

    def __init__(self, *, pixel_order:PixelOrder, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=None, **kwargs)
        self.is_simulated = True

    def write_to_device(self, buffer:NDArray[np.uint8]) -> None:
        super().write_to_device(buffer)
        assert self._spi_buffer is not None
        buffer = self._spi_buffer.T.flatten()

        if self.neopixel is None:
            raise ValueError("Attribute `neopixel` is not set")

        def convert(bits: np.ndarray) -> int:
            # bits = np.ndarray[uint8, uitn8, uint8, uint8] = 4 double bits = 8 bits = 1 byte
            bit_values = {0xCC: 0b11, 0xC8: 0b10, 0x8C: 0b01, 0x88: 0b00} # SPI encodings {1byte: 2bits}
            result = 0
            for bit in bits:
                result = (result << 2) | bit_values[bit] # shift 2 bits and inject 2 new bits
            return result

        double_bits_per_pixel = 12 if len(self._pixel_order.name) == 3 else 16 # check if is GRB or GRBW...
        buffer = buffer.reshape([buffer.shape[0]//double_bits_per_pixel, double_bits_per_pixel]) # ...and reshape the buffer accordingly
        print('', end='\r')
        for bits in buffer: 
            g, r, b = convert(bits[0:4]), convert(bits[4:8]), convert(bits[8:12])
            w = convert(bits[12:16]) if bits.shape[0]>12 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
            print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='', flush=True) # print the LEDs


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
        self._num_lit_pixels: int = 0
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
        self._device = device
        self._pixel_buffer = np.zeros((self._num_pixels, device.pixel_order.num), dtype=np.float32)
        self._device.set_neopixel(self)
        return self


    def begin_update(self) -> 'Neopixel':
        """In auto write mode, **no** update will occour until end_update() and the counter is 0"""
        self._update_counter += 1
        return self


    def end_update(self, force_update:bool=False) -> 'Neopixel':
        self._update_counter -= 1

        if force_update or self._update_counter < 0:
            self._update_counter = 0

        if self._update_counter <= 0:
            self.auto_show()

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

        rgb_buffer = self.pixel_buffer[:]

        # Apply brightness and gamma correction
        rgb_buffer = np.clip(self._gamma_func(rgb_buffer * self._brightness), 0.0, 1.0)

        # calculate power consumption
        watts = self.watts_per_led if self.has_W else self.watts_per_led[:3]
        self._current_power = np.sum(watts * rgb_buffer)  

        # Power consumption limiter
        if (self._max_power > 1e-6) and (self._current_power > self._max_power):
            rgb_buffer *= self._max_power/self._current_power
            self._current_power = self._max_power

        # scale to [0, 255], and convert to uint8:
        rgb_buffer = np.clip(np.round(255 * rgb_buffer), 0, 255).astype(np.uint8)
        self._num_lit_pixels = int(np.count_nonzero(np.max(rgb_buffer, axis=1)))

        # rows are now pixels, columns are R,G,B,(W)

        if self.reversed:
            rgb_buffer = rgb_buffer[::-1]

        # Send data to device:
        self.device.write_bytes(rgb_buffer)


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

        if (value.ndim == 2) and (isinstance(index, int)):
            index = slice(index, index+value.shape[0])
        
        
        if value.shape[-1] == 1: # an array of single numbers is boradcasted to the white pixels only:
            if not self.has_W:
                #ignore if no white LED can be set
                return self
            
            self.pixel_buffer[index, -1] = value[:, 0]
            return self.auto_show()

        if (value.shape[-1] == 2) or (value.shape[-1] > 4):
            raise ValueError("Wrong number of values")

        rgb = (color_mode or self._color_mode).convert_to(value, ColorMode.RGB)

        if (value.shape[-1] > 3) and self.has_W:
            # Put back the white channel to the last dimension if RGBW
            rgb = np.concatenate([rgb, value[..., 3][..., np.newaxis]], axis=-1)
            self.pixel_buffer[index] = rgb
        else:
            self.pixel_buffer[index, :3] = rgb
    
        return self.auto_show()




    def next_value(self, value: PixelValue, color_mode: ColorMode | None = None) -> int:
        """Set the value for the next pixel in the iteration"""
        result = next(self)
        self.set_value(result, value=value, color_mode=color_mode)
        return result

    def fill(self, value: PixelValue, color_mode: ColorMode | None = None) -> 'Neopixel':
        """Fill all pixels with a given value"""
        return self.set_value(slice(None), value=value, color_mode=color_mode)

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


    def __iter__(self) -> 'Neopixel':
        return self

    def __next__(self) -> int:
        """ Iterate over the indices of the pixels. """
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        self._index += 1
        return self._index - 1


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
        return self.__len__()

    @property 
    def num_lit_pixels(self) -> int:
        """Get the number of lit pixels in the strip."""
        return self._num_lit_pixels

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
