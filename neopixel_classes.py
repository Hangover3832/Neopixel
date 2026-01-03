from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from colors import ColorMode, PixelOrder, G
from PIL import Image


PixelIndex = int | list[int] | tuple[int, ...] | slice
PixelValue = NDArray[np.float32] | list[float] | tuple[float, ...] | float | int

#-----------------------------------------------------------
class NeopixelDevice(ABC):
    """
    Abstract base class for Neopixel devices.
    This class defines the interface for Neopixel devices.
    :param pixel_order: The pixel order used by the Neopixel device.
    :type pixel_order: PixelOrder
    """

    def __init__(self, *args, pixel_order:PixelOrder = PixelOrder.GRB, **kwargs) -> None:
        super().__init__()
        # self._neopixel: Neopixel | None = None
        self._pixel_order: PixelOrder = pixel_order
        self.is_simulated: bool = False
        self.args: tuple = args
        self.kwargs: dict = kwargs
        # self._is_open: bool = False
        self._num_pixels: int = 0


    @abstractmethod
    def write_to_device(self, buffer:NDArray[np.float32], device_data: Any) -> Any:
        """Write the given buffer to the Neopixel device.
        Override this method in subclasses to implement device-specific writing logic.

        :param buffer: A 2D array of shape (num_pixels, num_channels) containing pixel values.
        :type buffer: np.ndarray[[float, ...], [float, ...]]
        :param device_data: Optional device-specific data from the Neopixel instance.
        :type device_data: Any
        """
        raise NotImplementedError("Subclasses must implement write_to_device() method.")


    def _write_buffer(self, buffer:NDArray[np.float32], device_data: Any) -> Any:
        """
        Write the given buffer to the Neopixel device after rearranging it to match the pixel order.

        :param buffer: A 2D array of shape (num_pixels, num_channels) containing pixel values.
        :type buffer: np.ndarray[[float, ...], [float, ...]]
        :param device_data: Optional device-specific data returned from open_().
        :type device_data: Any
        :returns: The result of the write_to_device() method.
        """

        #rearange the buffer to the correct pixel order and call write_to_device()
        buffer = buffer[:, [self.pixel_order.name.index(c) for c in 'RGBW' if c in self.pixel_order.name]]
        self.write_to_device(buffer, device_data)


    def _to_uint8(self, buffer:NDArray[np.float32]) -> NDArray[np.uint8]:
        """scale to [0, 255], and convert to uint8 array"""
        return np.clip(np.round(255. * buffer), 0., 255.).astype(np.uint8)


    def _to_uint32(self, buffer:NDArray[np.float32]) -> NDArray[np.uint32]:
        """Convert the 2D buffer to 1D uint32."""
        b = self._to_uint8(buffer).astype(np.uint32)
        shifts = np.array([24, 16, 8, 0], dtype=np.uint32) if self.pixel_order.num == 4 else np.array([16, 8, 0], dtype=np.uint32)
        return np.bitwise_or.reduce(b << shifts, axis=1, dtype=np.uint32)


    def open_(self, num_pixels:int) -> Any:
        """Open the Neopixel device and return device-specific data if needed.
        Override this method in subclasses to implement device-specific opening logic.
        Returns device-specific data that will be passed to _write_buffer() calls.
        """
        #result = not self._is_open
        #self._is_open = True
        #return result
        self._num_pixels = num_pixels
        return None


    def close_(self) -> Any:
        """Close the Neopixel device. Override this method in subclasses to implement device-specific closing logic."""
        #result = self._is_open
        #self._is_open = False
        #return result
        return None


    def __del__(self):
        """Ensure the device is closed upon deletion."""
        #if self._is_open:
        self.close_()

    '''
    def set_num_pixels(self, num:int) -> None:
        """Set the number of pixels for the Neopixel device.
        Override this method in subclasses if needed in case the number of pixels changes."""
        if num <= 0:
            raise ValueError("Error: Number of pixels must be > 0")
        self._num_pixels = num
    '''

    @property
    def num_pixels(self) -> int:
        assert self._num_pixels > 0, "Attibute _num_pixel is not set."
        return self._num_pixels

    '''
    @num_pixels.setter
    def num_pixels(self, num:int) -> None:
        self.set_num_pixels(num)
    '''


    @property
    def pixel_order(self) -> PixelOrder:
        return self._pixel_order


#-----------------------------------------------------------
class NeopixelDevices:
    """This class contains a list of devices that can be attached to a Neopixel class"""
    def __init__(self, *, num_pixels:int, **kwargs) -> None:
        self.devices: list[dict[str, Any]] = []
        self.num_pixels = num_pixels


    def add_device(self, device:NeopixelDevice) -> None:
        # Check if device not already in list:
        for dev in self.devices:
            if dev["device"] is device:
                return

        data = device.open_(self.num_pixels)
        self.devices.append({"device": device, "data": data})


    def write_to_devices(self, buffer:NDArray[np.float32]) -> Any:
        for device in self.devices:
            device["device"]._write_buffer(buffer, device["data"])


#-----------------------------------------------------------
class Neopixel:
    """
    Main Neopixel class that manages pixel data and interfaces with a NeopixelDevice.

    :param num_pixels: Number of pixels in the Neopixel strip.
    :type num_pixels: int
    :param gamma_func: A function for gamma correction. Defaults to None.
    :type gamma_func: Callable | None
    :param color_mode: The color mode used for pixel values. Defaults to ColorMode.HSV.
    :type color_mode: ColorMode
    :param brightness: Brightness level (0.0 to 1.0). Defaults to 1.0.
    :type brightness: float
    :param auto_write: If True, changes are automatically written to the device. Defaults to False.
    :type auto_write: bool
    :param max_power: Maximum power consumption limit. Defaults to 0.0 (no limit).
    :type max_power: float
    """

    def __init__(self, 
            num_pixels:int,
            *,
            gamma_func: Callable | None = None,
            color_mode: ColorMode = ColorMode.HSV,
            brightness: float = 1.0, 
            auto_write: bool = False,
            max_power: float = 0.0,
            **kwargs,
                ) -> None:

        # Public attributes
        self.kwargs:dict = kwargs
        self.reversed: bool = False
        self.auto_write:bool = auto_write
        self.watts_per_led: np.ndarray = np.array([0.081, 0.081, 0.08, 0.09])

        # Private attributes
        self._num_pixels = num_pixels
        self._color_mode: ColorMode = color_mode
        self._current_power: float = 0.0
        self._max_power: float = max_power
        self._index: int = 0
        self._brightness: float = float(np.clip(brightness, 0., 1.))
        self._gamma_func: Callable = gamma_func or (lambda x: x)
        self._update_counter: int = 0
        self._mini_screens: list[np.ndarray] = []
        self._pixel_buffer: NDArray[np.float32] = np.zeros((self._num_pixels, 4), dtype=np.float32)

        self._device: NeopixelDevice | None = None
        self._device_data: Any = None

        # self._devices = NeopixelDevices(num_pixels=self._num_pixels)


    def to(self, device: NeopixelDevice) -> 'Neopixel':
        """Attach a NeopixelDevice to this Neopixel instance."""

        # self._devices.add_device(device)

        # Close the old device if switching devices
        if self._device is not None and self._device is not device:
            self._device.close_()

        self._device = device

        # Capture device-specific data returned from open_
        self._device_data = self._device.open_(self._num_pixels)

        return self.auto_show()


    def close_(self):
        if self._device is not None:
            self._device.close_()


    def begin_update(self) -> 'Neopixel':
        """In auto write mode, **no** update will occour until end_update() and the counter is 0"""
        self._update_counter += 1
        return self


    def end_update(self, reset:bool=False) -> 'Neopixel':
        """Decrement the update counter. If it reaches 0, an automatic show() will be performed"""
        if self._update_counter > 0:
            self._update_counter -= 1

        if reset or self._update_counter <= 0:
            self._update_counter = 0
            return self.auto_show()

        return self


    def add_virtual_screen(self, config: NDArray[np.uint16]) -> int:
        """
        Add a virtual 2 dimensional screen area to the Neopixel stripe.

        :param config: A 2-dimensional array that contains all the pixel indices that bild up the screen,
        mapping from the top left to the bottom right line by line on the virtual screen.
        :type config: np.ndarray[[int, ...], ...]
        :returns: The index number of the newly created screen.
        :rtype: int
        """

        self._mini_screens.append(config.astype(np.int16))
        return len(self._mini_screens)-1


    def virtual_screen_data(self, screen_index: int, data: NDArray[np.float32], color_mode: ColorMode  | None = None) -> 'Neopixel':
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

        assert data.shape[:2] == self._mini_screens[screen_index].shape
        indices = self._mini_screens[screen_index].flatten()
        data = data.reshape(-1, data.shape[2]).squeeze()

        self.begin_update()
        # no auto write in this loop
        for d, i in enumerate(indices):
            self.set_value(i, data[d], color_mode=color_mode)

        return self.end_update()
    

    def display_image(self, index:int, image:Image.Image, size:tuple) -> 'Neopixel':
        """
        Displays a PIL image on the Neopixel. The image will be scaled to the `size` parameter.
        The image will be put on the stripe at `index` line by line (scanline).

        :param index: Start index the image should appear
        :type index: int
        :param image: image to display, the image will be resized to the `size` parameter.
        :type image: Pil.Image.Image
        :param size: (width, height)
        :type size: tuple
        :return: self
        :rtype: Neopixel
        """

        w, h = size
        img = np.asarray(image.convert('RGB').resize((w, h), resample=Image.Resampling.HAMMING))

        # flatten the image data [h, w, rgb] uint8 to a 2D array [x, RGB] float32 of pixels:
        img = img.reshape(-1, img.shape[-1])/255.

        return self.set_value(index, img, color_mode=ColorMode.RGB)


    def _write_buffer(self) -> None:
        """Write pixel data to the Neopixels device"""
        assert self._pixel_buffer is not None, "Pixel buffer is not initialized."

        # Apply brightness and gamma correction
        if self._gamma_func is not None:
            rgb_buffer = self._gamma_func(self.pixel_buffer.copy())
        else:
            rgb_buffer = self.pixel_buffer.copy()

        rgb_buffer = np.clip(self._brightness * rgb_buffer, 0.0, 1.0)

        # calculate power consumption
        watts = self.watts_per_led if self.device.pixel_order.has_W else np.append(self.watts_per_led[:3], 0.0)
        self._current_power = np.sum(watts * rgb_buffer).astype(float)

        # Power consumption limiter
        if (self._max_power > 1e-6) and (self._current_power > self._max_power):
            rgb_buffer *= self._max_power/self._current_power
            self._current_power = self._max_power

        if self.reversed:
            rgb_buffer = rgb_buffer[::-1]

        # Send data to device with device-specific data:
        self.device._write_buffer(rgb_buffer, self._device_data)


    def __setitem__(self, index: int | slice, value: PixelValue) -> None:
        """Indexed or sliced Neopixel write"""
        self.set_value(index, value)


    def __getitem__(self, index: int | slice) -> NDArray[np.float32]:
        """Indexed or sliced Neopixel read"""
        rgb = self.pixel_buffer[index]
        result = ColorMode.RGB.convert_to(rgb, self._color_mode)

        if self.device.pixel_order.has_W:
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
        This is the core routine that writes pixel value at index to self.pixel_buffer.
        
        :param index: Pixel index number(s)
        :type index: int or slice
        :param value: Pixel value. 
            If value is a number, it affects the White LED only in a RGBW stripe. A non RGBW stripe, it will ignored in this case.
        :type value: number or array like
        :param color_mode: The color mode of the provided value. If None, the current color mode of the Neopixel instance is used.
        :type color_mode: ColorMode | None
        :returns: The current instance of Neopixel.
        :rtype: Neopixel
        """

        value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)

        if value.ndim == 0:# a single number applies to the white LED only

            #if not self.has_W:
                #ignore if no white LED can be set
            #    return self

            self.pixel_buffer[index, 3] = float(np.clip(value, 0.0, 1.0))
            return self.auto_show() if self.device.pixel_order.has_W else self

        if (value.shape[-1] == 2) or (value.shape[-1] > 4):
            raise ValueError("Wrong number of values")

        if (value.ndim == 2) and (isinstance(index, int)):
            # If the value contains an array with multiple pixel values, we try to broadcast them to a slice in the buffer
            index = slice(index, index+value.shape[0])

        if value.shape[-1] == 1: # an array of single numbers is boradcasted to the white pixels only:
            #if not self.has_W:
                # ignore if no white LED is available
            #    return self

            self.pixel_buffer[index, 3] = value[:, 0]
            return self.auto_show()

        # The internal pixel buffer is held in ColorMode.RGB
        rgb = (color_mode or self._color_mode).convert_to(value, ColorMode.RGB)

        if (value.shape[-1] > 3): # and self.has_W:
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
        """In auto write mode, show the current pixel buffer if the update counter is 0"""
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
                        index:int=0, 
                        count:int=0, 
                        color_mode:ColorMode | None = None) -> 'Neopixel':
        
        """
        Create a color gradient from `from_value` to `to_value` starting at `index` for `count` pixels.
        If `count` is 0, the gradient will fill up to the end of the pixel strip.

        :param from_value: The starting color value of the gradient.
        :type from_value: PixelValue
        :param to_value: The ending color value of the gradient.
        :type to_value: PixelValue
        :param index: The starting index for the gradient. Defaults to 0.
        :type index: int
        :param count: The number of pixels to fill with the gradient. If 0, fills to the end of the strip. Defaults to 0.
        :type count: int
        :param color_mode: The color mode of the provided values. If None, the current color mode of the Neopixel instance is used.
        :type color_mode: ColorMode | None
        :returns: The current instance of Neopixel.
        :rtype: Neopixel
        """

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

            #if self.has_W:
            if from_value.shape[0] > 3:
                rgb.append(np.linspace(from_value[3], to_value[3], count))

            gradient = np.stack(rgb, axis=1, dtype=np.float32)
        else: # W only value array
            gradient = np.linspace(from_value, to_value, count)[:, np.newaxis].astype(np.float32)

        return self.set_value(index, gradient[:self.num_pixels-index], color_mode=color_mode)


    # __dunder__ methods =============================
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
        self._pixel_buffer =  np.clip(self.pixel_buffer + value, 0., 1., dtype=np.float32)
        return self.auto_show()

    def __imul__(self, value: np.ndarray | float) -> Neopixel:
        """Multiply value with the pixel buffer in RGB space, e.g. neo *= 0.9"""
        self._pixel_buffer = np.clip(self.pixel_buffer * value, 0., 1., dtype=np.float32)
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
    
    '''
    @property
    def has_W(self) -> bool:
        """Check if the pixel type has a white channel."""
        return self.device.pixel_order.num == 4 #self.pixel_buffer.shape[-1] > 3
    '''

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
