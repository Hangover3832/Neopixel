from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Literal
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

    def __init__(self, *args, pixel_order:PixelOrder = PixelOrder.GRB, gamma_function: Callable | None = None, **kwargs) -> None:
        super().__init__()

        self._pixel_order: PixelOrder = pixel_order
        self._num_pixels: int = 0
        self._current_power: float = 0.0

        self.is_simulated: bool = False
        self.args: tuple = args
        self.kwargs: dict = kwargs
        self.watts_per_led: np.ndarray = np.array([0.081, 0.081, 0.08, 0.09])
        self.max_power: float = 0.0
        self.reversed: bool = False
        self.brightness: float = 1.0
        self.gamma_function = gamma_function or (lambda x: x)
        self.on_write_buffer: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None
        self.on_uint8: Callable[[NDArray[np.uint8]], NDArray[np.uint8]] | None = None
        self.on_uint32: Callable[[NDArray[np.uint32]], NDArray[np.uint32]] | None = None


    @abstractmethod
    def write_to_device(self, buffer:NDArray[np.float32], device_data: Any) -> Any:
        """Write the given buffer to the Neopixel device.
        Override this method in subclasses to implement device-specific writing logic.

        :param buffer: A 2D array of shape (num_pixels, num_channels=4) containing pixel values.
        :type buffer: np.ndarray[float32]
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

        #reduce the array to 2D if neccessary:
        buffer = buffer.reshape(-1, buffer.shape[-1])
        buffer = np.clip(self.brightness * self.gamma_function(buffer), 0.0, 1.0)

        if self.on_write_buffer is not None:
            buffer = self.on_write_buffer(buffer)

        watts = self.watts_per_led if self.pixel_order.has_W else np.append(self.watts_per_led[:3], 0.0)
        self._current_power = np.sum(watts * buffer).astype(float)

        # limit the power consumption
        if (self.max_power > 1e-6) and (self._current_power > self.max_power):
            buffer *= self.max_power/self._current_power
            self._current_power = self.max_power

        if self.reversed:
            buffer = buffer[::-1]

        #rearange the buffer to the correct pixel order and call write_to_device()
        buffer = buffer[:, [self.pixel_order.name.index(c) for c in 'RGBW' if c in self.pixel_order.name]]
        self.write_to_device(buffer, device_data)


    def _to_uint8(self, buffer:NDArray[np.float32]) -> NDArray[np.uint8]:
        """scale to [0, 255], and convert to 2D uint8"""
        if self.on_uint8 is not None:
            return self.on_uint8(np.clip(np.round(255. * buffer), 0., 255.).astype(np.uint8))
        else:
            return np.clip(np.round(255. * buffer), 0., 255.).astype(np.uint8)


    def _to_uint32(self, buffer:NDArray[np.float32]) -> NDArray[np.uint32]:
        """Convert the 2D buffer to 1D uint32."""
        b = self._to_uint8(buffer).astype(np.uint32)
        shifts = np.array([24, 16, 8, 0], dtype=np.uint32) if self.pixel_order.num == 4 else np.array([16, 8, 0], dtype=np.uint32)
        if self.on_uint32 is not None:
            return self.on_uint32(np.bitwise_or.reduce(b << shifts, axis=1, dtype=np.uint32))
        else:
            return np.bitwise_or.reduce(b << shifts, axis=1, dtype=np.uint32)


    def open_(self, neopixel: Neopixel) -> Any:
        """Open the Neopixel device and return device-specific data if needed.
        Override this method in subclasses to implement device-specific opening logic.
        Returns device-specific data that will be passed to _write_buffer() calls.
        """
        self._num_pixels = neopixel.num_pixels
        return None


    def close_(self) -> Any:
        """Close the Neopixel device. Override this method in subclasses to implement device-specific closing logic."""
        return None


    def __del__(self):
        """Ensure the device is closed upon deletion."""
        self.close_()


    @property
    def num_pixels(self) -> int:
        assert self._num_pixels > 0, "Attribute _num_pixels is not set, did you forget to open the device?"
        return self._num_pixels


    @property
    def pixel_order(self) -> PixelOrder:
        return self._pixel_order


#-----------------------------------------------------------
class NeopixelDevices:
    """This class contains a list of devices that can be attached to a Neopixel class"""
    def __init__(self, *, neopixel:Neopixel, **kwargs) -> None:
        self.devices: list[dict[str, Any]] = []
        #self.num_pixels = num_pixels


    def add_device(self, device:NeopixelDevice, neopixel:Neopixel) -> None:
        # Check if device not already in list:
        for dev in self.devices:
            if dev["device"] is device:
                return

        data = device.open_(neopixel=neopixel)
        self.devices.append({"device": device, "data": data})


    def close_(self):
        for dev in self.devices:
            dev["device"].close_()


    def remove_device(self, device:NeopixelDevice) -> None:
        for i, dev in enumerate(self.devices):
            if dev["device"] is device:
                device.close_()
                del self.devices[i]
                return
            

    def write_to_devices(self, buffer:NDArray[np.float32]) -> Any:
        for device in self.devices:
            device["device"]._write_buffer(buffer, device["data"])


    def __del__(self):
        for device in self.devices:
            device["device"].close_()


    @property
    def num_devices(self) -> int:
        return len(self.devices)

    
    @property
    def current_power(self) -> float:
        total_power = 0.0
        for device in self.devices:
            total_power += device["device"]._current_power
        return total_power

    @property
    def reversed(self):
        raise RuntimeError("Please read the `reversed` flag from the device.")
    
    @reversed.setter
    def reversed(self, value:bool):
        for device in self.devices:
            device["device"].reversed = value

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
            shape: int | tuple[int, ...] | list[int] | NDArray[np.int32],
            *,
            # gamma_func: Callable | None = None,
            color_mode: ColorMode = ColorMode.HSV,
            brightness: float = 1.0, 
            auto_write: bool = False,
            max_power: float = 0.0,
            **kwargs,
                ) -> None:

        # Public attributes
        self.kwargs:dict = kwargs
        self.auto_write:bool = auto_write

        # Private attributes
        shape = np.asarray(shape)
        self._num_pixels = np.prod(shape, dtype=int)
        self._color_mode: ColorMode = color_mode
        self._index: int = 0
        self._brightness: float = np.clip(brightness, 0., 1., dtype=float)
        self._update_counter: int = 0
        self._mini_screens: list[np.ndarray] = []
        shape = np.append(shape, 4) # add RGBW dimesion
        self._pixel_buffer: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
        self._devices = NeopixelDevices(neopixel=self)


    def to(self, device: NeopixelDevice) -> 'Neopixel':
        """Attach a NeopixelDevice to this Neopixel instance."""

        self._devices.add_device(device, self)
        return self.auto_show()


    def close_(self):
        self._devices.close_()


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

        self._mini_screens.append(config)
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
    

    def display_image(self, index:int, image:NDArray[np.float32], zigzag:Literal[0, 1, 2], transpose:bool=False) -> 'Neopixel':
        """
        Displays an image on a rectangular Neopixel area. The image will be scaled to the `size` parameter
        and then put on the stripe at `index` line by line (scanline).

        :param index: Start index the image should appear
        :type index: int
        :param image: image to display: np.asarray(PIL.Image.Image)
        :type image: np.NDArray[np.float32] (0..255)
        :param zigzag: Set `zigzag` > 0 if the Neopixel area is connected in a zigzag scheme:
            a value of 1 flips the odd, a 2 flips the even row numbers
        :type zigzag: int[0, 1, 2]
        :param transpose: Set `transpose` to True if the Neopixel area is arranged in columns
        :type transpose: bool
        :return: self
        :rtype: Neopixel
        """

        img = image.copy()

        if transpose:
            img = img.transpose(1, 0, 2)

        if zigzag == 1:
            img[1::2] = img[1::2][:, ::-1]
        elif zigzag == 2:
            img[0::2] = img[0::2][:, ::-1]

        # flatten the image data [h, w, rgb] uint8 to a 2D array [x, RGB] float32 of pixels:
        img = img.reshape(-1, img.shape[-1]).astype(np.float32) / 255.0

        return self.set_value(index, img, color_mode=ColorMode.RGB)


    def _write_buffer(self) -> None:
        """Write pixel data to the Neopixels device"""

        # Apply brightness and gamma correction
        rgb_buffer = np.clip(self._brightness * self.pixel_buffer, 0.0, 1.0)

        # Send data to device with device-specific data:
        self._devices.write_to_devices(rgb_buffer)       

        
    def __len__(self) -> int:
        """Get the number of pixels in the strip."""
        return self._num_pixels


    def set_temperature(self, index:PixelIndex, temperature:float, brightness:float = 1.0) -> 'Neopixel':
        """Set pixel value at index using an approximation for the black body heat radiation.
        The temperature ranges from [0.0 .. 1.0]"""

        self.pixel_buffer[index, :3] = np.clip(brightness * self._color_mode.kelvin_to_rgb(temperature), 0.0, 1.0)
        return self.auto_show()


    def __setitem__(self, index, value: PixelValue) -> None:
        """Sliced Neopixel write"""

        if isinstance(index, (int, slice)):
            index = (index,)

        if isinstance(value, (float, int)):
            # Set the white LED only
            self._pixel_buffer[*index, ..., 3] = float(value)
            return

        value = np.asarray(value)
        # Preserve the white LED if the value is [R, G, B] only
        self._pixel_buffer[*index, ..., :len(value)] = self.color_mode.convert_to(value, ColorMode.RGB)


    def __getitem__(self, index) -> NDArray[np.float32]:
        """Indexed or sliced Neopixel read"""
        rgb = self.pixel_buffer[index]
        result = ColorMode.RGB.convert_to(rgb, self._color_mode)

        # if self.device.pixel_order.has_W:
            # Put back the white channel to the last dimension if available
        #result = np.concatenate([result, rgb[..., 3][..., np.newaxis]], axis=-1)

        return result


    def set_value(self, index: PixelIndex, value: PixelValue, color_mode: ColorMode | None = None) -> 'Neopixel':
        """
        This is the core routine that writes pixel value at index to self.pixel_buffer.

        :param index: Pixel index number(s)
        :type index: int, slice, list, tuple
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

            self._pixel_buffer[index, 3] = np.clip(value, 0.0, 1.0).astype(float)
            return self.auto_show()

        if (value.shape[-1] == 2) or (value.shape[-1] > 4):
            raise ValueError("Wrong number of values")

        if (value.ndim == 2) and (isinstance(index, int)):
            # If the value contains an array with multiple pixel values, we try to broadcast them to a slice in the buffer
            index = slice(index, index+value.shape[0])

        if value.shape[-1] == 1: # an array of single numbers is boradcasted to the white pixels only:
            #if not self.has_W:
                # ignore if no white LED is available
            #    return self

            self._pixel_buffer[index, 3] = value[:, 0]
            return self.auto_show()

        # The internal pixel buffer is held in ColorMode.RGB
        rgb = (color_mode or self._color_mode).convert_to(value, ColorMode.RGB)

        '''
        if (value.shape[-1] > 3): # and self.has_W:
            # Put back the white channel to the last dimension if RGBW
            rgb = np.concatenate([rgb, value[..., 3][..., np.newaxis]], axis=-1)
            self._pixel_buffer[index, ...] = rgb
        else:
        '''

        self._pixel_buffer[index, :value.shape[-1]] = rgb
    
        return self.auto_show()


    def next_value(self, value: PixelValue, color_mode: ColorMode | None = None) -> 'Neopixel':
        """Set the value for the next pixel in the iteration"""
        return self.set_value(next(self), value=value, color_mode=color_mode)

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


    def __iter__(self) -> 'Neopixel':
        return self

    def __next__(self) -> int:
        """ Iterate over the indices of the pixels. """
        if self._index >= self.num_pixels:
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
        return PixelOrder.GRBW.blank

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
        self._brightness = np.clip(value, 0.0, 1.0, dtype=float)
        self.auto_show()

    @property
    def num_pixels(self) -> int:
        """Get the number of pixels in the strip."""
        return self._num_pixels

    @property
    def power_consumption(self) -> float:
        """Returns the total power consumption [0..1]"""
        return self._devices.current_power

    @property
    def reversed(self):
        return self._devices.reversed
    
    @reversed.setter
    def reversed(self, value:bool):
        self._devices.reversed = value

    @property
    def pixel_buffer(self) -> np.ndarray:
        return self._pixel_buffer

    @property
    def devices(self) -> NeopixelDevices:
        return self._devices
