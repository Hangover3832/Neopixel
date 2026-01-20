from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from time import sleep
from typing import Any, Callable, Literal
import numpy as np
from numpy.typing import NDArray
from colors import ColorMode, PixelOrder, G
from PIL import Image
import warnings


# PixelIndex = int | list[int] | tuple[int, ...] | slice
PixelValue = NDArray[np.float32] | list[float] | tuple[float, ...] | float | int
Callable_uint8 = Callable[[NDArray[np.uint8]], NDArray[np.uint8]]
Callable_uint32 = Callable[[NDArray[np.uint32]], NDArray[np.uint32]]
Callable_float32 = Callable[[NDArray[np.float32]], NDArray[np.float32]]

#-----------------------------------------------------------
class SliceHelper:
    """Singleton class to help with slice operations in Neopixel class."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __getitem__(self, index: Any) -> Any:
        return index
    
    def pre_slice(self, n: int) -> list[slice]:
        """
        Generate a list of slice(None) for pre-slicing.

        :param n: Number of preceding dimesions to generate.
        :type n: int
        """
        result = [slice(None) for _ in range(n)]
        return result

slice_helper = SliceHelper()


#-----------------------------------------------------------
class NeopixelDevice(ABC):
    """
    Abstract base class for Neopixel devices.
    This class defines the interface for Neopixel devices.
    :param pixel_order: The pixel order used by the Neopixel device.
    :type pixel_order: PixelOrder
    :param gamma_function: A function for gamma correction. Defaults to None.
    :type gamma_function: Callable | None
    """

    def __init__(self, *args, pixel_order:PixelOrder = PixelOrder.GRB, gamma_function: Callable | None = None, **kwargs) -> None:
        super().__init__()

        self._pixel_order: PixelOrder = pixel_order
        #self._num_pixels: int = 0
        self.neopixel: Neopixel | None = None
        self._current_power: float = 0.0

        self.is_simulated: bool = False
        self.args: tuple = args
        self.kwargs: dict = kwargs
        self.watts_per_led: np.ndarray = np.array([0.081, 0.081, 0.08, 0.09])
        self.max_power: float = 0.0
        self.reversed: bool = False
        self.brightness: float = 1.0
        self.gamma_function: Callable | None = gamma_function
        self.on_write_buffer: Callable_float32 | None = None
        self.on_write_to_device: Callable_float32 | None = None
        self.on_uint8: Callable_uint8 | None = None
        self.on_uint32: Callable_uint32 | None = None
        self._num_lit_pixels: int = 0


    @abstractmethod
    def write_to_device(self, buffer:NDArray[np.float32], device_data: Any) -> Any:
        """Write the given buffer to the Neopixel device.
        Override this method in subclasses to implement device-specific writing logic.

        :param buffer: A 2D array of shape (num_pixels, num_channels=4) containing pixel values.
        :type buffer: np.ndarray[float32]
        :param device_data: Optional device-specific data from the Neopixel instance.
        :type device_data: Any
        """
        raise RuntimeError("Abstract method error: Please override `write_to_device()` in the child class.")


    def _write_buffer(self, buffer:NDArray[np.float32], device_data: Any) -> Any:
        """
        Write the given buffer to the Neopixel device after rearranging it to match the pixel order.

        :param buffer: The buffer in tis initialized shape containing pixel values.
        :type buffer: np.ndarray[[float, ...], [float, ...]]
        :param device_data: Optional device-specific data returned from open_().
        :type device_data: Any
        :returns: The result of the write_to_device() method.
        """

        if self.on_write_buffer is not None:
            buffer = self.on_write_buffer(buffer)

        #reduce the array to 2D if neccessary:
        buffer = buffer.reshape(-1, buffer.shape[-1])
        if self.gamma_function is not None:
            buffer = self.gamma_function(buffer)
        buffer = np.clip(self.brightness * buffer, 0.0, 1.0)

        watts = self.watts_per_led if self.pixel_order.has_W else np.append(self.watts_per_led[:3], 0.0)
        self._current_power = np.sum(watts * buffer).astype(float)

        # limit the power consumption
        if (self.max_power > 1e-6) and (self._current_power > self.max_power):
            buffer *= self.max_power/self._current_power
            self._current_power = self.max_power

        if self.reversed:
            # The reversion happens after conversion to 2D
            buffer = buffer[::-1]

        #rearange the buffer to the correct pixel order and call write_to_device()
        buffer = buffer[:, [self.pixel_order.name.index(c) for c in 'RGBW' if c in self.pixel_order.name]]

        if self.on_write_to_device is not None:
            buffer = self.on_write_to_device(buffer)

        self.write_to_device(buffer, device_data)


    def _to_uint8(self, buffer:NDArray[np.float32]) -> NDArray[np.uint8]:
        """scale to [0, 255], and convert to uint8"""
        result = np.clip(np.round(255. * buffer), 0., 255.).astype(np.uint8)

        # count the total amount of element in the array that are non zero
        self._num_lit_pixels = np.count_nonzero(result)

        if self.on_uint8 is not None:
            return self.on_uint8(result)
        else:
            return result


    def _to_uint32(self, buffer:NDArray[np.float32]) -> NDArray[np.uint32]:
        """Convert the 2D buffer to 1D uint32."""
        assert buffer.ndim == 2, "Buffer must be a 2D array"
        result = self._to_uint8(buffer).astype(np.uint32)
        shifts = np.array([24, 16, 8, 0], dtype=np.uint32) if self.pixel_order.num == 4 else np.array([16, 8, 0], dtype=np.uint32)
        result = np.bitwise_or.reduce(result << shifts, axis=1, dtype=np.uint32)
        if self.on_uint32 is not None:
            return self.on_uint32(result)
        else:
            return result


    def open_(self, neopixel: Neopixel) -> Any:
        """Open the Neopixel device and return device-specific data if needed.
        Override this method in subclasses to implement device-specific opening logic.
        Returns device-specific data that will be passed to _write_buffer() calls.
        """
        self.neopixel = neopixel
        return None


    def close_(self) -> Any:
        """Close the Neopixel device. Override this method in subclasses to implement device-specific closing logic."""
        self.on_write_buffer = None
        self.on_write_to_device = None
        self.on_uint8 = None
        self.on_uint32 = None
        return None


    def __del__(self):
        """Ensure the device is closed upon deletion."""
        self.close_()


    @property
    def num_pixels(self) -> int:
        assert self.neopixel is not None, "Attribute neopixel is not set, did you forget to open the device?"
        return self.neopixel.num_pixels

    @property
    def num_lit_pixels(self) -> int:
        return self._num_lit_pixels

    @property
    def pixel_order(self) -> PixelOrder:
        return self._pixel_order


#-----------------------------------------------------------
DeviceAndData = dict[Literal['device', 'data'], NeopixelDevice | Any]
DeviceList = list[DeviceAndData]


class NeopixelDevices:
    """This class contains a list of devices that can be attached to a Neopixel class"""
    def __init__(self, *, neopixel:Neopixel, **kwargs) -> None:
        self.devices: DeviceList = []
        #self.num_pixels = num_pixels


    def __getitem__(self, index: int) -> DeviceAndData:
        return self.devices[index]


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
        """
        Write a buffer of float32 values to all registered devices.
        
        Iterates through each device in the devices collection and writes
        the provided buffer data using each device's _write_buffer method,
        passing along the device-specific data.
        
        Args:
            buffer (NDArray[np.float32]): A NumPy array of float32 values
                to be written to the devices.
        
        Returns:
            Any: The return value depends on the device implementations.
        """
        for device in self.devices:
            device['device']._write_buffer(buffer, device['data'])


    @property
    def num_devices(self) -> int:
        return len(self.devices)
    
    @property
    def current_power(self) -> float:
        total_power = 0.0
        for device in self.devices:
            total_power += device['device']._current_power
        return total_power

    @property
    def reversed(self):
        raise RuntimeError("Please read the `reversed` flag from the specific device.")
    
    @reversed.setter
    def reversed(self, value:bool):
        for device in self.devices:
            device["device"].reversed = value

    @property
    def num_lit_pixels(self) -> int:
        result = 0
        for dev in self.devices:
            result += dev["device"].num_lit_pixels
        return result

#-----------------------------------------------------------
class Neopixel:
    """
    Main Neopixel class that manages pixel data and interfaces with a NeopixelDevice.

    :param shape: The shape of the Neopixel strip (number of pixels or multi-dimensional shape).
    :type shape: int | tuple[int, ...] | list[int] | NDArray[np.int32]
    :param color_mode: The color mode used for pixel values (e.g., RGB, HSV).
    :type color_mode: ColorMode
    :param brightness: The brightness level of the Neopixels (0.0 to 1.0).
    :type brightness: float
    :param auto_write: If True, automatically updates the Neopixels on pixel changes.
    :type auto_write: bool

    Options can be passed to NeopixelDevice instances via kwargs.
    """

    def __init__(self, 
            shape: int | tuple[int, ...] | list[int] | NDArray[np.int32],
            *,
            color_mode: ColorMode = ColorMode.HSV,
            brightness: float = 1.0, 
            auto_write: bool = False,
            **kwargs,
                ) -> None:


        # Public attributes
        self.kwargs:dict = kwargs
        self.auto_write:bool = auto_write

        # Private attributes
        self._shape = np.asarray(shape)
        self._num_pixels = np.prod(self._shape, dtype=int)
        self._color_mode: ColorMode = color_mode
        self._temp_color_mode: ColorMode | None = None
        self._index: int = 0
        self._brightness: float = np.clip(brightness, 0., 1., dtype=float)
        self._update_counter: int = 0
        self._mini_screens: list[np.ndarray] = []
        self._pixel_buffer: NDArray[np.float32] = np.zeros(np.append(self._shape, 4), dtype=np.float32)
        self._devices = NeopixelDevices(neopixel=self)
        self._transpose: tuple[int, ...] | None = None
        self._flatten: bool = False
        self.roll = self.Roll(self)


    class Roll:
        """Neopixel buffer rolling helper class allows complex sliced rolling operations"""
        def __init__(self, neopixel:Neopixel) -> None:
            self.neo = neopixel

            # default values for roll:
            self.shift: int = 1
            self.axis: int = 0


        def __getitem__(self, index):
            """Allows rolling with slice oprations, e.g. neopixel.roll[:]"""
            self.neo._transpose = None
            if self.shift != 0:
                self.neo.pixel_buffer[index] = np.roll(self.neo.pixel_buffer[index], shift=self.shift, axis=self.axis)
                self.neo.auto_show()

            self.shift = 1
            self.axis = 0
            return self.neo


        def __setitem__(self, index, value):
            """Assign a value to the rolled-in pixels, e.g. neopixel.roll[:] = pixel_value"""
            self.neo._transpose = None
            if self.shift != 0:
                buffer = np.roll(self.neo.pixel_buffer[index], shift=self.shift, axis=self.axis)

                if isinstance(value, (int, float)) or (value := np.asarray(value)).shape[-1] == 1:
                    # a single number applies to the white LED's at index [3] olny
                    shape = slice_helper[3:4]
                else:
                    # apply RGB(W)
                    shape = slice_helper[:value.shape[-1]]

                #apply `value` to the pixels that are shifted in, depending on the shift direction
                shift = slice_helper[:self.shift] if self.shift > 0 else slice_helper[self.shift:]

                buffer[*slice_helper.pre_slice(self.axis), shift, ..., shape] = self.neo.color_mode.convert_to(value, ColorMode.RGB)
                self.neo.pixel_buffer[index] = buffer

                self.neo.auto_show()
            
            self.shift = 1
            self.axis = 0
            return self.neo


        def __call__(self, shift=None, axis=None):
            """Set number of shits and/or the axis to roll along.
            For example `neopixel.RGB.roll(shift=3, axis=1)[1] = (1.0, 0.0, 0.0)`,
            means: `roll the [1st] dimesion along axis [1] by 3 and set the rolled-in pixels to red`"""
            self.neo._transpose = None

            self.shift = shift if shift is not None else 1
            self.axis = axis if axis is not None else 0

            if shift is None and axis is None:
                self[:]
                return self.neo

            return self


    def to(self, device: NeopixelDevice) -> 'Neopixel':
        """Attach a NeopixelDevice to this Neopixel instance."""
        self._devices.add_device(device, self)
        return self


    def close_(self):
        self._devices.close_()


    def begin_update(self) -> 'Neopixel':
        """In auto write mode, **no** update will occour until end_update() and the update counter is 0"""
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


    def _write_buffer(self) -> None:
        """Write pixel data to the Neopixels device"""

        #reset the temporary operations
        self._transpose = None
        self._flatten = False
        self._temp_color_mode = None

        # Apply brightness
        rgb_buffer = np.clip(self._brightness * self.pixel_buffer, 0.0, 1.0)

        # Send data to devices:
        self._devices.write_to_devices(rgb_buffer)       


    def __len__(self) -> int:
        """Get the number of pixels in the strip."""
        return self._num_pixels


    '''
    def set_temperature(self, index, temperature:float, brightness:float = 1.0) -> 'Neopixel':
        """Set pixel value at index using an approximation for the black body heat radiation.
        The temperature ranges from [0.0 .. 1.0]"""

        self.pixel_buffer[index] = np.clip(brightness * self._color_mode.kelvin_to_rgb(temperature), 0.0, 1.0)
        return self.auto_show()
    '''


    def __setitem__(self, index, value: PixelValue | Image.Image):
        """Indexed or sliced Neopixel write"""

        if self._transpose is None:
            buf = self.pixel_buffer[index] 
        else: 
            buf = self._pixel_buffer.transpose(self._transpose)[index]
            self._transpose = None

        if isinstance(value, (int, float)):
            # a single number applies to the white LED'a at index[..., 3]
            value = np.asarray((value,))

        elif isinstance(value, Image.Image):
            assert buf.ndim == 3, "Expected slice to be 3-dimensional (H, W, RGB) to display an image"
            h, w = buf.shape[0], buf.shape[1]
            # rescale the image to fit into the slice
            value = np.asarray(
                value.resize((w,h), resample=Image.Resampling.NEAREST),
                dtype=np.float32) / 255.
        else :
            value = np.asarray(value, dtype=np.float32)

        if self._flatten:
            # write a flattened input array
            self._flatten = False
            value = value.reshape(*buf.shape[:-1], value.shape[-1])

        if value.shape[-1] == 1:
            # white LED only
            buf[..., 3:4] = value 
        else:
            # and for the color LED's
            buf[..., :value.shape[-1]] = self.color_mode.convert_to(value, ColorMode.RGB)

        return self.auto_show()


    def __getitem__(self, index) -> NDArray[np.float32]:
        """Indexed or sliced Neopixel read"""
        result = ColorMode.RGB.convert_to(
                        self.pixel_buffer[index], 
                        self.color_mode)

        return np.asarray(result)


    def decay(self, factor: float = 0.95, delay: float = 0.01) -> 'Neopixel':
        """Decay to darkness"""
        auto_write = self.auto_write
        self.auto_write = True
        while self.num_lit_pixels > 0:
            self *= factor
            sleep(delay)
        self.auto_write = auto_write
        return self


    def fill(self, value: PixelValue) -> 'Neopixel':
        """Fill all pixels with a given value"""
        self[:] = value
        return self #.auto_show()


    def clear(self) -> 'Neopixel':
        """Clear all pixels by setting them to black."""
        self[:] = self.blank
        return self #.auto_show()


    def auto_show(self) -> 'Neopixel':
        """In auto write mode, show the current pixel buffer if the update counter is 0"""
        return self.show() if self.auto_write and self._update_counter <= 0 else self


    def show(self) -> 'Neopixel':
        """Update the NeoPixels with the current pixel buffer."""
        self._write_buffer()
        return self


    def create_gradient(self, 
                        from_value:PixelValue, 
                        to_value:PixelValue, 
                        count:int) -> NDArray[np.float32]:
        
        """
        Create a color gradient from `from_value` to `to_value` for `count` pixels.

        :param from_value: The starting color value of the gradient.
        :type from_value: PixelValue
        :param to_value: The ending color value of the gradient.
        :type to_value: PixelValue
        :param count: The number of pixels to fill with the gradient. If 0, fills to the end of the strip. Defaults to 0.
        :type count: int
        :returns: The color gradient array
        :rtype: NDArray[np.float32]
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

            if from_value.shape[0] > 3:
                rgb.append(np.linspace(from_value[3], to_value[3], count))

            return np.stack(rgb, axis=1, dtype=np.float32)

        else: # W only value array
            return np.linspace(from_value, to_value, count)[:, np.newaxis].astype(np.float32)


    def t(self, *axis) -> 'Neopixel':
        """Set transpose for the next get/set operation."""
        self._transpose = axis
        return self


    # __dunder__ methods =============================
    def __call__(self) -> 'Neopixel':
        """
        Calls `show()` that updates the stripe.
        """
        return self.show()


    def __iadd__(self, value: NDArray[np.float32] | float) -> 'Neopixel':
        """Add value to the pixel buffer in RGB space, e.g. pixels += 0.1"""
        self._pixel_buffer = np.clip(self.pixel_buffer + value, 0., 1., dtype=np.float32)
        return self.auto_show()

    def __imul__(self, value: NDArray[np.float32] | float) -> Neopixel:
        """Multiply value with the pixel buffer in RGB space, e.g. neo *= 0.9"""
        self._pixel_buffer = np.clip(self.pixel_buffer * value, 0., 1., dtype=np.float32)
        return self.auto_show()

    def __ilshift__(self, amount: int) -> 'Neopixel':
        """roll to the left by amount, e.g. `pixels <<= 1`"""
        self.roll(-int(abs(amount)), axis=0)[:]
        return self
    
    def __irshift__(self, amount: int) -> 'Neopixel':
        """roll to the right by amount, e.g. `pixels >>= 1`"""
        self.roll(int(abs(amount)), axis=0)[:]
        return self

    def __invert__(self)-> 'Neopixel':
        """Invert all colors of all pixels, e.g. `~pixels` """
        self._pixel_buffer = 1.0 - self.pixel_buffer
        return self.auto_show()


    # properties =============================
    @property
    def blank(self) -> np.ndarray:
        """Get a black color value"""
        return PixelOrder.GRBW.blank

    @property 
    def color_mode(self) -> ColorMode:
        """Get the current color mode."""
        if self._temp_color_mode is None:
            result = self._color_mode
        else:
            result = self._temp_color_mode
            self._temp_color_mode = None

        return result


    @color_mode.setter
    def color_mode(self, new_mode: ColorMode) -> None:
        """Set a new color mode."""
        self._temp_color_mode = None
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
    
    @property
    def num_lit_pixels(self) -> int:
        return self.devices.num_lit_pixels

    @property
    def RGB(self) -> Neopixel:
        """Set temporary color mode to RGB for the next set operation."""
        self._temp_color_mode = ColorMode.RGB
        return self

    @property
    def HSV(self) -> Neopixel:
        """Set temporary color mode to HSV for the next set operation."""
        self._temp_color_mode = ColorMode.HSV
        return self

    @property
    def T(self):
        """ Set transpose for the next set operation, swap axis [-2] and [-3]
        For custom transpose operation, use the t() method"""
        new_order = list(range(self.pixel_buffer.ndim))
        new_order[-3], new_order[-2] = new_order[-2], new_order[-3]  # Swap -3 and -2
        self._transpose = tuple(new_order)
        return self

    @property
    def F(self):
        """ Set flatten for the next set operation."""
        self._flatten = True
        return self 
