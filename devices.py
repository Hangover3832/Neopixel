from enum import Enum
from typing import Any, Callable
from numpy.typing import NDArray
import numpy as np
from neopixel_classes import NeopixelDevice, Neopixel
from colors import PixelOrder
import tkinter as tk

class Spi_Clock_Rates(Enum):
    # SPI clock rates
    CLOCK_400KHZ  = 1_625_000
    CLOCK_800KHZ  = 3_250_000
    CLOCK_1200KHZ = 6_500_000


#-----------------------------------------------------------
class SPIDevice(NeopixelDevice):
    """
    **Base class for SPI based Neopixel devices.**</br>
    Implements the encoding of pixel data to SPI bit patterns.
    The actual SPI transmission must be implemented by child classes.

    1 SPI byte encodes 2 bits of pixel data:
        '10' -> 0xC0
        '11' -> 0xCC
        '00' -> 0x80
        '01' -> 0x8C
    
    1 pixel = 8 bits per color channel
    For RGB pixels: 1 pixel = 24 bits = 12 SPI bytes
    For RGBW pixels: 1 pixel = 32 bits = 16 SPI bytes
    The instance creates a SPI buffer in open_() that is passed to write_to_device() calls.
    """

    SPI_HIGH_BIT   = 0xC0
    SPI_LOW_BIT    = 0x80
    SPI_HIGH_BIT2  = 0x0C
    SPI_LOW_BIT2   = 0x08

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, gamma_function: Callable | None = None, **kwargs) -> None:

        super().__init__(pixel_order=pixel_order, gamma_function=gamma_function, **kwargs)

        if self._pixel_order.num == 4:
            self._double_bits_per_pixel = 16
            self._msb_mask = 0x80000000
            self._c_mask = 0xFFFFFFFF
        else:
            self._double_bits_per_pixel = 12
            self._msb_mask = 0x800000
            self._c_mask = 0xFFFFFF


    def open_(self, neopixel:Neopixel) -> NDArray[np.uint8]:
        """Open device and create SPI buffer to be stored in Neopixel instance."""
        super().open_(neopixel)
        return np.zeros([self._double_bits_per_pixel, self._num_pixels], dtype=np.uint8)


    def write_to_device(self, buffer: NDArray[np.float32], device_data: NDArray[np.uint8]) -> Any:
        """Encode pixel data to SPI byte patterns.
        
        :param buffer: Pixel data to encode
        :type buffer: np.ndarray[[float, ...], [float, ...]]
        :param device_data: The SPI buffer from the Neopixel instance
        :type device_data: np.ndarray[np.uint8]
        :raises ValueError: If device_data is None
        """

        rgb_buffer = self._to_uint32(buffer)

        # shift out 2 bits of each pixel and encode them to a byte for SPI transmission:
        for i in range(self._double_bits_per_pixel):
            bit1 = (rgb_buffer & self._msb_mask).astype(bool)
            rgb_buffer = ((rgb_buffer << 1) & self._c_mask).astype(np.uint32)
            bit2 = (rgb_buffer & self._msb_mask).astype(bool) 
            rgb_buffer = ((rgb_buffer << 1) & self._c_mask).astype(np.uint32)
            # encode 2 pixel bits into 1 SPI byte:
            device_data[i] = (np.where(
                        bit1, self.SPI_HIGH_BIT,  self.SPI_LOW_BIT
                        ) | np.where(
                        bit2, self.SPI_HIGH_BIT2, self.SPI_LOW_BIT2)
                    ).astype(np.uint8)

        # device_data is now ready to be used by a child class


#-----------------------------------------------------------
class ConsoleSimulationDevice(NeopixelDevice):
    """
    A console based Neopixel device simulation.
    It prints colored characters to the console to represent the Neopixel colors.
    You can configure line endings, line splits, and the character used to represent LEDs.
    """

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, gamma_function: Callable | None = None, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=None, gamma_function=gamma_function, **kwargs)
        self.is_simulated = True
        self.line_end: str = ''
        self.split_lines: int = 0
        self.inverse: bool = False
        self.led_char = "\u25CF"

    def close_(self) -> Any:
        super().close_()
        print()


    def write_to_device(self, buffer:NDArray[np.float32], device_data: None) -> Any:
        print('', end='\r')
        for i, value in enumerate(self._to_uint8(buffer), start=1):
            g, r, b = value[:3]
            w = value[3] if len(value) > 3 else 0
            if self.inverse:
                print(f"\033[48;2;{r};{g};{b}m", end='') # the background color simulates the white LED in a GRBW Neopixel
                print(f"\033[38;2;{w};{w};{w}m{self.led_char}\033[0m", end='', flush=True) # print the LEDs
            else:
                print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
                print(f"\033[38;2;{r};{g};{b}m{self.led_char}\033[0m", end='', flush=True) # print the LEDs
            if (self.split_lines > 0) and (i % self.split_lines) == 0:
                print()

        if (self.split_lines > 0) and (i % self.split_lines) > 0: # type: ignore
            print()

        if self.line_end:
            print('', self.line_end)


#-----------------------------------------------------------
class ConsoleSPISimulationDevice(SPIDevice):
    """
    A console based Neopixel device simulation for SPI based Neopixel devices.
    It prints colored circles to the console to represent the Neopixel colors.
    This class is made for testing SPI based Neopixel devices.
    """

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, gamma_function: Callable | None = None, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=None, gamma_function=gamma_function, **kwargs)
        self.is_simulated = True
        self.line_end: str = ''
        self.led_char: str = "\u25CF"

    def close_(self) -> Any:
        super().close_()
        print()

    def write_to_device(self, buffer:NDArray[np.float32], device_data: NDArray[np.uint8]) -> Any:
        super().write_to_device(buffer, device_data)
 
        if device_data is None:
            raise ValueError("ConsoleSPISimulationDevice requires device_data to be passed")

        def convert(bits: np.ndarray) -> int:
            # bits = np.ndarray[uint8, uitn8, uint8, uint8] = 4 double bits = 8 bits = 1 byte
            bit_values = {0xCC: 0b11, 0xC8: 0b10, 0x8C: 0b01, 0x88: 0b00} # SPI encodings {1byte: 2bits}
            result = 0
            for bit in bits:
                result = (result << 2) | bit_values[bit] # shift 2 bits and inject 2 new bits
            return result

        print('', end='\r')
        for bits in device_data.T: 
            g, r, b = convert(bits[0:4]), convert(bits[4:8]), convert(bits[8:12])
            w = convert(bits[12:16]) if bits.shape[0]>12 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
            print(f"\033[38;2;{r};{g};{b}m{self.led_char}\033[0m", end='', flush=True) # print the LEDs

        if self.line_end:
            print('', self.line_end)

