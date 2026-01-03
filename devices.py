from enum import Enum
from typing import Any, Callable
from numpy.typing import NDArray
import numpy as np
from neopixel_classes import NeopixelDevice #, Neopixel
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

    Child classes can access the encoded SPI byte buffer via the 'spi_buffer' property.
    """

    SPI_HIGH_BIT   = 0xC0
    SPI_LOW_BIT    = 0x80
    SPI_HIGH_BIT2  = 0x0C
    SPI_LOW_BIT2   = 0x08

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, **kwargs) -> None:

        super().__init__(pixel_order=pixel_order, **kwargs)

        if self._pixel_order.num == 4:
            self._double_bits_per_pixel = 16
            self._msb_mask = 0x80000000
            self._c_mask = 0xFFFFFFFF
        else:
            self._double_bits_per_pixel = 12
            self._msb_mask = 0x800000
            self._c_mask = 0xFFFFFF

        self._spi_buffer: NDArray[np.uint8] | None = None


    def set_num_pixels(self, num: int) -> Any:
        result = super().set_num_pixels(num)
        self._spi_buffer = np.zeros([self._double_bits_per_pixel, num], dtype=np.uint8)
        return result


    def write_to_device(self, buffer: NDArray[np.float32]) -> Any:
        """Encode pixel data to SPI byte patterns."""

        rgb_buffer = self._to_uint32(buffer)

        # shift out 2 bits of each pixel and encode them to a byte for SPI transmission:
        for i in range(self._double_bits_per_pixel):
            bit1 = (rgb_buffer & self._msb_mask).astype(bool)
            rgb_buffer = ((rgb_buffer << 1) & self._c_mask).astype(np.uint32)
            bit2 = (rgb_buffer & self._msb_mask).astype(bool) 
            rgb_buffer = ((rgb_buffer << 1) & self._c_mask).astype(np.uint32)
            # encode 2 pixel bits into 1 SPI byte:
            self.spi_buffer[i] = (np.where(
                        bit1, self.SPI_HIGH_BIT,  self.SPI_LOW_BIT
                        ) | np.where(
                        bit2, self.SPI_HIGH_BIT2, self.SPI_LOW_BIT2)
                    ).astype(np.uint8)

        # self.spi_buffer is now ready to be used by a child class

    @property
    def spi_buffer(self) -> NDArray[np.uint8]:
        assert self._spi_buffer is not None, "Attribute _spi_buffer is None"
        return self._spi_buffer


#-----------------------------------------------------------
class ConsoleSimulationDevice(NeopixelDevice):
    """
    A console based Neopixel device simulation.
    It prints colored circles to the console to represent the Neopixel colors.
    You can configure line endings, line splits, white LED's and the character used to represent LEDs.
    """

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=None, **kwargs)
        self.is_simulated = True
        self.line_end: str = ''
        self.split_lines: int = 0
        self.inverse: bool = False
        self.led_char = "\u25CF"

    def close_(self) -> Any:
        if super().close_():
            print()

    def write_to_device(self, buffer:NDArray[np.float32]) -> Any:
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

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=None, **kwargs)
        self.is_simulated = True
        self.line_end: str = ''
        self.led_char: str = "\u25CF"

    def close_(self) -> Any:
        if super().close_():
            print()

    def write_to_device(self, buffer:NDArray[np.float32]) -> Any:
        super().write_to_device(buffer)
 
        assert self._spi_buffer is not None

        def convert(bits: np.ndarray) -> int:
            # bits = np.ndarray[uint8, uitn8, uint8, uint8] = 4 double bits = 8 bits = 1 byte
            bit_values = {0xCC: 0b11, 0xC8: 0b10, 0x8C: 0b01, 0x88: 0b00} # SPI encodings {1byte: 2bits}
            result = 0
            for bit in bits:
                result = (result << 2) | bit_values[bit] # shift 2 bits and inject 2 new bits
            return result

        print('', end='\r')
        for bits in self.spi_buffer.T: 
            g, r, b = convert(bits[0:4]), convert(bits[4:8]), convert(bits[8:12])
            w = convert(bits[12:16]) if bits.shape[0]>12 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
            print(f"\033[38;2;{r};{g};{b}m{self.led_char}\033[0m", end='', flush=True) # print the LEDs

        if self.line_end:
            print('', self.line_end)


#-----------------------------------------------------------
class GraphicSimulation(NeopixelDevice):
    """
    A graphical Neopixel device simulation using tkinter.
    This really does not do very well....
    """

    def print_hello(self):
        print("Hello")

    def __init__(self, 
                 *, 
                 pixel_order: PixelOrder = PixelOrder.RGB, 
                 led_size: int = 25,
                 horizontal: bool = False,
                 split: int = 0,
                 **kwargs) -> None:
        
        super().__init__(pixel_order=pixel_order, **kwargs)

        self.horizontal = horizontal
        self.split = split
        self.led_size = led_size
        self.window = tk.Tk()
        self.window.config(bg='black')
        self.canvas = tk.Canvas(self.window, bg='black', borderwidth=0)
        self.canvas.pack()


    def open_(self):
        if super().open_():
            width, height = self.led_size, self.led_size*(self.num_pixels)
            if self.horizontal:
                width, height = height, width

            self.canvas.configure(width=width, height=height)
            return "test"


    def close_(self):
        if super().close_():
            self.window.mainloop()
            return True
        return False


    def write_to_device(self, buffer: NDArray[np.float32]) -> Any:

        if not self._is_open:
            raise RuntimeError("Error: Cannor write to the device, it is closed.")

        buf = self._to_uint32(buffer) if self.horizontal else self._to_uint32(buffer[::-1])
        for i in range(buf.shape[0]):
            color_code = f"#{hex(buf[i])[2:].zfill(6)}"
            if self.horizontal:
                self.canvas.create_oval(i*self.led_size, 0, (i+1)*self.led_size, self.led_size, fill=color_code)
            else:
                self.canvas.create_oval(0, i*self.led_size, self.led_size, (i+1)*self.led_size, fill=color_code)
        
        self.window.update()
