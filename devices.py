from enum import Enum
from typing import Any
from numpy.typing import NDArray
import numpy as np
from neopixel_classes import NeopixelDevice, Neopixel
from colors import PixelOrder

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
class SPIDevice(NeopixelDevice):

    def __init__(self, *args, pixel_order:PixelOrder=PixelOrder.GRB, **kwargs) -> None:
        super().__init__(*args, pixel_order=pixel_order, **kwargs)
        self._spi_buffer: NDArray[np.uint8] | None = None


    def set_neopixel(self, neopixel: Neopixel):
        super().set_neopixel(neopixel)
        if self._pixel_order.num == 4:
            self._double_bits_per_pixel = 16
            self._msb_mask = 0x80000000
            self._c_mask = 0xFFFFFFFF
        else:
            self._double_bits_per_pixel = 12
            self._msb_mask = 0x800000
            self._c_mask = 0xFFFFFF

        # Pre-allocate buffer for the encoded bits
        self._spi_buffer = np.zeros([self._double_bits_per_pixel, self.neopixel.num_pixels], dtype=np.uint8)


    def write_to_device(self, buffer: NDArray[np.float32]) -> Any:
        if self.neopixel is None or self._spi_buffer is None:
            raise ValueError("Attribute `neopixel` or its pixel buffer is not set")

        rgb_buffer = self._to_uint8(buffer)

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
            
        # self._spi_buffer is now ready to be used by a child class


#-----------------------------------------------------------
class ConsoleSimulationDevice(NeopixelDevice):

    LED_CHAR = "\u25CF"

    def __init__(self, *args, line_end:str='', pixel_order:PixelOrder=PixelOrder.GRB, **kwargs) -> None:
        super().__init__(*args, pixel_order=pixel_order, custom_cs=None, **kwargs)
        self.is_simulated = True
        self.line_end = line_end

    def close_(self) -> Any:
        if super().close_():
            print()

    def write_to_device(self, buffer:NDArray[np.float32]) -> Any:
        print('', end='\r')
        for value in (_ := self._to_uint8(buffer)):
            g, r, b = value[:3]
            w = value[3] if len(value) > 3 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
            print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='', flush=True) # print the LEDs

        if self.line_end:
            print('', self.line_end)


#-----------------------------------------------------------
class ConsoleSPISimulationDevice(SPIDevice):

    LED_CHAR = "\u25CF"

    def __init__(self, *args, line_end:str='', pixel_order:PixelOrder=PixelOrder.GRB, **kwargs) -> None:
        super().__init__(*args, pixel_order=pixel_order, custom_cs=None, **kwargs)
        self.is_simulated = True
        self.line_end = line_end

    def close_(self) -> Any:
        if super().close_():
            print()

    def write_to_device(self, buffer:NDArray[np.float32]) -> Any:
        super().write_to_device(buffer)
        assert self._spi_buffer is not None

        if self.neopixel is None:
            raise ValueError("Attribute `neopixel` is not set")

        def convert(bits: np.ndarray) -> int:
            # bits = np.ndarray[uint8, uitn8, uint8, uint8] = 4 double bits = 8 bits = 1 byte
            bit_values = {0xCC: 0b11, 0xC8: 0b10, 0x8C: 0b01, 0x88: 0b00} # SPI encodings {1byte: 2bits}
            result = 0
            for bit in bits:
                result = (result << 2) | bit_values[bit] # shift 2 bits and inject 2 new bits
            return result

        print('', end='\r')
        for bits in self._spi_buffer.T: 
            g, r, b = convert(bits[0:4]), convert(bits[4:8]), convert(bits[8:12])
            w = convert(bits[12:16]) if bits.shape[0]>12 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
            print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='', flush=True) # print the LEDs

        if self.line_end:
            print('', self.line_end)
