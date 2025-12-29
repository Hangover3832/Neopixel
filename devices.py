from typing import Optional
from numpy.typing import NDArray
import numpy as np
from neopixel_classes import OutputDevice, NeopixelDevice, Neopixel, Spi_Bit_Encoding
from colors import PixelOrder


#-----------------------------------------------------------
class _NoOutputDevice(OutputDevice):
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
class SPIDevice(NeopixelDevice):

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, custom_cs:OutputDevice | None = None, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=custom_cs, **kwargs)
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


    def write_to_device(self, buffer: NDArray[np.float32]) -> int:
        if self.neopixel is None or self._spi_buffer is None:
            raise ValueError("Attribute `neopixel` is not set")
        
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
            
        return len(self._spi_buffer)
    
        # self._spi_buffer is now ready to be used by the child class


#-----------------------------------------------------------
class ConsoleSimulationDevice(NeopixelDevice):

    LED_CHAR = "\u25CF"

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=None, **kwargs)
        self.is_simulated = True

    def write_to_device(self, buffer:NDArray[np.float32]) -> int:
        print('', end='\r')
        for value in (_ := self._to_uint8(buffer)):
            g, r, b = value[:3]
            w = value[3] if len(value) > 3 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
            print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='', flush=True) # print the LEDs

        return 0

#-----------------------------------------------------------
class ConsoleSimulationSPIDevice(SPIDevice):

    LED_CHAR = "\u25CF"

    def __init__(self, *, pixel_order:PixelOrder=PixelOrder.GRB, **kwargs) -> None:
        super().__init__(pixel_order=pixel_order, custom_cs=None, **kwargs)
        self.is_simulated = True

    def write_to_device(self, buffer:NDArray[np.float32]) -> int:
        super().write_to_device(buffer)
        assert self._spi_buffer is not None
        _buffer = self._spi_buffer.T.flatten()

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
        _buffer = _buffer.reshape([_buffer.shape[0]//double_bits_per_pixel, double_bits_per_pixel]) # ...and reshape the buffer accordingly
        print('', end='\r')
        for bits in _buffer: 
            g, r, b = convert(bits[0:4]), convert(bits[4:8]), convert(bits[8:12])
            w = convert(bits[12:16]) if bits.shape[0]>12 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
            print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='', flush=True) # print the LEDs

        return 0
    