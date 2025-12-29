from typing import Literal
import gpiozero # type: ignore
from spidev import SpiDev # type: ignore
from neopixel_classes import OutputDevice, Spi_Clock, Spi_Bit_Encoding
import numpy as np
from numpy.typing import NDArray
from neopixel_classes import Neopixel, NoOutputDevice, SPIDevice
from colors import PixelOrder, ColorMode

class GPIOzeroOutputDevice(OutputDevice):

    def __init__(self, *args, bcm_pin:int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = gpiozero.OutputDevice(bcm_pin, active_high=False, initial_value=True)

    def enable(self):
        self.device.on()
    
    def disable(self):
        self.device.off()

    @property
    def enabled(self) -> bool:
        return self.device.value

#-----------------------------------------------------------
class RpiSpiDev(SPIDevice):

    def __init__(self, 
                 *, 
                 device:Literal[0, 1], 
                 pixel_order: PixelOrder,
                 clock_rate:Spi_Clock=Spi_Clock.CLOCK_800KHZ, 
                 custom_cs:OutputDevice | None = None,
                 **kwargs) -> None:

        super().__init__(pixel_order=pixel_order, custom_cs=custom_cs, **kwargs)
        self._spi_buffer: NDArray[np.uint8] | None = None
        # Setup SPI:
        try:
            self._spi = SpiDev()
            self._spi.open(bus=0, device=device)
            self._spi.max_speed_hz = clock_rate.value
            self._spi.mode = 0
            self._spi.bits_per_word = 8
            if custom_cs:
                self._spi.no_cs = True
        except OSError: # catching a possible SpiDev.no_cs error as the rasbian kernel driver might not suppoprt it
            pass # in this case, the default chip select signal on pin 8 or 9 is still handled by the driver beside the user defined pin
        except: 
            raise RuntimeError("Error: Could not open SPI device. Ensure SPI is enabled in raspi-config and the device number is correct.")

        self._cs = custom_cs or NoOutputDevice()

    def write_to_device(self, buffer: NDArray[np.uint8]) -> None:
        super().write_to_device(buffer)
        assert self._spi_buffer is not None
        self._spi.writebytes2(self._spi_buffer.T.flatten())
