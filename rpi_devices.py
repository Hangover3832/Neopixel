from typing import Any, Callable, Literal
import gpiozero # type: ignore
from spidev import SpiDev # type: ignore
from devices import Spi_Clock_Rates as Spi_Clock
import numpy as np
from numpy.typing import NDArray
from devices import SPIDevice
from colors import PixelOrder

#-----------------------------------------------------------
class RpiSpiDev(SPIDevice):

    def __init__(self, 
                 *, 
                 device:Literal[0, 1], 
                 pixel_order: PixelOrder = PixelOrder.GRB,
                 gamma_function: Callable | None = None, 
                 clock_rate:Spi_Clock = Spi_Clock.CLOCK_800KHZ,
                 custom_cs: int | None = None,
                 **kwargs) -> None:


        if device not in [0, 1]:
            raise ValueError("Error: device number must be 0 or 1.")

        super().__init__(pixel_order=pixel_order, gamma_function=gamma_function, **kwargs)

        if custom_cs is not None:
            self._cs = gpiozero.OutputDevice(custom_cs, active_high=False, initial_value=True)
        else:
            self._cs = None

        # Setup SPI:
        try:
            self._spi = SpiDev()
            self._spi.open(bus=0, device=device)
            self._spi.max_speed_hz = clock_rate.value
            self._spi.mode = 0
            self._spi.bits_per_word = 8
            if custom_cs is not None:
                self._spi.no_cs = True
        except OSError: # catching a possible SpiDev.no_cs error as the rasbian kernel driver might not suppoprt it
            pass # in this case, the default chip select signal on pin 8 or 9 is still handled by the driver beside the user defined pin
        except: 
            raise RuntimeError("Error: Could not open SPI device. Ensure SPI is enabled in raspi-config and the device number is correct.")


    def write_to_device(self, buffer: NDArray[np.float32], device_data: NDArray[np.uint8] | None = None) -> Any:
        """Write pixel data to SPI device."""
        assert device_data is not None, "RpiSpiDev requires device_data (spi_buffer) to be passed from Neopixel._write_buffer()"

        super().write_to_device(buffer=buffer, device_data=device_data)

        if self._cs is not None:
            self._cs.on()

        self._spi.writebytes2(device_data.T.flatten())

        if self._cs is not None:
            self._cs.off()
