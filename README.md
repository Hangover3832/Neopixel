# Neopixel

A lightweight Python library for driving and simulating Neopixel / WS281x-style LED strips. The project provides device abstractions, color utilities, common effects, and demo/test scripts so you can run and prototype LED animations on real hardware or in a console simulator.

## Features

- Simple `Neopixel` device abstraction and buffer helpers.
- Multiple device backends and simulations (SPI, RPi SPI, console simulation).
- Color utilities and conversions for common color spaces (RGB and HSV are currently implemented).
- Built-in effects (Fire, Meteor, etc.) and a test/demo harness.
- Supports multi dimensional pixel arrangements like 2D display area or 3D volumetric display or even higher dimensions.

## Requirements

- Python 3.8+
- See `requirements.txt` for optional dependencies.

## Installation

Clone the repository and install dependencies (optional):

```bash
git clone <repo-url>
cd Neopixel
python -m pip install -r requirements.txt
```

This library is designed to be run on Linux/Raspberry Pi for SPI-backed devices, but includes simulation devices for development as well as an abstract device base class to implement custom hardware on other platforms.

## Quick start

- Run the demo and tests (console simulation or real device depending on configuration):

```bash
python demo_and_test.py
```

- Use the library from your own script:

```python
from neopixel_classes import Neopixel
from devices import ConsoleSimulationDevice
from colors import ColorMode

# Create a Neopixel instance (30 pixels). Use `auto_write=True` to update on each change.
np = Neopixel(30, color_mode=ColorMode.HSV, auto_write=False)

# Attach a device (console simulation for development):
dev = ConsoleSimulationDevice()
np.to(dev)  # attach the device

# Write pixels (values are floats in range 0.0..1.0) and push to the device
np.fill((0.0, 1.0, 1.0))  # fill red in HSV color space
#or
np.RGB[:] = (0.0, 0.0, 1.0) # fill blue in RGB color space
np.show()
```

Adjust the above to match your concrete device class from `devices.py` or `rpi_devices.py` (see `RpiSpiDev` for Raspberry Pi SPI usage).

## Usage Examples

- Console simulation (development, no hardware required):

```python
from neopixel_classes import Neopixel
from devices import ConsoleSimulationDevice
from colors import ColorMode

np = Neopixel(30, color_mode=ColorMode.RGB, auto_write=False).to(ConsoleSimulationDevice)
np.fill((0.2, 0.0, 0.8))  # purple-ish (floats 0..1)
np.show()
```

- Raspberry Pi SPI device (real hardware):

```python
from neopixel_classes import Neopixel
from rpi_devices import RpiSpiDev
from colors import ColorMode

np = Neopixel(60, auto_write=False, color_mode=ColorMode.RGB)
dev = RpiSpiDev(device=0)  # device 0 or 1
np.to(dev)
np.fill((0.0, 0.0, 1.0))  # blue
np.show()
```

- Auto-write and batch updates:

```python
# immediate updates on change
np.auto_write = True
np[0:5] = (1.0, 0.0, 0.0)  # first five pixels become red immediately

# batched updates (useful when making many changes)
np.begin_update()
np[0] = (1.0, 0.0, 0.0)
np[1] = (0.0, 1.0, 0.0)
np.end_update()  # triggers a single update/show when done
```

- Multi-dimensional pixel arrays:
```python
from neopixel_classes import Neopixel
from rpi_devices import RpiSpiDev
from colors import ColorMode, GAMMA
from PIL import Image

# create 4 quadratic areas, 8x8 each on a 256 pixel stripe
# stacked on top of each other will build a volumetric display
np = Neopixel((4, 8, 8), color_mode=ColorMode.RGB)
dev = RpiSpiDev(device=0), gamma_function=GAMMA['srgb'])  # device 0 or 1, use sRGB gamma correction
np.to(dev)
np[0] = 0.0, 0.0, 1.0  # 1st area blue
np[1] = 1.0, 0.0, 0.0  # 2nd area red
np[2] = 0.0, 1.0, 0.0  # 3rd area green
np[3] = Image.opem('image.png') # 4th area display a PIL Image, it will be resized to 8x8 autmatically
np() # or np.show()

# vertical rainbow on area 1
neo.HSV.F[0] = neo.create_gradient((0,1,1), (1,1,1), 64)
np()

# horizontal rainbow on area 3
neo.HSV.T.F[2] = neo.create_gradient((0,1,1), (1,1,1), 64)
np()

```

- Using built-in effects:

```python
from effects import Fire
from time import sleep

np = Neopixel(60, auto_write=False)
dev = ConsoleSimulationDevice()
np.to(dev)

fire = Fire(np)
while True:
	fire.progress()
	sleep(0.02)
```

See `devices.py`, `rpi_devices.py`, and `effects.py` for more examples and options.

## API Overview

Key modules and where to look:

- **Neopixel core**: [neopixel_classes.py](neopixel_classes.py) — `Neopixel`, `NeopixelDevice`, helpers and buffer utilities.
- **Devices / backends**: [devices.py](devices.py) — SPI base classes and console simulations.
- **Raspberry Pi SPI driver**: [rpi_devices.py](rpi_devices.py) — `RpiSpiDev` SPI-backed device.
- **Colors & enums**: [colors.py](colors.py) — `ColorMode`, `PixelOrder`, and color helpers.
- **Color conversions**: [color_conversions.py](color_conversions.py) — conversion utilities between color spaces.
- **Effects**: [effects.py](effects.py) — built-in `NeopixelEffect` subclasses like `Fire`, `Meteor`.
- **Demo / tests**: [demo_and_test.py](demo_and_test.py) — example usage and quick manual tests.

## Examples & Recipes

- Use `ConsoleSimulationDevice` from `devices.py` to iterate effects without hardware.
- Use `RpiSpiDev` from `rpi_devices.py` on a Raspberry Pi with SPI enabled to drive real strips.

### Custom Hardware On Any Platform/System

```python
from neopixel_classes import NeopixelDevice

class CustomDevice(NeopixelDevice)

    def __init__(self, *, 
            pixel_order:PixelOrder=PixelOrder.GRB, 
            gamma_function: Callable | None = None, 
            **kwargs
            ) -> None:
        super().__init__(pixel_order=pixel_order, gamma_function=gamma_function, **kwargs)
        # do your stuff here to initialize the device, allocating buffer etc


    def open_(self, neopixel:Neopixel) -> Any:
        """Open the device."""
        super().open_(neopixel)
        device_data = "any desired data will be stored in the Neopixel instance and gets passed to the write_methods"
        return device_data


    def close_(self) -> Any:
        """Close the Neopixel device. Override this method in subclasses to implement device-specific closing logic."""
        return super().close_()


    def _write_buffer(self, buffer:NDArray[np.float32], device_data: Any) -> Any:
        """This would be the right place to modify the pixel buffer in its initial shape before it gets flattened and written to the device. You do not need to override this method if this is not neccessary."""
        return super()._write_buffer(buffer, device_data)


    def write_to_device(self, buffer:NDArray[np.float32], device_data: Any) -> Any:
        """Write the given buffer to the Neopixel device.
        Override this method in subclasses to implement device-specific writing logic.

        :param buffer: A 2D array of shape (num_pixels, num_channels=4) containing pixel values.
        :type buffer: np.ndarray[float32]
        :param device_data: Optional device-specific data from the Neopixel instance.
        :type device_data: Any
        """
        # implement your logic to write the data to you device

```

## Contributing

Contributions, bug reports and improvements are welcome. Please open issues or pull requests.

## License

This project is provided under the terms of the included `LICENSE` file.
