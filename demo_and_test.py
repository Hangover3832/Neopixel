from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray
from neopixel_classes import Neopixel, slice_helper
from devices import ConsoleSimulationDevice, NeopixelDevice, ConsoleSPISimulationDevice
from colors import ColorMode, PixelOrder, create_gamma_function, SOME_COLORS, GAMMA
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
from random import random, randint
from effects import Fire, Meteor
from time import monotonic, sleep
from PIL import Image


try:
    from rpi_devices import RpiSpiDev # type: ignore

    neo1dev = RpiSpiDev(device=0, pixel_order=PixelOrder.GRBW)
    neo2dev = RpiSpiDev(device=1, pixel_order=PixelOrder.GRB)

except ModuleNotFoundError:
    print("""
            Note: The python libraries 'gpiozero' and/or 'spidev' could not be imported.
            Using console simulation devices on a non Rapberry PI system.
            """)

    neo1dev = ConsoleSimulationDevice()
    neo2dev = ConsoleSPISimulationDevice(pixel_order=PixelOrder.GRBW)



def test_custom_device():

    class MyNeopixel(NeopixelDevice):
        """Example how to implement a custom Neopixel device and passing custom parameters.
        This one simply prints the content of the whole 8bit RGB array"""

        def __init__(self, *args, pixel_order: PixelOrder = PixelOrder.GRB, **kwargs) -> None:
            super().__init__(*args, pixel_order=pixel_order, **kwargs)
            print("example Neopixel device implementaion")

            print("Keyword arguments passed:")
            for key, value in kwargs.items():
                print(f"  - {key}={value}")
    
            print("Setting up a chip select output pin if desired...")


        def write_to_device(self, buffer:NDArray[np.float32], device_data: int):
            assert device_data == self.num_pixels, "This is just a test to show how to pass device_data returned from open_() to the device."
            print("Enabling chip select pin")
            print("Byte array:")
            print(self._to_uint8(buffer))
            print("Disbaling chip select pin")


        def open_(self, neopixel:Neopixel) -> Any:
            super().open_(neopixel=neopixel)
            print(f"Opening Neopixel device...", end='')
            print(f"There are {self.num_pixels} {self.pixel_order.name} pixels")
            print(f"Opening the chip select device...")
            return self.num_pixels


        def close_(self) -> Any:
            if super().close_():
                print(f"Closing the chip select device...")
                print(f"Closing Neopixel device...")
    

    neo = Neopixel(num := 1000, color_mode=ColorMode.RGB).to(MyNeopixel(param1=1234, param2=5678))
    neo[:] = np.random.rand(num, 4).astype(np.float32)
    neo().close_()



def basic_tests():
    """Basic tests an o 1D stripe"""

    n1 = Neopixel(16).to(neo1dev)
    n2 = Neopixel(10).to(neo2dev)

    n1[:] = 0.3
    n1[0:5] = 0.2, 0.3, 0.4
    print('\nn1=', n1()[:])

    n2[:] = 1.0
    n2[0] = 0.2, 0.3, 0.4
    n2[1:6] = 0.6, 0.7, 0.8, 0.0
    print('\nn2=', n2()[:])

    print()

    print("RGB & HSV:")
    n2.clear().color_mode=ColorMode.RGB
    n2[0] = 1,0, 0.0, 0.0 # red
    n2[1] = 0.0, 1.0, 0.0 # green
    n2[2] = 0.0, 0.0, 1.0 # blue
    n2.HSV[3] = (0.0, 1.0, 1.0) # red
    n2.HSV[4] = (0.333, 1.0, 1.0) # green
    n2.HSV[5] = (0.666, 1.0, 1.0) # blue
    n2()
    print()
    print(f"[RGB]{n2[:]=}")
    print(f"[HSV]{n2.HSV[:]=}")

    print()

    n2.clear()
    a = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3,0.3]]
        )
    print(f"Broadcasting {a=}")


    print("broadcast to index 0:")
    n2[0:3] = a 
    n2()
    print('\n', f"{n2[:]=}")
    print()

    print("broadcast to index 7 (to the end)")
    n2[7:] = a
    n2()
    print('\n', f"{n2[:]=}")
    print()

    print("broadcast to slice 4:7")
    n2[4:4+a[0].size] = a
    n2()
    print('\n', f"{n2[:]=}")
    print()

    try:
        print("testing shape missmatch...", end='')
        n2[0:7] = a
        assert False
    except ValueError:
        print("Ok")


    n1.decay()
    n2.decay()



def ColorModeTest():
    print("\nColor mode conversion tests:")

    def run_test():

        def process_mode(mode: ColorMode):
            print(f"Mode {mode.name}:")

            for i, (name, color) in enumerate(SOME_COLORS.items()):
                color = color[:3]
                v1 = ColorMode.RGB.convert_to(color, mode)
                v2 = mode.convert_to(v1, ColorMode.RGB)
                print(f"[{i}] RGB {color} to {mode.name}: {np.round(v1, 3)}")
                print(f"[{i}] {mode.name}->RGB->{mode.name}:         {np.round(v2, 3)}")

        for cm in ColorMode:
            process_mode(cm)

    run_test()
    print()


def GammaTest() -> None:

    neo1dev.gamma_function = GAMMA['srgb']
    neo = Neopixel(100).to(neo1dev).clear()
    # Create a brightness gradient
    neo[:50] = neo.create_gradient((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 50) # white using RGB LED's
    neo[50:] = neo.create_gradient(0.0, 1.0, 50) # using white LED's
    neo()
    neo.decay()


def Rainbow(neo: Neopixel):
    
    neo.color_mode = ColorMode.HSV

    @Every.every(0.5, n=neo) # note that the interval gets overriden in the function
    def drop(n:Neopixel):
        """Drop in some white pixels"""
        n[0] = 1.0 # drop a white pixel in a random interval
        drop.interval = random()

    @Every.every(0.01, n=neo)
    def roll(n:Neopixel):
        n.roll()()[-1] = 0.0 # clear the white LED at the end so it doesn't roll in again
        # is equivalent to:
        # n.roll()
        # n.show() # [or simply n()]
        # n[-1] = 0.0

    # Create a rainbow pattern in the default HSV space
    neo[:] = neo.create_gradient([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 150)

    @Every.While(5) # repeat for 5s
    def proceed():
        drop()
        roll()

    neo.decay()
    print()


def Raindrops(neo: Neopixel):

    @Every.every(0.1)
    def drop(n: Neopixel):
        # place a random colored pixel at a random location in a random interval
        index = randint(0, n.num_pixels-1) # random position
        hue = random() # a random color in HSV color space
        n.HSV[index] = (hue, 1.0, 1.0)
        drop.interval = random()/5

    @Every.every(1.0)
    def dropW(n:Neopixel):
        # place a white pixel at a random location every second
        index = randint(0, n.num_pixels-1) # random position
        value = random() # random brightness
        n[index] = value

    @Every.While(5.0, n=neo) # repeat for 5s
    def proceed(n:Neopixel):
        drop(n)
        dropW(n)
        n *= 0.98 # brightness decay
        n()

    # Decay until all pixels are dark:
    while neo.num_lit_pixels > 0:
        neo *= 0.98
        neo()

    neo.decay()
    print()


def light_show():
    neo = Neopixel(150).to(neo1dev)

    @Every.While(10)
    def loop():
        Rainbow(neo)
        Raindrops(neo)
        neo1dev.reversed = not neo2dev.reversed

    neo.close_()


def power_measure():
    neo2dev.gamma_function = None
    neo = Neopixel(100, color_mode=ColorMode.RGB).to(neo2dev)
    neo2dev.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
    for i, _ in enumerate(range(5), start=1):
        neo[:] = 1.0/i, 1.0/i, 1.0/i, 1.0/i
        neo()
        print(f"{neo().power_consumption=}W")
        sleep(0.2)

    neo.decay()
    neo.close_()


def effects():
    """Fire and Meteor effect on 2 devices at ones"""

    neo1dev.brightness = 1.0
    neo1dev.gamma_function = GAMMA['srgb']

    neo2dev.brightness = 1.0
    neo2dev.gamma_function = GAMMA['srgb']

    neo = Neopixel(23, brightness=0.5)
    neo.to(neo1dev)
    neo.to(neo2dev)

    candle = Fire(
        neo,
        spectrum=(0.9, 0.0),
        decay_factor=(0.98, 0.92),
        spark_interval_factor=0.05,
        spark_propagation_interval=0.01
        )

    meteor = Meteor(
        neo, 
        decay_value=0.925, 
        roll_interval=0.025,
        shoot_intervall=1.5
        )
    
    candle.show_temperature_gradient()

    #@Every.While(10)
    #def run_effects():
    while True:
        neo.reversed = False
        candle.resume()
        Every.While(5)(candle.progress)
        candle.pause()
        neo.decay()

        neo.reversed = True
        meteor.resume()
        Every.While(5)(meteor.progress) # drop for 10s
        meteor.pause()
        neo.decay()

    print()


def pixel_array_test():
    """Demo for a 32x8 pixel array.
    The pixels are arranged column by column, wired in zigzag:
    """
    """
    | ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ ┌ ┐ |
    | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘ └ ┘
    """

    def prepare_buffer(buffer: NDArray):
        """Before writing the buffer to the device, rearrange it to match the physical wiring."""
        buffer = buffer.transpose(1, 0, 2) # rearrange the buffer to be column by column
        buffer[1::2] = buffer[1::2][:, ::-1] # reverse every odd column for zigzag wiring
        return buffer


    neo2dev.on_write_buffer = prepare_buffer
    neo2dev.gamma_function = GAMMA['srgb']

    height = 8
    width = 32
    neo = Neopixel((height, width), color_mode=ColorMode.RGB, brightness=0.25, auto_write=True)
    neo.to(neo2dev)

    loop:bool = True

    while loop:
        loop = False
        x, y = 0, 0
        neo[y, x] = (1.0, 0.0, 0.0) # top-left pixel red
        neo[0, -1] = (0.0, 1.0, 0.0) # top-right pixel green
        neo[-1, 0] = (0.0, 0.0, 1.0) # bottom-left pixel blue
        neo[-1, -1] = (1.0, 1.0, 0.0) # bottom-right pixel yellow

        neo.decay()

        # create a rainbow column-wise from left to right
        neo.HSV.T.F[:] = neo.create_gradient((0.0, 1.0, 1.0), (1.0, 1.0, 1.0), 256)
        neo.decay()

        # create a rainbow row-wise from top to bottom
        neo.HSV.F[:] = neo.create_gradient((0.0, 1.0, 1.0), (1.0, 1.0, 1.0), 256)
        sleep(1)

        # shift row by row increasing and fill the shhifted in pixels with some color
        for i in range(8):
            neo.roll(shift=i, axis=0)[i] = (0.1,0.5,0.25)
            sleep(0.125)

        #sleep(1)

        # roll column-wise by column
        for i in range(32):
            neo.roll(shift=i, axis=0)[:, i]
            sleep(0.02)

        neo.color_mode = ColorMode.RGB
        neo.begin_update()
        neo.clear()
        # Draw a colorful grid:
        neo[:, 0]  = (0.0, 1.0, 0.0) # 1st column green
        neo[:, 15] = (0.0, 0.0, 1.0) # middle column blue
        neo[:, -1] = (1.0, 1.0, 0.0) # last column yellow
        neo[0] = (1.0, 0.0, 0.0) # 1st row red
        neo[-1] = (0.0, 1.0, 1.0) # last row cyan
        neo.end_update()
        #sleep(1)

        # Create random noise pattern:
        @Every.While(1)
        def noise():
            neo.RGB.F[:] = np.random.random([256, 3]).astype(np.float32)
            sleep(0.01)

        # Clear a rectangle in the middle:
        neo[2:6, 10:22] = neo.blank
        neo.decay()


        # Display 8x8 images:
        neo.begin_update()
        neo.RGB[0:8, 0:8] = Image.open('icon1.png')
        neo.RGB[0:8, 8:16] = Image.open('icon2.png')
        neo.RGB[0:8, 16:24] = Image.open('icon3.png')
        neo.RGB[0:8, 24:32] = Image.open('icon4.png')
        neo.end_update()
        neo.decay()

        # Display a 32x8 image:
        neo.RGB[:] = Image.open('__00002_.png')
        sleep(1)

        neo[:] = neo[::-1, :] # flip vertically
        sleep(1)
        neo[:] = neo[:, ::-1] # flip horizontally
        sleep(1)

        for _ in range(8):
            neo >>= 1 # roll along axis 0
            sleep(0.1)

        for _ in range(8):
            neo <<= 1

        for _ in range(32):
            neo.roll(1, axis=1)[:]

        for _ in range(32):
            v = 0.5, 0.5, 0.5
            neo.RGB.roll(-1, axis=1)[:] = v

        neo.decay()

    neo2dev.on_write_buffer = None


def pixel_volumetric_test():
    """Demo for a volumetric display 4 layers, 8x8pixels each"""


    def prepare_buffer(buffer: NDArray):
        """Before writing the buffer to the device, rearrange it to match the physical wiring."""
        buffer = buffer.transpose(0, 2, 1, 3) # rearrange the buffer to be layer-wise, column by column
        buffer[:, 1::2] = buffer[:, 1::2][:, :, ::-1] # reverse every odd column for zigzag wiring
        return buffer


    neo2dev.on_write_buffer = prepare_buffer
    neo2dev.gamma_function = GAMMA['srgb']
    # neo2dev.brightness = 1.0

    layers = 4
    height = 8
    width = 8
    neo = Neopixel((layers, height, width), color_mode=ColorMode.RGB, brightness=0.25, auto_write=True)
    neo.to(neo2dev)

    loop: bool = True

    while loop:
        loop = False

        neo.HSV.T.F[:] = neo.create_gradient((0,1,1), (1,1,1), neo.num_pixels)
        sleep(1)

        # single pixels
        neo.begin_update()
        neo.clear()
        x, y = 0, 0
        neo[0, y, x] = 1,0,0
        neo[1, y, x] = 0,1,0
        x, y = 2, 5
        neo[0, y, x] = 0,0,1
        neo[1, y, x] = 1,1,0
        neo.end_update()

        sleep(1)

        # whole layers
        neo.begin_update()
        neo[0] = 1,0,0 # 1st layer red
        neo[1] = 0,1,0 # 2nd layer green
        #neo[2] = 0,0,1 # 3rd layer blue
        neo[3] = np.random.random([8 ,8, 4]).astype(np.float32) # 4th layer random
        neo.end_update()
        sleep(1)

        # image 8x8 on layer 1
        neo[1] = np.asarray(Image.open("icon2.png"), dtype=np.float32) / 255.0
        sleep(1)

        # roll image on layer 1 horizontally (axis=1)
        for i in range(64):
            neo.begin_update()
            neo.roll(axis=1)[1]
            neo.HSV[0] = 0, 1, 1 - i/64.
            neo.HSV[2] = 0, 1, i/64.
            neo.end_update()
            sleep(0.01)


        # vertical rainbow on layer 0
        neo.HSV.F[0] = neo.create_gradient((0,1,1), (1,1,1), 64)

        # horizontal rainbow on layer 2
        neo.HSV.T.F[2] = neo.create_gradient((0,1,1), (1,1,1), 64)

        sleep(1)

        # roll through the areas
        for _ in range(4):
            neo.roll[:]
            sleep(1)

        # roll through all color channels of each layer
        for i in range(4):
            for _ in range(8):
                neo.roll(axis=2)[i]
                sleep(0.05)

        # roll through all color channels
        for _ in range(32):
            neo.roll(axis=3)[:]
            sleep(0.05)

        neo.decay()

    neo2dev.on_write_buffer = None



if __name__ == "__main__":
    Neopixel(150).to(neo1dev).clear()().close_()
    Neopixel(256).to(neo2dev).clear()().close_()

    #test_custom_device()
    #basic_tests()
    #GammaTest()
    #ColorModeTest()
    #power_measure()
    #light_show()
    #effects()
    #pixel_array_test()
    pixel_volumetric_test()
