from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray
from neopixel_classes import Neopixel
from devices import ConsoleSimulationDevice, NeopixelDevice, GraphicSimulation, ConsoleSPISimulationDevice
from colors import ColorMode, PixelOrder, create_gamma_function, SOME_COLORS, GAMMA
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
from random import random, randint
from effects import Fire, Meteor, NeopixelEffect
from time import monotonic, sleep


try:
    from rpi_devices import RpiSpiDev # type: ignore

    neo1dev = RpiSpiDev(device=0)
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
    

    Neopixel(num := 1000, color_mode=ColorMode.RGB).to(MyNeopixel(param1=1234, param2=5678)).set_value(0, np.random.rand(num, 4,).astype(np.float32))().close_()
    """ 
    equals to:
        num = 1000
        dev = MyNeopixel(param1=1234, param2=5678)
        neo = Neopixel(num, color_mode=ColorMode.RGB)
        neo.to(dev)
        neo.set_value(0, np.random.rand(num, 4))
        neo()
        neo.close_()
    """


def basic_tests():
    n1 = Neopixel(16).to(neo1dev)
    n2 = Neopixel(10).to(neo2dev)

    n1[:] = 0.3
    n1[0] = 0.2, 0.3, 0.4
    print('\nn1=', n1()[:])

    n2[:] = 1.0
    n2[0] = 0.2, 0.3, 0.4
    n2[1] = 0.6, 0.7, 0.8, 0.0
    print('\nn2=', n2()[:])

    print()

    print("RGB:")
    n2.clear().color_mode=ColorMode.RGB
    n2[0] = 1,0, 0.0, 0.0 # red
    n2[1] = 0.0, 1.0, 0.0 # green
    n2[2] = 0.0, 0.0, 1.0 # blue
    n2.set_value(3, (0.0, 1.0, 1.0), color_mode=ColorMode.HSV) # red
    n2.set_value(4, (0.333, 1.0, 1.0), color_mode=ColorMode.HSV) # green
    n2.set_value(5, (0.666, 1.0, 1.0), color_mode=ColorMode.HSV) # blue
    n2()
    print()
    print(n2[:])

    print()

    print("Broadcasting")
    n2.clear()
    a = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3,0.3]]
        )

    n2[0] = a # broadcast to index 0
    n2()
    print('\n', n2[:])
    print()

    n2[7] = a # broadcast to index 7 (to the end)
    n2()
    print('\n', n2[:])
    print()

    n2[4:7] = a # broadcast to slice 4:7
    n2()
    print('\n', n2[:])
    print()

    try:
        n2[8] = a
        assert False, "This should not happen"
    except ValueError:
        print("Ok")


    n2.create_gradient(0.0, 1.0) # create a gradient, but to the white pixels only
    n2()
    print('\n', n2[:])
    print()

    n1.create_gradient(0.0, 1.0) # create a gradient, but to the white pixels only
    n1()
    print('\nn1=', n1[:])
    print()

    print("Rainbow gradient:")
    n1.clear().color_mode=ColorMode.HSV
    start = [0.0, 1.0, 1.0]
    end   = [1.0, 1.0, 1.0]
    n1.create_gradient(from_value=start, to_value=end)
    n1()
    print('\n', n1[:])
    n1.close_()
    n2.close_()



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

    neo = Neopixel(11, gamma_func=GAMMA['srgb'])
    neo.to(neo1dev).clear()
    neo.to(neo2dev).clear()
    # Create a brightness gradient
    #neo.create_gradient((0.0, 0.0, 0.0),(0.0, 0.0, 1.0)) # brightness gradient
    # neo.set_value([137, 149], (0.66, 1.0, 0.1))
    # neo.create_gradient(0., 1.0) # white LED gradient opposite ordered
    neo.create_gradient((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    neo()
    print()
    neo.close_()



def Rainbow(neo: Neopixel):
  
    @Every.every(0.5, n=neo) # note that the interval gets overriden in the function
    def drop(n:Neopixel):
        """Drop in some white pixels"""
        n.roll(value=1.0)() # drop a white pixel in a random interval
        drop.interval = random()

    @Every.every(0.01, n=neo)
    def roll(n:Neopixel):
        n.roll()()[-1] = 0.0
        # is equivalent to:
        # n.roll()
        # n.show() [or simply n()]
        # n[-1] = 0.0

    # Create a rainbow pattern in the default HSV space
    neo.create_gradient([0.0, 1.0, 1.0], [1.0, 1.0, 1.0])

    @Every.While(1) # repeat for 5s
    def proceed():
        drop()
        roll()

    print()


def Raindrops(neo: Neopixel):
    @Every.every(0.1)
    def drop(n: Neopixel):
        # place a random colored pixel at a random location in a random interval
        index = randint(0, n.num_pixels-1) # random position
        hue = random() # a random color in HSV color space
        n(index, (hue, 1.0, 1.0))
        drop.interval = random()/5

    @Every.every(1.0)
    def dropW(n:Neopixel):
        # place a white pixel at a random location every second
        index = randint(0, n.num_pixels-1) # random position
        value = random() # a random color in HSV color space
        n(index, value)

    @Every.While(1, n=neo) # repeat for 5s
    def proceed(n:Neopixel):
        drop(n)
        dropW(n)
        n *= 0.98 # pixel decay
        n()

    print()


def light_show():
    neo = Neopixel(150).to(neo2dev)

    @Every.While(5)
    def loop():
        Rainbow(neo)
        Raindrops(neo)
        neo2dev.reversed = not neo2dev.reversed

    neo.close_()


def power_measure():
    neo = Neopixel(100, color_mode=ColorMode.RGB, gamma_func=None).to(neo2dev)
    neo2dev.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
    for i, _ in enumerate(range(5), start=1):
        neo[:] = 1.0/i, 1.0/i, 1.0/i, 1.0/i
        neo()
        print(f"{neo().power_consumption=}W")
        sleep(0.2)
    neo.close_()


def effects():
    """Fire and Meteor effect on 2 devices at ones"""

    neo1dev.brightness = 1.0
    neo1dev.gamma_function = GAMMA['srgb']

    neo2dev.brightness = 0.2
    neo2dev.gamma_function = GAMMA['gamma25']
    neo = Neopixel(23, brightness=1., gamma_func=GAMMA['srgb'])
    neo.to(neo1dev)
    neo.to(neo2dev)

    candle = Fire(
        neo,
        spectrum=(0.4, -0.2),
        decay_factor=(0.98, 0.92),
        spark_interval_factor=0.05,
        spark_propagation_interval=0.015
        )

    meteor = Meteor(
        neo, 
        decay_value=0.925, 
        roll_interval=0.025,
        shoot_intervall=1.5
        )
    

    #@Every.While(10)
    #def run_effects():
    while True:
        neo.reversed = False
        candle.resume()
        Every.While(10)(candle.progress)
        candle.pause()
        Every.While(2)(candle.progress) # settle for 1s

        neo.reversed = True
        meteor.resume()
        Every.While(10)(meteor.progress) # drop for 10s
        meteor.pause()
        Every.While(2)(meteor.progress) # settle for 1s


    print()



def show_image():
    """Show an image on the neopixel, assuming the pixels are aranged 32x8, line by line."""
    from PIL import Image

    neo = Neopixel(256, brightness=1.0) # This is a Neopixel screen 32x8
    neo2dev.brightness = 0.1
    neo.to(neo2dev)

    console = ConsoleSimulationDevice()
    console.split_lines = 8
    neo.to(console)

    img = Image.open('icon1.png')
    neo.display_image(0*64, np.asarray(img), zigzag=1, transpose=True)

    img = Image.open('icon2.png')
    neo.display_image(1*64, np.asarray(img), zigzag=1, transpose=True)

    img = Image.open('icon3.png')
    neo.display_image(2*64, np.asarray(img), zigzag=1, transpose=True)

    img = Image.open('icon4.png')
    neo.display_image(3*64, np.asarray(img), zigzag=1, transpose=True)
    neo()


def sclicing_test():
    v = (0.25, 0.0, 0.0, 1.0)
    neo = Neopixel(64, color_mode=ColorMode.RGB).to(neo2dev)
    print(neo.pixel_buffer.shape)
    #neo._pixel_buffer = neo.pixel_buffer.reshape([8,8,4])
    print(neo.pixel_buffer.shape)
    # set pixel at [2, 5] to v:
    neo[0:8] = v

    neo()


def graphic_simulator():
    #no, don't
    dev = GraphicSimulation(horizontal=True, led_size=10)
    neo = Neopixel(10).to(dev)
    neo.create_gradient((0.0, 1.0, 1.0), (1.0, 1.0, 1.0))()
    dev.close_()


if __name__ == "__main__":
    Neopixel(23).to(neo1dev).clear()().close_()
    Neopixel(256).to(neo2dev).clear()().close_()

    test_custom_device()
    #basic_tests()
    #show_image()
    #GammaTest()
    #ColorModeTest()
    #power_measure()
    #light_show()
    #effects()
    #graphic_simulator() <- don't
    sclicing_test()
