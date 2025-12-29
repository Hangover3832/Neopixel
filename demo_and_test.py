from time import sleep, monotonic
from matplotlib import axis
import numpy as np
from numpy.typing import NDArray
from neopixel_classes import Neopixel
from devices import ConsoleSimulationDevice, NeopixelDevice
from colors import ColorMode, PixelOrder, create_gamma_function, G, SOME_COLORS
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
from random import random, randint
from effects import Fire, Meteor


try:
    from rpi_devices import GPIOzeroOutputDevice, RpiSpiDev

    CS1 = None # GPIOzeroOutputDevice(bcm_pin=12) # use bcm_pin 12 for chip select
    CS2 = None # GPIOzeroOutputDevice(bcm_pin=13) # use bcm_pin 12 for chip select
    neo1 = RpiSpiDev(device=0, custom_cs=CS1, pixel_order=PixelOrder.GRB)
    neo2 = RpiSpiDev(device=1, custom_cs=CS2, pixel_order=PixelOrder.GRBW)

except ModuleNotFoundError:
    print("""
            Note: The python libraries 'gpiozero' and/or 'spidev' could not be imported.
            Using dummy devices on a non Rapberry PI system to simulate a Neopixel stripe in the console.
            """)


    CS1 = None
    CS2 = None
    neo1 = ConsoleSimulationDevice(pixel_order=PixelOrder.GRB)
    neo2 = ConsoleSimulationDevice(pixel_order=PixelOrder.GRBW)


#------------------------------------------------------------------------------
class CustomDevice(NeopixelDevice):
    """Example how to implement a custom Neopixel device and passing custom parameters.
    This one simply prints content of the whole 8bit RGB array"""
    def write_to_device(self, buffer:NDArray[np.float32]) -> int:
        print("example Neopixel device implementaion")
        print(f"There are {self.neopixel.num_pixels} {self.pixel_order.name} pixels")
        print(f"{self.kwargs=}")
        print(self._to_uint8(buffer))
        return 0

Neopixel(num := 10, color_mode=ColorMode.RGB).to(CustomDevice(pixel_order=PixelOrder.RGBW, param1=1234, param2=5678)).set_value(0, np.random.rand(num, 4))()
#------------------------------------------------------------------------------



def basic_tests():
    n1 = Neopixel(10, color_mode=ColorMode.RGB).to(neo1)
    n2 = Neopixel(10, color_mode=ColorMode.RGB).to(neo2)
    
    n1[:] = 0.3
    n1[0] = 0.2, 0.3, 0.4

    n2[:] = 1.0
    n2[0] = 0.2, 0.3, 0.4
    n2[1] = 0.6, 0.7, 0.8, 0.0

    n1()
    print('\n', n1[:])
    n2()
    print('\n', n2[:])
    print()

    print("RGB:")
    n2.clear()
    n2[0] = 1,0, 0.0, 0.0
    n2[1] = 0.0, 1.0, 0.0
    n2[2] = 0.0, 0.0, 1.0
    n2.set_value(3, (0.0, 1.0, 1.0), color_mode=ColorMode.HSV)
    n2.set_value(4, (0.333, 1.0, 1.0), color_mode=ColorMode.HSV)
    n2.set_value(5, (0.666, 1.0, 1.0), color_mode=ColorMode.HSV)
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
    n2[4] = a # broadcast to index 4
    print('\n', n2()[:])
    print()

    print("Rainbow gradient:")
    n1.clear().color_mode=ColorMode.HSV
    start = [0.0, 1.0, 1.0]
    end   = [1.0, 1.0, 1.0]
    n1.create_gradient(from_value=start, to_value=end)
    n1()
    print('\n', n1[:])


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


def GammaTest() -> None:

    neo = Neopixel(150).to(neo2)
    # Create a brightness gradient
    for i in neo:
        neo[i] = 0.0, 0.0, i/(neo.num_pixels-1)
    if neo.device.is_simulated:
        print()

    neo()


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

    neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
    
    # Create a rainbow pattern in the default HSV space
    neo.create_gradient([0.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    #for i in neo:
    #    neo[i] = (i/(neo.num_pixels-1), 1.0, 1.0)


    @Every.While(5, n=neo) # repeat for 5s
    def proceed(n:Neopixel):
        drop()
        roll()


def Raindrops(neo: Neopixel):
    neo.gamma_func = G.linear.value
    
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

    @Every.While(30, n=neo) # repeat for 30s
    def proceed(n:Neopixel):
        drop(n)
        dropW(n)
        n *= 0.98 # pixel decay
        n()


def light_show():
    neo = Neopixel(150).to(neo2)
    while True:
        Rainbow(neo)
        Raindrops(neo)
        neo.reversed = not neo.reversed


def power_measure():
    lin_gamma = create_gamma_function(np.array([0.0, 1.0]))
    neo = Neopixel(100, color_mode=ColorMode.RGB, gamma_func=lin_gamma).to(neo2)
    neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
    neo[:] = 1.0, 1.0, 1.0, 1.0
    neo().clear()()
    print(f"{neo().power_consumption=}")


def fire():
    candle1 = Fire(
        Neopixel(23, brightness=0.25).to(neo1),
        spectrum=(0.8, 0.0),
        decay_factor=(0.95, 0.85),
        spark_interval_factor=0.05,
        spark_propagation_interval=0.01
        )

    candle2 = Fire(
        Neopixel(23, brightness=1.0).to(neo2),
        #spectrum=(1.0, 0.0),
        #decay_factor=(0.95, 0.85),
        #spark_interval_factor=0.15,
        #spark_propagation_interval=0.01
        )

    # Let the candles burn:
    while True:
        candle1.progress()
        #candle2.progress()


def meteor_shower():
    meteor = Meteor(Neopixel(23, brightness=1.0).to(neo1), 
                    decay_value=0.95, 
                    roll_interval=0.02
                    )
    while True:
        meteor.progress()


if __name__ == "__main__":
    quit()
    basic_tests()

    # One line to rule them all:
    Neopixel(1).to(ConsoleSimulationDevice(pixel_order=PixelOrder.GRB)).set_value(0, (1.0, 0.0, 0.0, 0.0), color_mode=ColorMode.RGB)()

    GammaTest()
    ColorModeTest()
    power_measure()

    Neopixel(23).to(neo1).clear()()
    Neopixel(150).to(neo2).clear()()

    # light_show()
    fire()
    # meteor_shower()
