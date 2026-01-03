from abc import ABC, abstractmethod
from random import random, randint
import numpy as np
from colors import ColorMode, G
from neopixel_classes import Neopixel
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
from typing import Callable, Tuple


class NeopixelEffect(ABC):
    def __init__(self, neopixel:Neopixel) -> None:
        self.neopixel: Neopixel = neopixel

    @abstractmethod
    def progress(self) -> 'NeopixelEffect':
        raise NotImplementedError
    
    @abstractmethod
    def pause(self) -> 'NeopixelEffect':
        raise NotImplementedError

    @abstractmethod
    def resume(self) -> 'NeopixelEffect':
        raise NotImplementedError


class Fire(NeopixelEffect):
    """
    Fire effect for the neopixel_spi library.

    :param neopixel: A Neopixel instance
    :type neopixel: Neopixel
    :param spectrum: Lower and upper bound of the color temperature (black body radiation spectrum).
        A values near 0.0 is redish, near 1.0 is blueish.
        Values beyond 0..1 are possible to narrow the spectrum down.
    :type spectrum: tuple[lower bound:float, upper bound:float]
    :param decay_factor: Determines how fast the flame decays at the bottom (1st value)
        and at the top (2nd value). A lower value leads to a faster decay.
    :type decay_factor: tuple[bottom:float, top:float]
    :param spark_interval_factor: Delay between flame spark ignitions near the bottom.
        A lower value leads to more spark ignites.
    :type spark_interval_factor: float
    :param spark_propagation_delay: How fast a ignited sparl traveles up the flame.
        A lower value means faster.
    :type spark_propagation_delay: float
    """
    def __init__(self, 
                 neopixel:Neopixel,
                 spectrum:Tuple[float, float]=(0.5, -0.2),
                 decay_factor:Tuple[float, float]=(0.95, 0.85),
                 spark_interval_factor:float=0.15,
                 spark_propagation_interval:float=0.01,) -> None:

        super().__init__(neopixel)
        self.spectrum = spectrum
        self.decay_factor = decay_factor
        self.spark_interval_factor = spark_interval_factor
        neopixel.auto_write = True
        self.index = 0
        self.start = 0
        self.end = self.neopixel.num_pixels - 1
        self.ignite_spark = Every(0.1, execute_immediately=True).do(self._ignite_spark)
        self.decay = Every(1.0/30, keep_interval=False).do(self._decay)
        self.propagate = Every(spark_propagation_interval, keep_interval=False).do(self._propagate)
        self._propagating: bool = False
        self._bright: float = 1.0


    @property
    def decay_factor(self) -> Tuple[float, float]:
        return self._decay_factor

    @decay_factor.setter
    def decay_factor(self, value: Tuple[float, float]) -> None:
        self._decay_factor = value
        self._decay_array = np.linspace(value[0], value[1], self.neopixel.num_pixels)[:, np.newaxis]


    def get_indexed_temp(self, index:int) -> float:
        """Return the color temperature based on the pixel index.
        We use only a part of the spectrum,  at the bottom hotter (more blueish)
        at the top colder (more redish) based on the self.spectrum parameter"""
        x = index / (self.neopixel.num_pixels-1)
        result = 0.1*random() + float(np.interp(x, (0.0, 1.0), self.spectrum))
        return result


    def _ignite_spark(self) -> None:
        if not self._propagating:
            self.start = randint(0, self.neopixel.num_pixels // 4) # start at the lower 4rd
            self.end = randint(self.neopixel.num_pixels - self.neopixel.num_pixels // 3, self.neopixel.num_pixels) # end at the upper 3rd
            self.index = self.start
            self._bright = 0.5 * (1 + random())


    def _propagate(self) -> None:
        if self.start <= self.index < self.end:
            self._propagating = True
            self.neopixel.set_temperature(self.index, 
                     self.get_indexed_temp(self.index), self._bright)
            self.index += 1
        else:
            self._propagating = False
            self.ignite_spark.reset().interval = self.spark_interval_factor * random()


    def _decay(self) -> None:
        self.neopixel *= self._decay_array

    def show_temperature_gradient(self) -> None:
        for i in self.neopixel:
            self.neopixel.set_temperature(i, self.get_indexed_temp(i))


    def progress(self) -> 'Fire':
        self.ignite_spark()
        self.propagate()
        self.decay()
        return self


    def pause(self) -> 'Fire':
        self.ignite_spark.pause()
        return self


    def resume(self) -> 'Fire':
        self.neopixel.clear()
        self.ignite_spark.reset().resume().execute()
        return self



class Meteor(NeopixelEffect):
    def __init__(self, 
                 neopixel: Neopixel,
                 roll_interval: float = 0.02,
                 decay_value: float = 0.9,
                 shoot_intervall: float = 2.0,
                 ) -> None:
        super().__init__(neopixel)
        neopixel.auto_write = True
        neopixel.color_mode = ColorMode.HSV
        neopixel.reversed = True
        self.decay_value = decay_value
        self.shoot:Every = Every(shoot_intervall * neopixel.num_pixels * roll_interval, execute_immediately=True, keep_interval=False).do(self._shoot)
        self.roll:Every = Every(roll_interval, keep_interval=False).do(self._roll)

    def _shoot(self) -> None:
        # self.neopixel.set_temperature(0, random())
        self.neopixel[0] = random(), 1.0, 1.0

    def _roll(self) -> None:
        self.neopixel.roll(value=self.neopixel[0] * np.array([1., 1., self.decay_value], dtype=np.float32))

    def progress(self) -> 'Meteor':
        self.shoot()
        self.roll()
        return self

    def pause(self) -> 'Meteor':
        self.shoot.pause()
        return self

    def resume(self) -> 'Meteor':
        self.neopixel.clear()
        self.shoot.reset().resume().execute()
        return self
