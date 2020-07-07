#!/usr/bin/python3
'''
**********************************************************************
* Filename    : Servo.py
* Description : Driver module for servo, with PCA9685
* Author      : Cavon
* Brand       : SunFounder
* E-mail      : service@sunfounder.com
* Website     : www.sunfounder.com
* Update      : Cavon    2016-09-13    New release
*               Cavon    2016-09-21    Change channel from 1 to all
**********************************************************************
bus_number is the I2C bus on the nano, default i2c_0
channel is where the pwm is intalled servo
'''
import signal
import time
import PCA9685

angle = 180


class Servo(object):
    '''Servo driver class'''
    _MIN_PULSE_WIDTH = 600
    _MAX_PULSE_WIDTH = 2400
    _DEFAULT_PULSE_WIDTH = 1500
    _FREQUENCY = 60

    _DEBUG = False
    _DEBUG_INFO = 'DEBUG "Servo.py":'

    def __init__(self, channel, offset=0, lock=True, bus_number=0, address=0x40):
        ''' Init a servo on specific channel, this offset '''
        if channel < 0 or channel > 16:
            raise ValueError(
                "Servo channel \"{0}\" is not in (0, 15).".format(channel))
        self._debug_("Debug on")
        self.channel = channel
        self.offset = offset
        self.lock = lock

        self.pwm = PCA9685.PWM(bus_number=bus_number, address=address)
        self.frequency = self._FREQUENCY
        self.write(90)

    def _debug_(self, message):
        if self._DEBUG:
            print(self._DEBUG_INFO, message)

    def setup(self):
        self.pwm.setup()

    def _angle_to_analog(self, angle):
        ''' Calculate 12-bit analog value from giving angle '''
        pulse_wide = self.pwm.map(
            angle, 0, 180, self._MIN_PULSE_WIDTH, self._MAX_PULSE_WIDTH)
        analog_value = int(float(pulse_wide) / 1000000 * self.frequency * 4096)
        self._debug_('Angle %d equals Analog_value %d' % (angle, analog_value))
        return analog_value

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self.pwm.frequency = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        ''' Set offset for much user-friendly '''
        self._offset = value
        self._debug_('Set offset to %d' % self.offset)

    def write(self, angle):
        ''' Turn the servo with giving angle. '''
        if self.lock:
            if angle > 180:
                angle = 180
            if angle < 0:
                angle = 0
        else:
            if angle < 0 or angle > 180:
                raise ValueError("Servo \"{0}\" turn angle \"{1}\" is not in (0, 180).".format(
                    self.channel, angle))
        val = self._angle_to_analog(angle)
        val += self.offset
        self.pwm.write(self.channel, 0, val)
        self._debug_('Turn angle = %d' % angle)

    @property
    def debug(self):
        return self._DEBUG

    @debug.setter
    def debug(self, debug):
        ''' Set if debug information shows '''
        if debug in (True, False):
            self._DEBUG = debug
        else:
            raise ValueError(
                'debug must be "True" (Set debug on) or "False" (Set debug off), not "{0}"'.format(debug))

        if self._DEBUG:
            print(self._DEBUG_INFO, "Set debug on")
        else:
            print(self._DEBUG_INFO, "Set debug off")


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


def test():
    ena = Servo(13)
    enb = Servo(14)
    ena.setup()
    enb.setup()
    while(True):
        ena.write(180)
        enb.write(180)
        time.sleep(0.1)


if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	test()
