class Weather(object):

    def __init__(self, weather, cfg) -> None:
        """Customed weather controller

        Args:
            weather (carla weather): object
            cfg (hydra config): config
        """
        self.weather = weather

        self._bright = Bright()
        self._clear = Clear()
        self._cloudy = Cloudy()
        self._dawn = Dawn()
        self._evening = Evening()
        self._evening_rain = EveRain()
        self._fuzzy = Fuzzy()
        self._hazzy_rain = HazzyRain()
        self._morning_fog = MorningFog()
        self._rain = Rain()
        self._reflection = Reflection()

        current_weather = self.get_weather(cfg)

        self.weather.cloudiness = current_weather.cloud
        self.weather.precipitation = current_weather.rain
        self.weather.precipitation_deposits = current_weather.puddles
        self.weather.wind_intensity = current_weather.wind
        self.weather.fog_density = current_weather.fog
        self.weather.wetness = current_weather.wetness
        self.weather.sun_azimuth_angle = current_weather.azimuth
        self.weather.sun_altitude_angle = current_weather.altitude
        print("All weather settings done.")

    def get_weather(self, cfg):
        if cfg.weather == 'bright':
            return self._bright
        if cfg.weather == 'clear':
            return self._clear
        if cfg.weather == 'cloudy':
            return self._clear
        if cfg.weather == 'dawn':
            return self._dawn
        if cfg.weather == 'evening':
            return self._evening
        if cfg.weather == 'evening_rain':
            return self._evening_rain
        if cfg.weather == 'fuzzy':
            return self._fuzzy
        if cfg.weather == 'hazzy_rain':
            return self._hazzy_rain
        if cfg.weather == 'morning_fog':
            return self._morning_fog
        if cfg.weather == 'rain':
            return self._rain
        if cfg.weather == 'reflection':
            return self._reflection


class Bright(object):

    def __init__(self):
        self.cloud = 0.
        self.rain = 0.
        self.puddles = 40.
        self.wind = 5.
        self.fog = 0.
        self.wetness = 0.
        self.azimuth = 3.06
        self.altitude = 47.42


class Clear(object):

    def __init__(self) -> None:
        self.cloud = 25.
        self.rain = 0.
        self.puddles = 75.
        self.wind = 40.
        self.fog = 0.
        self.wetness = 0.
        self.azimuth = 61.30
        self.altitude = 35.08


class Cloudy(object):

    def __init__(self) -> None:
        self.cloud = 42.
        self.rain = 2.
        self.puddles = 85.
        self.wind = 40.
        self.fog = 0.
        self.wetness = 12.
        self.azimuth = 353.07
        self.altitude = 49.92


class Dawn(object):

    def __init__(self) -> None:
        self.cloud = 41.
        self.rain = 1.
        self.puddles = 0.
        self.wind = 40.
        self.fog = 0.
        self.wetness = 5.
        self.azimuth = 20.31
        self.altitude = 1.61


class Evening(object):

    def __init__(self) -> None:
        self.cloud = 0.
        self.rain = 0.
        self.puddles = 0.
        self.wind = 5.
        self.fog = 0.
        self.wetness = 0.
        self.azimuth = 12.23
        self.altitude = -16.12


class EveRain(object):

    def __init__(self) -> None:
        self.cloud = 90.
        self.rain = 75.
        self.puddles = 65.
        self.wind = 5.
        self.fog = 30.
        self.wetness = 100
        self.azimuth = 34.58
        self.altitude = -16.44


class Fuzzy(object):

    def __init__(self) -> None:
        self.cloud = 63.
        self.rain = 23.
        self.puddles = 85.
        self.wind = 40.
        self.fog = 13.
        self.wetness = 100.
        self.azimuth = 349.09
        self.altitude = 48.93


class HazzyRain(object):

    def __init__(self) -> None:
        self.cloud = 90.
        self.rain = 80.
        self.puddles = 85.
        self.wind = 90.
        self.fog = 30.
        self.wetness = 100.
        self.azimuth = 42.66
        self.altitude = 1.30


class MorningFog(object):

    def __init__(self) -> None:
        self.cloud = 17.
        self.rain = 40.
        self.puddles = 7.
        self.wind = 7.
        self.fog = 7.
        self.wetness = 87.
        self.azimuth = 23.48
        self.altitude = 8.25


class Rain(object):

    def __init__(self) -> None:
        self.cloud = 90.
        self.rain = 79.
        self.puddles = 85.
        self.wind = 90.
        self.fog = 30
        self.wetness = 100
        self.azimuth = 338.25
        self.altitude = 40.68


class Reflection(object):

    def __init__(self) -> None:
        self.cloud = 90.
        self.rain = 80.
        self.puddles = 85.
        self.wind = 90.
        self.fog = 30.
        self.wetness = 100.
        self.azimuth = 39.96
        self.altitude = -4.54


class Location(object):

    def __init__(self, scene_num: str) -> None:
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0
        self.pitch = 0
        self.set_location(scene_num)

    def set_location(self, scene_num: str):
        if scene_num == '01':
            self.x = -35
            self.y = 7
            self.z = 6
            self.yaw = 130
            self.pitch = -15
        if scene_num == '02':
            self.x = -16.9
            self.y = -70.8
            self.z = 6
            self.yaw = 40
            self.pitch = -15
        if scene_num == '03':
            self.x = 116.9
            self.y = -4.4
            self.z = 6
            self.yaw = 130
            self.pitch = -15
        if scene_num == '04':
            self.x = -115.6
            self.y = -3.6
            self.yaw = 62
            self.z = 6
            self.pitch = -15
        if scene_num == '05':
            self.x = -55.1
            self.y = 58.9
            self.yaw = 50
            self.z = 6
            self.pitch = -15
        if scene_num == '06':
            self.x = -32.7
            self.y = 35.5
            self.z = 6
            self.pitch = -20
            self.yaw = 225
