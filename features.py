
_vi_features = [
    'ndvi_noir_rgb_method', 'gndvi_noir_rgb_method', 'rdvi_noir_rgb_method',
    'savi_noir_rgb_method', 'sr_noir_rgb_method', 'evi_noir_rgb_method',
    'cigreen_noir_rgb_method', 'gli', 'vari'
]

_weather_features = [
    'Solar', 'Precipitation', 'AirTemp', 'Vapor_Pressure', 'AtmPressure', 'RelHumidity'
]

_canopy_temp_features = [
    'lepton_mean_temp', 'lepton_min_temp', 'lepton_max_temp'
]

_output_variable = 'GYLD_kg_m2'



def get_vi_features(exclude_rgb_vis = False):
    return _vi_features[:-2] if exclude_rgb_vis else _vi_features

def get_weather_features(exclude= False):
    return [] if exclude else _weather_features

def get_canopy_temp_features(exclude = False):
    return [] if exclude else _canopy_temp_features

def get_output_variable():
    return _output_variable



