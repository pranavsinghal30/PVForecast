import json
import numpy as np
from datetime import datetime, timezone
import requests
import sys
import pandas as pd
import numpy  as np
import warnings
import pvlib
from pvlib.pvsystem    import PVSystem
from pvlib.location    import Location
from pvlib.modelchain  import ModelChain
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, message=r'.*highly experimental.*')
    from pvlib.forecast    import ForecastModel
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import pandas as pd
import numpy  as np

import sys


import warnings
import pvlib
from pvlib.pvsystem    import PVSystem
from pvlib.location    import Location
from pvlib.modelchain  import ModelChain
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, message=r'.*highly experimental.*')
    from pvlib.forecast    import ForecastModel
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib             import irradiance


ModuleName        = "LG_Electronics_Inc__LG325N1W_V5"        # select your panel (or same Wp as yours, eg. 325Wp)
InverterName      = "SMA_America__SB10000TL_US__240V_"      # select an inverter comparable name plate power (eg. 10kW)
NumStrings        =  "2"                                    # number of strings 
NumPanels         =  "15"                                    # number of panels per string
API_KEY            = "775d7fd2cd57f54d5e5daeba471f39f3"           
# ----------------------------------------------------- PVWatts definition
InverterPower     = "10000"                                  # name-plate inverter max. power
NominalEfficiency = "0.965"                                  # nominal European inverter efficiency
SystemPower       = "9750"                                  # system power [Wp]
TemperatureCoeff  = "-0.0036"                                # temperature coefficient (efficiency loss per 1C)

# ----------------------------------------------------- orientation of solar panels
# location (Latitude, Longitude) defined in [DEFAULT] section above
Tilt              =  "30"
Azimuth           = "127"    
NominalEfficiency  =  '0.96'                       # nominal inverter efficiency, default of pvwatts model
TemperatureCoeff   = '-0.005'                      # temperature coefficient of module, default of pvwatts model
TemperatureModel   = 'open_rack_glass_glass'       # https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.temperature.sapm_cell.html
Clearsky_model     = 'simplified_solis'            # default clearsky model
Altitude           = '0'                           # default altitude sea level
Model              = 'CEC'                         # default PV modeling stratey

class Forecast:
    """Abstract class of forecast data structures"""

    def __init__(self):
        self.DataTable    = None                                                         # Pandas dataframe with most recent read weather data
        self.IssueTime    = None                                                         # Issue time of weather data forecast (string, iso format, UTC)
        self.SQLTable     = None                                                         # SQL table name to be used for storage (see DBRepository.loadData())
        self.InfluxFields = []                                                           # fields to export to InfluxDB
        self.csvName      = None
        self.storePath    = None

    def get_ParaNames(self):                                                             # get parameter names of self.DataTable
        return(list(self.DataTable))

    def writeCSV(self):                                                                  # write self.DataTable to .csv file
        if self.csvName is not None and self.storePath is not None:
            try:
                self.DataTable.to_csv(self.storePath + "/" + self.csvName, compression='gzip')

            except Exception as e:
                print("writeCSV: " + str(e))
                sys.exit(1)
        else:
            print("writeCSV: csvName or storePath not defined")
class PVModel(Forecast):
    """Model PV output based on irradiance or cloud coverage data"""

    def __init__(self, Latitude,Longitude,section = 'PVSystem'):
        """Initialize PVModel
        config      configparser object with section [<section>]
                    <section> defaults to 'PVSystem'"""

        try:
            self._pvversion = pvlib.__version__
            if self._pvversion > '0.8.1':
                print("Warning --- pvmodel not tested with pvlib > 0.8.1")
            elif self._pvversion < '0.8.0':
                raise Exception("ERROR --- require pvlib >= 0.8.1")
            super().__init__()
            self._cfg   = section

            self._location   = Location(latitude  = float(Latitude),
                                      longitude = float(Longitude), 
                                      altitude  = float(Altitude),
                                      tz='UTC')  # let's stay in UTC for the entire time ...                              
            self._pvsystem          = None                                               # PV system, once defined with init_CEC() or init_PVWatts()
            self._mc                = None                                               # Model chain, once defined in init_CEC() or init_PVWatts()
            self._weather           = None                                               # weather data used for getIrradiance() and runModel()
            self._cloud_cover_param = None                                               # weather data parameter used for cloud coverage (see _weatherFields)
            self.irradiance_model   = None                                               # model name if irradiance data calculated in getIrradiance()
            self.irradiance         = None                                               # calculated irradiance data
            self.pv_model           = None                                               # CEC or PVWatts once solar system is defined
            # assuming the SQLTable will be "pvsystem"
            #self.SQLTable           = self._cfg.lower()                                  # which SQL table name is this data stored to (see DBRepository.loadData())
            #self.SQLTable = "pvsystem"
            
            #self.storePath          = self.config[self._cfg].get('storePath')            # where to store .csv file
            self.storePath = "./temp/"

            self._init_CEC()
        except Exception as e:
            print("pvmodel __init__: " + str(e))
            #sys.exit(1)

    def _init_CEC(self):
        """Configure PV system based on actual components available in pvlib CEC database"""

        try:
            moduleName     = ModuleName                        #self.config[self._cfg].get('ModuleName')
            inverterName   = InverterName                      #self.config[self._cfg].get('InverterName')
            tempModel      = TemperatureModel                  #self.config[self._cfg].get('TemperatureModel')
            self._pvsystem = PVSystem(surface_tilt                 = float(Tilt),                #self.config[self._cfg].getfloat('Tilt'),
                                      surface_azimuth              = float(Azimuth),             #self.config[self._cfg].getfloat('Azimuth'),
                                      module_parameters            = pvlib.pvsystem.retrieve_sam('cecmod')[moduleName],
                                      inverter_parameters          = pvlib.pvsystem.retrieve_sam('cecinverter')[inverterName],
                                      strings_per_inverter         = int(NumStrings),          #self.config[self._cfg].getint('NumStrings'),
                                      modules_per_string           = int(NumPanels),           #self.config[self._cfg].getint('NumPanels'),
                                      temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm'][tempModel])
            self._mc       = ModelChain(self._pvsystem, self._location, aoi_model='physical', spectral_model='no_loss')
            self.pv_model  = 'CEC'
        except Exception as e:
            print("init_CEC: " + str(e))
            #sys.exit(1)


    def getIrradiance(self, weather: Forecast, model='disc'):
        """Get irradiance data from weather files (see DWDForecast()) using various models
        weather             object eg. created by DWDForecast.weatherData
                            must contain: weatherData.DataTable, weatherData.IssueTime
        model               one of: 'disc', 'dirint', 'dirindex', 'erbs'     (GHI decomposition models)
                                    'erbs_kt'                                (as 'erbs', but with kt as input parameter; this needs a minor
                                                                              modification to pvlib.irradiance.erbs)
                                    'campbell_norman', 'clearsky_scaling'    (cloud coverage to irradiance)
                                    'clearsky'                               (clear sky model)
        cloud_cover_param   name of cloud cover parameter in weather"""

        try:                                                                   # if weather is a Forecast object, this will work
            weatherData    = weather.DataTable                                       # extract weather data table from weather
            self.IssueTime = weather.IssueTime
            #f              = self._weatherFields[weather.SQLTable]                   # field mapping dicionary weather --> pvlib
            f = { 'temp_air'   : 'temp',                                             # translation: OWM parameter names --> pvlib parameter names
                                            'wind_speed' : 'wind_speed',                 #    Note that temp_air and temp_dew are in Celsius, TTT in Kelvin
                                            'pressure'   : 'pressure',
                                            'temp_dew'   : 'dew_point',
                                            'clouds'     : 'clouds' }
            if (model == 'clearsky'):
                clearsky_model  = Clearsky_model        #self.config[self._cfg].get('clearsky_model')
                self.irradiance = self._location.get_clearsky(weatherData.index,         # calculate clearsky ghi, dni, dhi for clearsky
                                                              model=clearsky_model)
            elif (model == 'clearsky_scaling' or model == 'campbell_norman'):
                if model == 'campbell_norman' and self._pvversion == '0.8.0':
                    raise Exception("ERROR --- cloud based irradiance model 'campbell_norman' only supported in pvlib 0.8.1 and higher")
                fcModel = ForecastModel('dummy', 'dummy', 'dummy')                       # only needed to call methods below
                
                #weatherData.to_csv("weatherData"+model+".csv")
                fcModel.set_location(latitude=self._location.latitude, longitude=self._location.longitude, tz=self._location.tz)
                self.irradiance = fcModel.cloud_cover_to_irradiance(weatherData[f['clouds']], how = model)
            else:
                raise Exception("ERROR --- incorrect irradiance model called: " + model)
        except Exception as e:
            print("getIrradiance: " + str(e))
            raise e
            #sys.exit(1)        
        self.irradiance_model       = model
        try:
            self.irradiance         = pd.concat([weatherData[f['temp_air']] - 273.15, weatherData[f['wind_speed']], self.irradiance], axis=1)
            self.irradiance.rename(columns={f['temp_air'] : 'temp_air', f['wind_speed'] : 'wind_speed'}, inplace=True)
            self._cloud_cover_param = f['clouds']
        except:
            pass

    def runModel(self, weather: Forecast, model, modelLst = 'all'):
        """Run one PV simulation model (named in self.pv_model, set in getIrradiance())
        Weather data is inherited from prior call to getIrradiance() call
        Populates self.sim_result      pandas dataframe with simulation results"""

        try:
            model = model.lower()
            self.getIrradiance(weather, model)
            self._mc.run_model(self.irradiance)
            cols = ['ghi', 'dni', 'dhi']
            # CEC
            self.DataTable = pd.concat([self._mc.dc.p_mp, self._mc.ac, self.irradiance[cols]], axis=1)
            m                  = self.irradiance_model
            if (m == 'clearsky_scaling' or m == 'campbell_norman'):
                m = m + '_' + self._cloud_cover_param
            self.DataTable.columns = ['dc_' + m, 'ac_' + m, 'ghi_' + m, 'dni_' + m, 'dhi_' + m]
            self.InfluxFields.append('dc_' + m)
            return self.DataTable

        except Exception as e:
            print("runModel: " + str(e))
            raise e
            #sys.exit(1)

    def run_allModels(self, weather: Forecast, modelLst = 'all'):
        """Run all implemented models (default). Alternatively, 'modelLst' can contain a 
        comma separated list of valid models (see self.runModel()) to be calculated
        
        Populates self.DataTable   pandas dataframe with all simulation results"""

        dfList = []                                                                      # list of calculated models
        dfList.append(self.runModel(weather, 'clearsky_scaling', modelLst))              # ---- cloud based models
        if self._pvversion >= '0.8.1':                                                   # deprecated model 'liujordan' not implemented
            dfList.append(self.runModel(weather, 'campbell_norman', modelLst))
        dfList.append(self.runModel(weather, 'clearsky', modelLst))
        dfList.append(self._mc.solar_position.zenith)                                    # ---- add solar position
        self.DataTable = pd.concat(dfList, axis=1)
        drop           = []
        haveGHI        = False
        for col in self.DataTable:
            if 'ghi' in col and not (col.startswith('ghi_clearsky') or col.startswith('ghi_campbell')):
                if not haveGHI:
                    self.DataTable.rename(columns = {col: 'ghi'}, inplace=True)          # rename first GHI field as GHI, since this is input and hence same for all models
                    haveGHI = True
                else: drop.append(col)
        if (len(drop) > 0): self.DataTable = self.DataTable.drop(drop, axis=1)
        

class OWMForecast(Forecast):
    """Class for downloading weather data from openweathermap.org"""
    def __init__(self,lat,lng):
        """Initialize DWDForecast
        config      configparser object with section [OpenWeatherMap]"""

        super().__init__()
        #self.config    = config
        self.SQLTable  = 'owm'
        self.storePath = "./temp"
        self.latitude = lat
        self.longitude = lng

    def getForecast_OWM(self):
        try:
            latitude  = self.longitude
            longitude = self.latitude
            altitude = Altitude
            apikey    = API_KEY
            url = 'https://api.openweathermap.org/data/2.5/onecall?lat=' + latitude + '&lon=' + longitude + '&exclude=minutely,daily,alerts&appid=' + apikey
            req = requests.get(url)
            if (req.reason != 'OK'):
                raise Exception("ERROR --- Can't fetch OpenWeatherMap data from '" + url + "' --- Reason: " + req.reason)
            df                = pd.DataFrame(req.json()['hourly'])
            df_idx            = pd.to_datetime(df['dt'], unit='s', utc=True)
            df                = df.set_index(df_idx)
            df.index.name     = 'PeriodEnd'
            drop              = []
            for field in ['dt', 'weather', 'rain', 'snow', 'wind_gust']:
                if field in df: drop.append(field)
            self.DataTable    = df.drop(drop, axis=1)                                    # drop columns which are either not useful or non-float
            self.IssueTime    = str(datetime.fromtimestamp(req.json()['current']['dt'], timezone.utc))
            self.csvName      = 'owm_' + self.IssueTime[:16].replace(' ', '_').replace(':', '-') + '.csv.gz'

        except Exception as e:
            print("getForecast_OWM: " + str(e))
            
            
def handler(event, context):
    Latitude = event['Latitude']
    Longitude = event['Longitude']
    myWeather = OWMForecast(Latitude,Longitude)
    myWeather.getForecast_OWM()
    issue_time = datetime.fromisoformat(myWeather.IssueTime)
    myPV   = PVModel(Latitude,Longitude)
    model = "all"
    myPV.run_allModels(myWeather, model)
    start_time = myWeather.DataTable.index[0].astimezone('Asia/Kolkata').strftime("%Y-%m-%d  %H:%M:%S")
    end_time = myWeather.DataTable.index[-1].astimezone('Asia/Kolkata').strftime("%Y-%m-%d  %H:%M:%S")
    result = dict(myPV.DataTable.drop("zenith",axis=1).sum())
    result['start_time'] = start_time
    result['end_time'] = end_time
    result
    #res = int(np.sqrt(inp))

    return {
        'statusCode': 200,
        'body': result
    }