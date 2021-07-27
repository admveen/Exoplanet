import numpy as np
import pandas as pd
import requests
import os.path

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML
from copy import deepcopy

# one method for curve smoothing to approximate transit. hp is a second-order trend filter.
# this may be useful if we decide to convert data to images and use images for transit classification.
from statsmodels.tsa.filters.hp_filter import hpfilter

# tsfresh for extracting some relevant time series features from phased curves.
from tsfresh.feature_extraction import feature_calculators

#some scipy stats stuff and peak-finding for weak secondary transits
from scipy.stats import ttest_ind, norm
from scipy.signal import find_peaks

import logging

# this is the base path for the EXOMAST API 
base_url = "https://exo.mast.stsci.edu/api/v0.1/"

#kepler data validated time series
kepler_dv_url = base_url + "dvdata/kepler/"

# kepler cumulative table
caltech_KOI_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative'

#logger configuration
logging.basicConfig(filename = "..\\data\\external\\DVSeries\\download.log", level = logging.WARNING)

class KOIObject():
    

    def __init__(self, KICID, tce_input = 1):


        self.kicid = KICID

        # this corresponds to the TCE planetary number in the Kepler cumulative table

        # if no argument specified, defaults to first threshold crossing event (tce_index = 1).
        self.tce_index = tce_input

    
        self.total_TCE_num = len(self.list_all_TCE())

        # Full data table with data validated light curves for given TCE. 
        # Contains initial corrected light curves as well as whitened and median detrended versions of light curve.
        # Constructor auto-initializes to empty. Need to call load_data() method to fill this.

        self.full_datatable = None

        
        # metadata for TCE extracted by Kepler autofitting pipeline. 
        #  Many of the pipeline values are not exactly the same as in the cumulative table (entries there are fit much more carefully).
        # BUT period, duration, and depth are generally fit pretty decently. 
        # Transit period here generally agree almost exactly with cumulative table. Thus these can be used for record linkage, if need be.

        self.period = None
        self.duration = None
        self.depth = None


    #----------------------------DATA LOAD METHODS FOR GIVEN TCE------------------------------------------
    
    # total initialize can take a little time. chains load_data and load_metadata
    # if source is local, pulls data from local data dv folder
    # if source is remote, pulls data from MAST API request.

    def total_initialize(self, source = 'local'):
        self.load_data(source = source).load_metadata().compute_phase_parity()

        return self

        #default is TCE 1. Probably won't need this method directly. 
        #returns full dataframe for integer indexed TCE (from the TCE_list) for a given KOI.

    def load_data(self, source = 'local'):

        if source == 'remote':
            lc_data_url = kepler_dv_url + str(self.kicid) + '/table/?tce=' + str(self.tce_index)
            lcrequest = requests.get(lc_data_url)
            lightcurve_json= lcrequest.json()
            # curves are in data key in json. convert to pandas df.
            lcdata = pd.DataFrame(lightcurve_json['data'])
            # subset on a part of all the available data columns
            cols_to_keep = ['TIME', 'PHASE', 'LC_INIT', 'LC_DETREND']
            self.full_datatable = lcdata.get(cols_to_keep)
        elif source == 'local':
            self.full_datatable = pd.read_csv("..\\data\\external\\DVSeries\\" + str(self.kicid) + "_" + str(self.tce_index) + ".csv")

        return self

    
    def load_metadata(self):
        tcemeta_url = kepler_dv_url + str(self.kicid) + '/info/?tce=' + str(self.tce_index)
        metadata_req = requests.get(tcemeta_url)

        metadata = metadata_req.json()['DV Data Header']

        # in rare cases, some of the metadata keys are missing. We can treat these cases by doing a lookup on the cumulative table.
        #the cumulative table has all the period, duration, and depth values.


        # the keys for meta_dict are column names for making table requests to the kepler cumulative table API. The values are results of the request to the DV pipeline

        meta_dict = {'koi_period': metadata.get('TPERIOD'), 'koi_duration': metadata.get('TDUR'), 'koi_depth': metadata.get('TDEPTH')}

        # short enough of a dict that I don't feel bad doing what I'm about to do:

        
        for meta_key, meta_value in meta_dict.items():
            if meta_value is None:
                req_config_string = "&select=" + meta_key + "&where=kepid=" + str(self.kicid) + "&format=json"
                r = requests.get(caltech_KOI_url + req_config_string)
                meta_dict.update(r.json()[0])
        
        # now assign values to class attributes
        self.period = meta_dict.get('koi_period')
        self.duration = meta_dict.get('koi_duration')
        self.depth = meta_dict.get('koi_depth')
        
        return self
    

    # the KOI data validation pipeline doesn't distinguish between primary TCEs and secondary eclipse events
    # if the secondary event is above the threshold detection statistic.

    #The result is that the period extracted here is half the period listed in the Kepler cumulative table.

    #thus alternating cycles in the PHASE column could potentially correspond to primary and secondary eclipses
    # of a eclipsing binary false positive. We need to thus have a flag for whether a point belong to "even" or "odd" phase cycle.

    def compute_phase_parity(self):
        self.full_datatable['CYCLE_NUM'] = (self.full_datatable['TIME'] // self.period) # cycle number
        self.full_datatable['PHASE_PARITY'] = (self.full_datatable['TIME'] // self.period) % 2

        return self
    
    #-------------------------------DOWNLOAD DATA----------------------------------------------------
    def download_data(self):
        dest_path = "..\\data\\external\\DVSeries\\" + str(self.kicid) + "_" + str(self.tce_index) + ".csv"
        # load data from the remote source if the KIC/TCE data hasn't already been downloaded
        if os.path.isfile(dest_path):
            pass
        else:
            self.load_data(source = 'remote')

            if self.full_datatable is None:
                logging.warning(str(self.kicid) + "_" + str(self.tce_index))
            else:
                # output the data table to csv
                output_df = self.full_datatable
                output_df.to_csv(dest_path, index = False)



    #-------------------------------TIME SERIES PROCESSING-------------------------------------------

    def phase_binned(self, bin_width = None, parity = 'all'):

            # sampling interval in days
    
        if bin_width == None:
            #use Kepler sampling interval as default bin width for phase (which is in days)
            smplng_intvl = self.full_datatable['TIME'].diff().mean()
            bw = smplng_intvl
        else:
            bw = bin_width
        
        # extracts phase folded light curve from data table. 

        #phased light curve can be taken from entire scan or from odd or even transits

        if parity == 'all':
            phasedLC = self.full_datatable.groupby('PHASE').median().sort_index().loc[:, 'LC_DETREND']
            
        elif parity == 'even':
            phasedLC = self.full_datatable.groupby(['PHASE_PARITY','PHASE']).median().sort_index()['LC_DETREND'].loc[0]
        elif parity == 'odd':
            phasedLC = self.full_datatable.groupby(['PHASE_PARITY','PHASE']).median().sort_index()['LC_DETREND'].loc[1]

        
        phase_range = phasedLC.index.max() - phasedLC.index.min()
        bins = round(phase_range/bw)


        #convert this to dataframe for further manipulation.
        phasedLC_df = phasedLC.to_frame()
        phasedLC_df['phase_bin'] = pd.cut(phasedLC.index, bins)

        # gets midpoint of each phase bucket
        phasedLC_df['phase_mid'] = phasedLC_df['phase_bin'].apply(lambda x: x.mid)

        return phasedLC_df


    # automatically bins light curve and averages within each bin, checks for NaNs and interpolates
    def phase_binned_avg(self, bin_width = None, parity = 'all') : # bin width in days


        # bin average the detrended light curve 
        phase_binned_avg = self.phase_binned(bin_width = bin_width, parity = parity).groupby('phase_mid').mean().loc[:,'LC_DETREND']

        #phase_binned has a sorted categorical index, but we want to convert this to a floating point index
        floatind = phase_binned_avg.index.astype('float')

        phase_binned_avg.index = floatind

        # there could potentially be a few nans floating around. let's interpolate linearly, backfill for
        # nans at the beginning and ffill for nans at the end:

        phase_binned_avg = phase_binned_avg.interpolate(method='linear').fillna(method = 'bfill').fillna(method = 'ffill')

        return phase_binned_avg

        # purpose of this function is to automatically get a centered close-up on primary transit of TCE.
        # CS stands for centered/short

    def phase_binned_CS(self, window_mult = None, parity = 'all', xynorm = False):
        if window_mult == None:
            # set window size to four times the duration by default
            # duration is in hours so convert to days for phase.
            delta_phase = 2*0.0417*self.duration #windowsize is 2*delta_phase
        else:
            delta_phase = window_mult*2*0.0417*self.duration

        phaseb_ser = self.phase_binned_avg(parity = parity)
        phaseCS = phaseb_ser.loc[-delta_phase: delta_phase]

        #normalize series s.t. transit min depth at -1 on y and (-.5*duration, +.5*duration) --> to (-1,1) on x, if xynorm = True  
        if xynorm == True:
            # normalize y scale
            phase_norm = phaseCS/(0 - phaseCS.min())

            # normalize x scale
            xscale = 0.0416*self.duration
            phase_norm.index = phaseCS.index/xscale

            return phase_norm
            
        else:
            return phaseCS


    # second order trend filtering acts as smoothing, adaptive spline on transit curve. 

    #phase = odd, even, all

    def trend_filter(self, scan_type = 'close', window_mult = None, parity = 'all'):
        
        if scan_type == 'close':
            x = self.phase_binned_CS(parity = parity, window_mult = window_mult).index
            y = self.phase_binned_CS(parity = parity, window_mult = window_mult).values

            cycle, trend = hpfilter(y, 0.2) #seems like an OK value for curvature penalty tuning parameter

            trendfiltered = pd.Series(data = trend, index = x)

        elif scan_type == 'full':
            x = self.phase_binned_avg().index
            y = self.phase_binned_avg().values

            cycle, trend = hpfilter(y, 2)

            trendfiltered = pd.Series(data = trend, index = x)
        else: 
            raise Exception("Check scan type definition.")
        
        return trendfiltered

    def evenodd_transit_stagger(self):
        even_phased_LC = self.phase_binned_avg(parity = 'even')
        odd_phased_LC = self.phase_binned_avg(parity = 'odd')

        # stagger odd_phased_LC index by TCE period.

        staggered_index = odd_phased_LC.index + self.period
        odd_phased_LC.index = staggered_index

        phase_staggered_LC = even_phased_LC.append(odd_phased_LC)

        return phase_staggered_LC

    # subtract primary transit by locally subtracting phase binned average from phased LC curve.
    

    def subtract_primary(self):

        primary_phase_list = self.phase_binned_CS(window_mult = 0.5).index # gets list of phases of primary transit

        phasefold_noprimary = self.phase_binned_avg() #phase binned average

        # this has the primary transit cut out of the phase folded, bin-averaged curve
        phasefold_noprimary.loc[primary_phase_list] = phasefold_noprimary.loc[primary_phase_list] - self.phase_binned_CS(window_mult = 0.5) 

        return phasefold_noprimary

    

    
    #---------------------FEATURE EXTRACTION FOR LIGHT CURVES---------------------

    # we do an t-test between the amplitudes in the even and odd transits. 
    # this is a sign of a potential eclipsing binary in cases where the binary orbit is low eccentricity 


    def even_odd_statistic(self):

        stime = self.full_datatable['TIME'].diff().mean()

        # this gets list of transit depth minima at odd phases
        list_odd_vals = self.phase_binned(parity = 'odd').loc[-stime:stime]['LC_DETREND'].dropna()

        # this gets list of transit depth minima at even phases
        list_even_vals = self.phase_binned(parity = 'even').loc[-stime:stime]['LC_DETREND'].dropna()

        # t-test assuming equal variance -- after whitening and assuming stationarity of white noise spectrum -- seems reasonable and some
        # histograms will show that this is largely true
        tstat, pval = ttest_ind(list_odd_vals, list_even_vals) 

        return pval


    def secondarypeak_detect(self):

        noprimary = self.subtract_primary()

        num_sigma = 1.0 # this sigma level is low. we're largely guaranteed to have 'peaks' at this level even on data with no secondary.

        # typically the secondary peak will be close to but not exactly the value of the primary's duration. 
        # this has to do with the orbit's deviation from circularity and orientation of the orbital plane.
        #  we thus pick 2 times the duration as a upper limit peak window. 0.0416 is the hour-to-day conversion.

        sampling_intval = pd.Series(noprimary.index).diff().mean()
        window_length = round((2*0.0416*(self.duration))/sampling_intval) 
        # gets integer peak location, and dict of prominence + left/right bounds of peak supports.
        peak_loc_array, fit_dict = find_peaks(-noprimary, prominence = num_sigma*noprimary.std(), wlen = window_length)


        # in some cases, there may be no peaks found. we need to make a rule to skirt this issue:        

        if peak_loc_array.size == 0:

            max_peak_prominence = 0
            peak_phase = 0
            Lphase = 0
            Rphase = 0
            secondary_depth_amp = noprimary.mean()
            p_obs = 0.32 # probability that observation or more extreme observation was generated by floor noise distribution at 1 sigma

            peak_dict = {'peak_phase': peak_phase, 'left_base': Lphase, 'right_base': Rphase, 'secondary_depth': secondary_depth_amp}

            peak_dict['backg_mean'] = noprimary.mean()
            peak_dict['backg_std'] = noprimary.std()
            peak_dict['p_sec'] = p_obs


        else:


            max_peak_prominence = fit_dict['prominences'].max()

            peak_number = np.where(fit_dict['prominences'] == max_peak_prominence)[0][0]

            #now we can extract peak, left/right support indices for this max peak:

            peak_index = peak_loc_array[peak_number]
            peak_Lsupport = fit_dict['left_bases'][peak_number]
            peak_Rsupport = fit_dict['right_bases'][peak_number]

            # convert to phase of scan

            peak_phase = noprimary.index[peak_index]
            Lphase = noprimary.index[peak_Lsupport]
            Rphase = noprimary.index[peak_Rsupport]

            secondary_depth_amp = noprimary.loc[peak_phase]

            peak_dict = {'peak_phase': peak_phase, 'left_base': Lphase, 'right_base': Rphase, 'secondary_depth': secondary_depth_amp}


            
            # let's just slice out the primary and secondary and use remaining values as sample to test for secondary peak significance. 

            noprimnosec = deepcopy(noprimary) # initialize 
            
            primaryLphase = -0.5*0.0416*self.duration
            primaryRphase = +0.5*0.0416*self.duration
            noprimnosec = noprimnosec.loc[:primaryLphase].append(noprimnosec.loc[primaryRphase:Lphase]).append(noprimnosec.loc[Rphase:])

            
            peak_dict['backg_mean'] = noprimnosec.mean()
            peak_dict['backg_std'] = noprimnosec.std()

            # we'll calculate how probable an observation at least as extreme as the transit depth is. We're going to assume normality.
            # Justification: 1) baseline from detrended, whitened, and bin averaged series. 2) baseline histograms show approximate normality

            #probability that fitted depth and lower could be generated by gaussian noise centered at background mean with background's std.

            p_obs = norm.cdf(peak_dict['secondary_depth'], loc = peak_dict['backg_mean'], scale = peak_dict['backg_std'] )
            peak_dict['p_sec'] = p_obs # this is likely the main feature we will use and we probably will need to log transform it. can do this later.


        return peak_dict

    # use tsfresh to extract a few general statistical features that might be useful.

    def other_feat_extract(self):

        X = self.phase_binned_avg()

        ts_complexity = feature_calculators.cid_ce(X, normalize = True) # some types of false positives can have a lot of wiggles
        ts_rms = feature_calculators.root_mean_square(X)
        ts_max = feature_calculators.maximum(X) # series with persistent high positive values in phase-folded/averaged LCs are possible false positives
        ts_min = feature_calculators.minimum(X) # eclipsing binary FPs CAN have dip magnitudes that are much larger than planetary counterparts

        other_feat_dict = {'time_complexity': ts_complexity, 'rms': ts_rms, 'max': ts_max, 'min': ts_min}

        return other_feat_dict

    # the xy-normalized transit close up is a good starting point for comparing shapes of the primary transit. 
    # unfortunately while the phase limits for the normalized transits always extend from -2 to 2 transit durations, the 
    # number of points is variable for the different KOIs. We want to create a fixed length object across all KOIs for
    # feature construction.

    # when looped over our trainset will construct a feature matrix X that is high dimensional (dim = 141). Will learn
    # dimensionality reduction techniques on this 

    def transit_normxy_fixedlength(self, bin_num = 141):

        # fix to 141 bins w/ mean agg. This number is arbitrary but seems OK.

        trans_norm_df = self.phase_binned_CS(xynorm= True).to_frame()
        trans_norm_df['bin_range'] = pd.cut(trans_norm_df.index, bins = bin_num)

        trans_grouped = trans_norm_df.groupby('bin_range').mean()

        #since all TCEs have been xy normalized to the same range and with the same number of bins, we
        # drop the bin-ranges and index by bin-number:

        trans_grouped = trans_grouped.reset_index().drop(columns = ['bin_range'])

        # In some cases, we will have down-sampled. But in other cases, we will have up-sampled
        # to account for this, let's fill in NaNs if they pop up:

        if trans_grouped['LC_DETREND'].isna().any() == True:
            trans_grouped = trans_grouped.interpolate(method = 'linear')

            
        return trans_grouped 

    # extracts all calculated features 
    def extract_allfeatures(self):

        feat_output_dict = {}
        feat_output_dict.update({'KIC_ID': self.kicid})
        feat_output_dict.update({'TCE_num': self.tce_index})

        feat_output_dict.update({'even_odd_stat': self.even_odd_statistic()})
        feat_output_dict.update({'p_secondary': self.secondarypeak_detect()['p_sec']})
        feat_output_dict.update(self.other_feat_extract())

        LC_features = self.transit_normxy_fixedlength()
        new_index = 'LCBIN_' + LC_features.index.map(str)
        LC_features.set_index(new_index, inplace=True)
        LC_feature_dict = LC_features['LC_DETREND'].to_dict()

        feat_output_dict.update(LC_feature_dict)

        return feat_output_dict





    #---------------------METHODS FOR LISTING ALL TCES FOR Kepler Object-----------------------

    # methods for listing all TCEs for a given Kepler catalog object (KIC).

    def list_all_TCE(self):
        tcelist_url = kepler_dv_url + str(self.kicid) + '/tces/'

        tcelist_req = requests.get(tcelist_url)
        tcelist = tcelist_req.json()
        
        return tcelist    

    #---------------------PLOTTING FUNCTIONS----------------------------------------------------

    # this plotting function plots the light curve time series for the given TCE

    def plot_LC(self, lctype = 'detrend', mode = 'display'):
        #options for lctype are 'initial', 'detrend'

        if lctype == 'initial':
            x_nam = 'TIME'
            y_nam = 'LC_INIT'
            pltxlabel = 'Time (Baryocentric Julian Days)'
            pltylabel = 'Relative Flux'
            pltlabel = 'Initial LC'
            plttitle = 'Initial Light Curve'
        
        if lctype == 'detrend':
            x_nam = 'TIME'
            y_nam = 'LC_DETREND'
            pltlabel = 'Detrended LC'
            pltxlabel = 'Time (Baryocentric Julian Days)'
            pltylabel = 'Relative Flux'
            plttitle = 'Median Detrended Light Curve'

        lcfig = sns.scatterplot(x = x_nam, y = y_nam, data = self.full_datatable, s = 2, label = pltlabel)
        plt.ylabel(pltylabel)
        plt.xlabel(pltxlabel)
        plt.title(plttitle)
        plt.legend()
        if mode == 'display':
            plt.show()
        elif mode == 'save':
            return lcfig

    
    # plots the phase-folded, bin-averaged light curve zoomed in on the primary transit


    def plot_phasefolded(self, edge_color = 'yellow', marker = 'o', parity = 'all', mode = 'display'):
        phblongseries = self.phase_binned_avg(parity = parity)
        phasefoldfig = sns.lineplot(x = phblongseries.index, y = phblongseries.values, marker = marker, label = 'KIC: ' + str(self.kicid) + "\nTCE:" + str(self.tce_index) )
        plt.ylabel('Relative Flux'), 
        plt.xlabel('Phase (days)')
        plt.title('Phased Folded, Phased-Binned LC'  )
        if mode == 'display':
            plt.show()
        elif mode == 'save':
            return phasefoldfig

    # plots the phase-folded, bin-averaged light curve zoomed in on the primary transit
    #options for showing data with trend filtering or to just show trend filtered curve with no data
    # also options for plotting even or odd phase closeups

    def plot_transit_closeup(self, trendonly = False, window_mult = None, parity = 'all', edge_color = 'yellow', marker = 'o', marker_size = 80, mode = 'display'):
        if trendonly == False:
            phbseries = self.phase_binned_CS(parity = parity, window_mult = window_mult)
            transclosefig = sns.scatterplot(x = phbseries.index, y = phbseries.values, marker = marker, edgecolor = edge_color, s = marker_size, label = 'KIC: ' + str(self.kicid) + "\nTCE:" + str(self.tce_index) )
            self.trend_filter(parity = parity, window_mult = window_mult).plot(c = 'r',label = 'L2 Trend Filter')
            plt.legend()
        elif trendonly == True:
            self.trend_filter(window_mult = window_mult).plot(c = 'r')

        
        if parity == 'even':
            plt.title('Even Transit Closeup: Phase-Binned Avg.')
        elif parity == 'odd':
            plt.title('Odd Transit Closeup: Phase-Binned Avg.')
        else:
            plt.title('Primary Transit Closeup: Phase-Binned Avg.')

        
        plt.ylabel('Relative Flux')
        plt.xlabel('Phase (days)')

        if mode == 'display':
            plt.show()
        elif mode == 'save':
            return transclosefig


    # simple plot with x and y axis scaled for diagnostics on primary transit close up and i'll potentially modify later to make images to be used as input features for CNN 
    def plot_transitcloseup_scaled(self, window_mult = None, mode = 'display'):
        trans_norm = self.phase_binned_CS(xynorm=True, window_mult = window_mult)
        closeupscaledplot = sns.lineplot(x = trans_norm.index, y = trans_norm.values)
        plt.ylabel('Flux Scaled to Transit Depth')
        plt.xlabel('Time [Transit Durations]')
        if mode == 'display':
            plt.show()
        elif mode == 'save':
            return closeupscaledplot

    
    # plots even and odd transits staggered to look at potential primary and secondary eclipse in series
    def plot_oddandeven_transit(self, mode = 'display'):


        oddevenstaggerplot = self.evenodd_transit_stagger().plot()
        
        plt.xlabel('Phase (days)')
        plt.ylabel('Relative Flux')
        plt.title('Even and Odd Transits: Phase-Bin Averaged')
        plt.annotate('Even Phase', xy = (0,0))
        plt.annotate('Odd Phase', xy = (self.period, 0))
        if mode == 'display':
            plt.show()
        elif mode == 'save':
            return oddevenstaggerplot


    # visualize potential secondary transit and 'dip' left phase, right phase, estimated peak phase location.

    def plot_secondary(self, mode = 'display'):

        no_primary = self.subtract_primary()
        peak_dict = self.secondarypeak_detect()

        left_base = peak_dict['left_base']
        right_base = peak_dict['right_base']


        secondary_amp = peak_dict['secondary_depth']
        peak_phase = peak_dict['peak_phase']

        no_primary.plot(linewidth=2)
        secondaryplot = plt.scatter(x = peak_phase, y = secondary_amp, marker = '^', s = 100, c = 'r')
        plt.axvline(left_base, c = 'r', linestyle = '--')
        plt.axvline(right_base, c = 'r', linestyle = '--')

        plt.xlabel('Phase (days)')
        plt.ylabel('Relative Flux')
        plt.title('Secondary Peak Visualization')
        if mode == 'display':
            plt.show()
        elif mode == 'save':
            return secondaryplot



        

 


