import numpy as np



def hist_raw_data(tics, Y, xres=100, yres=100, edge=0.1):
    '''
    this function taken from navjeet's plot_tica.py script calculates the average distance histogram as explained in JcTc work.
    #INPUTS::
	tics - [arr] top2 dimensions of data to e used for histogram. If more, top2 are taken
	Y    - [arr] distances of frames
	xres - [int] d[100] 
	yres - [int] d[100]
    edge - [float] d[0.1]
    #OUTPUTS::
	xmin      - [float] values of 
	xmax      - [float]  extents of x 
	ymin      - [float]   and y  
	ymax      - [float]    axes
	histogram - [arr]   histogram
    '''
    x_min, x_max, y_min, y_max = np.min(tics[:,0])-edge, np.max(tics[:,0])+edge, np.min(tics[:,1])-edge, np.max(tics[:,1])+edge
    x_resolution, y_resolution = xres, yres

    count = tics.shape[0]
    histogram = np.zeros([x_resolution, y_resolution])
    norm = np.zeros([x_resolution, y_resolution])

    x_interval_length = (x_max - x_min) / x_resolution
    y_interval_length = (y_max - y_min) / y_resolution

    for i in range(count):
        x = int((tics[i,:][0] - x_min) / x_interval_length)
        y = int((tics[i,:][1] - y_min) / y_interval_length)

        norm[x,y] += 1
        histogram[x,y] += Y[i]


    histogram = np.divide(histogram, norm)
    histogram = histogram - np.min(histogram[~np.isnan(histogram)])

    return x_min, x_max, y_min, y_max, histogram




def hist_mean_std(tics, Y, xres=100, yres=100, edge=0.1):
    '''
    modified version of hist_raw_data used in JcTc work, calculates both mean and standard deviations.
    #INPUTS::
	tics - [arr] top2 dimensions of data to e used for histogram. If more, top2 are taken
	Y    - [arr] distances of frames
	xres - [int] d[100] 
	yres - [int] d[100]
    edge - [float] d[0.1]
    #OUTPUTS::
	xmin      - [float] values of 
	xmax      - [float]  extents of x 
	ymin      - [float]   and y  
	ymax      - [float]    axes
	mean      - [arr]   histogram
	std       - [arr]   standard deviations
    '''
    x_min, x_max, y_min, y_max = np.min(tics[:,0])-edge, np.max(tics[:,0])+edge, np.min(tics[:,1])-edge, np.max(tics[:,1])+edge
    x_resolution, y_resolution = xres, yres

    count = tics.shape[0]
    norm = np.zeros([x_resolution, y_resolution])
    mean = np.zeros([x_resolution, y_resolution])
    std = np.zeros([x_resolution, y_resolution])

    x_interval_length = (x_max - x_min) / x_resolution
    y_interval_length = (y_max - y_min) / y_resolution

    for i in range(count):
        x = int((tics[i,:][0] - x_min) / x_interval_length)
        y = int((tics[i,:][1] - y_min) / y_interval_length)

        norm[x,y] += 1
        mean[x,y] += Y[i]
        std[x,y] += Y[i]**2

    mean = np.divide(mean, norm)
    std = np.sqrt( np.divide(std, norm) - np.square(mean) )

    return x_min, x_max, y_min, y_max, mean, std



def hist_fraction(tics, Y, cutt, xres=100, yres=100, edge=0.1):
    '''
    modified version of hist_raw_data used in JcTc work, calculates fraction of datapoints within cutt of a particular bin.
    #INPUTS::
	tics - [arr] top2 dimensions of data to e used for histogram. If more, top2 are taken
	Y    - [arr] distances of frames
    cutt - [float] cutoff of Y, less than
	xres - [int] d[100] 
	yres - [int] d[100]
    edge - [float] d[0.1]
    #OUTPUTS::
	xmin      - [float] values of 
	xmax      - [float]  extents of x 
	ymin      - [float]   and y  
	ymax      - [float]    axes
	fraction  - [arr]   histogram
    '''
    x_min, x_max, y_min, y_max = np.min(tics[:,0])-edge, np.max(tics[:,0])+edge, np.min(tics[:,1])-edge, np.max(tics[:,1])+edge
    x_resolution, y_resolution = xres, yres

    count = tics.shape[0]
    norm = np.zeros([x_resolution, y_resolution])
    fraction = np.zeros([x_resolution, y_resolution])

    x_interval_length = (x_max - x_min) / x_resolution
    y_interval_length = (y_max - y_min) / y_resolution

    for i in range(count):
        x = int((tics[i,:][0] - x_min) / x_interval_length)
        y = int((tics[i,:][1] - y_min) / y_interval_length)

        norm[x,y] += 1
        if Y[i] < cutt:
            fraction[x,y] += 1

    fraction = np.divide(fraction, norm)

    return x_min, x_max, y_min, y_max, fraction
    




def hist_range(tics, Y, mini, maxi, xres=100, yres=100, edge=0.1):
    '''
    modified version of hist_raw_data used in JcTc work, calculates fraction of datapoints within range of Y.
    #INPUTS::
	tics - [arr] top2 dimensions of data to e used for histogram. If more, top2 are taken
	Y    - [arr] distances of frames
    mini - [float] minimum and maximum cutoffs          greater than equal to
    maxi - [float]  of Y to be selected                   less  than equal to
	xres - [int] d[100] 
	yres - [int] d[100]
    edge - [float] d[0.1]
    #OUTPUTS::
	xmin      - [float] values of 
	xmax      - [float]  extents of x 
	ymin      - [float]   and y  
	ymax      - [float]    axes
	fraction  - [arr]   histogram
    '''
    x_min, x_max, y_min, y_max = np.min(tics[:,0])-edge, np.max(tics[:,0])+edge, np.min(tics[:,1])-edge, np.max(tics[:,1])+edge
    x_resolution, y_resolution = xres, yres

    count = tics.shape[0]
    norm = np.zeros([x_resolution, y_resolution])
    fraction = np.zeros([x_resolution, y_resolution])

    x_interval_length = (x_max - x_min) / x_resolution
    y_interval_length = (y_max - y_min) / y_resolution

    for i in range(count):
        x = int((tics[i,:][0] - x_min) / x_interval_length)
        y = int((tics[i,:][1] - y_min) / y_interval_length)

        norm[x,y] += 1
        if Y[i] >= mini and Y[i] <= maxi:
            fraction[x,y] += 1

    fraction = np.divide(fraction, norm)

    return x_min, x_max, y_min, y_max, fraction








def classification_extent(hist, cutoff=0.1, strategy='number', pure_type='both',  
                          impure_weight=0.5, output_type='pure-impure', 
                          jm_lower=0.5, jm_higher=None, operation=2, jm_impure=0.5):
    '''
    This function computes the classfication efficiency (CE) of the tica free energy plot.
    The hist surface represent the fractional distribution of a particular state
    on free energy surface.

    Classification efficiency:
        It considers the free energy surface discretized in bins as either pure or impure.
        pure represent the bins where the state is either 
                            >= 1-cutoff or 
                            <= cutoff
        impure represent the bins otherwise.
        Now based on different combinations of strategy and output_type, classification extent is calculated.

    #INPUTS::
        hist        - [mat] a histogram in the form of np.ndarray
                            this histogram should be in the form of fraction 
                            i.e., ranginh between [0-1], output of navjeet_hist.hist_range()
                            the empty surface should be represented by Nan values.

        cutoff      - [float] d[0.1] E [0,0.5)
                            The cutoff fraction to define the purity.

        strategy    - [str] d[number] E [number, weighted, number-higher, weighted-higher, number-lower, weighted-lower]
                            strategy to define the number of pure bins:
                                number   - directly calculating number of pure bins
                                weighted - weighted sum of pure bins, weighted by fraction
                                    pure = N+(1/N)*\Sigma{pure higher bins}{i-(1-cutoff)}  +  N+(1/N)*\Sigma{pure lower bins}{cutoff-i}
                                    Note that in case of weighted, CE can exceed 1, but returns maximum 1 only.

        output_type - [str] d[pure-impure] E [pure-impure, pure-main, pure-higher]
                            
                            pure-impure :
                                        This calculates the CE by considering both the purity and impurity on surface:
                                        CE = ( pure - impure ) / max(pure, impure)
                                        CE E [-1,0,1]
                                        +1 - perfectly pure classification
                                         0 - both pure and impure classification equally
                                        -1 - prefectly impure classification i.e., no pure region

                            pure-main   :
                                        This calculates the CE by mainly emphasizing mainly on pure nodes and less impure nodes.
                                        CE = pure / (pure+impure)
                                        CE E [0,1]
                                         0 - no pure bins
                                         1 - no impure bins

                            pure-higher :
                                        This return the number of bins constituting mainly by state. i.e., >= 1-cutoff
                                        CE = pure_higher / pure
                                        CE E [0,1]
                                         0 - no pure higher bins
                                         1 - no pure lower bins

                            jm          :
                                        Note: this superceeded some parameters
                                        thsi function treats pure-higher and pure-lower separately, such that each type of pure bin can only 
                                        contribute some part of classification extent. Ultimately for a rare state, high number of pure-lower surface
                                        cannot dominate and it limited by jm_lower.
                                        The additional parameters include:
                                            operation - d[2] [1,2,3] corresponding to output_type pure-impure, pure-main and pure-higher
                                            jm_lower  - d[0.5] [0<x<1] based on rareness of state, more rare state-more lower value
                                            jm_impure - d[0.5] [0<x<1] to control the interference of impure bins {particularly when emphasis is on 
                                                                        identifying separation between pure-higher and pure-lower surface}.

                                        **Based on operation, can yeild CE in different ranges (see above).


    #OUTPUT::
    CE  - [float] the classification extent of given surface

    '''
    if not isinstance(hist, np.ndarray)  or  np.nanmin(hist) < 0  or  np.nanmax(hist) > 1:
        raise TypeError('problem with hist')
    if not isinstance(cutoff, float) or not 0 < cutoff <= 0.5:
        raise ValueError('problem with cutoff')

    hist = hist[~np.isnan(hist)]


    pure_higher = np.where( hist >= 1-cutoff )[0]
    pure_lower  = np.where( hist <= cutoff )[0]



    if strategy == 'number':
        pure_higher = pure_higher.shape[0]
        pure_lower = pure_lower.shape[0]

    elif strategy == 'weighted':
        pure_higher = pure_higher.shape[0] + np.mean( hist[pure_higher] - (1-cutoff) )
        pure_lower  = pure_lower.shape[0] + np.mean( cutoff - hist[pure_lower] )

    else:
        raise ValueError('strategy E [number, weighted]')
    


    if pure_type == 'both':
        pure = pure_higher + pure_lower

    elif pure_type == 'higher':
        pure = pure_higher + 0

    elif pure_type == 'lower':
        pure = pure_lower + 0

    else:
        raise ValueError('pure_type E [both, higher, lower]')

        

    impure = np.where( (hist < 1-cutoff) & (hist > cutoff) )[0].shape[0]
    if 0 < impure_weight < 1.00:
        impure = impure * impure_weight 
        pure = pure * (1 - impure_weight)
    else: raise ValueError(' 0 < impure_weight < 0 ')



    if output_type == 'pure-impure':
        ce = (pure - impure) / np.max([pure, impure])

    elif output_type == 'pure-main':
        ce = pure / (pure + impure)

    elif output_type == 'impure-main':
        ce = impure / (pure + impure)

    elif output_type == 'pure-higher':
        ce = pure_higher / pure

    elif output_type == 'pure-lower':
        ce = pure_lower / pure


    elif output_type == 'jm':

        if pure_type != 'both': raise AssertionError('pure_type should be "both"')
        if jm_higher == None: jm_higher = 1-jm_lower
        if jm_lower + jm_higher > 1.00 : raise AssertionError('jm_cutoffs should not exceed 1')

        if not 0 < jm_impure < 1: raise AssertionError(' 0 < jm_impure < 1')
        pure_higher = pure_higher * (1-jm_impure)
        pure_lower = pure_lower * (1-jm_impure)
        pure = pure_higher + pure_lower
        impure = impure * jm_impure
            
        if operation == 1:
            ph = (pure_higher - impure) / np.max([pure_higher, impure])
            ph = jm_higher * (2*ph - 1)

            pl = (pure_lower - impure) / np.max([pure_lower, impure])
            pl = jm_lower * (2*pl - 1)

        elif operation == 2:
            ph = pure_higher / (pure_higher + impure)
            ph = ph * jm_higher

            pl = pure_lower / (pure_lower + impure)
            pl = pl * jm_lower
            
        elif operation == 3:
            ph = pure_higher / pure
            ph = ph * jm_higher
            pl = pure_lower / pure
            pl = pl * jm_lower

        else: raise ValuerError('operation E [1, 2, 3]')

        ce = ph + pl


    else:
        raise ValueError('output_type E [pure-impure, pure-main, impure-main, pure-higher, pure-lower, jm]')


    return ce





    

def classification_jm2(hist, cutoff=0.1, strategy='number', 
                       jm_impure=0.5, jm_lower=0.5):

    hist = hist[~np.isnan(hist)]

    ph = np.where(hist >= 1-cutoff)[0]
    pl = np.where(hist <= cutoff)[0]

    if strategy == 'number':
        ph = ph.shape[0]
        pl = pl.shape[0]

    elif strategy == 'weighted':
        ph = ph.shape[0] + np.mean( hist[ph] - (1-cutoff) )
        pl = pl.shape[0] + np.mean( cutoff - hist[pl] )

    else: raise ValueError()

    impure = np.where( (hist < 1-cutoff) & (hist > cutoff) )[0].shape[0]

    if not 0 < jm_impure < 1: raise AssertionError()
    impure = impure * jm_impure
    ph = ph * (1-jm_impure)
    pl = pl * (1-jm_impure)

    if not 0 < jm_lower < 1: raise AssertionError()
    jm_higher = 1 - jm_lower
    pl = pl / (pl+impure)
    pl = pl * jm_lower
    ph = ph / (ph+impure)
    ph = ph * jm_higher

    ce = ph + pl

    return ce


    
