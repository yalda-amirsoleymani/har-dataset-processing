import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt, butter, filtfilt
import pandas as pd


class DataTransform:
    def __init__(self,dataframe):
        self.df = dataframe

    # Reminder: standardization is specific for each feature
    # don't apply for all the features together


    def calculate_vector_magnitude(self):
        df_transformed = self.df.copy()
        position_list = ["lw","lh","la","ra"]
        columns_to_calculate_VM = ["lw_x", "lw_y", "lw_z", "lh_x", "lh_y", "lh_z", "la_x", "la_y", "la_z", "ra_x", "ra_y", "ra_z"]
        for i in range(0,len(columns_to_calculate_VM),3):
            x_col = columns_to_calculate_VM[i]
            y_col = columns_to_calculate_VM[i+1]
            z_col = columns_to_calculate_VM[i+2]
            for p in position_list:
                magnitude_col = f"magnitude_{p}"
                df_transformed[magnitude_col] = np.sqrt(self.df[x_col]**2 + self.df[y_col]**2 + self.df[z_col]**2)
        df_result = df_transformed.drop(columns=columns_to_calculate_VM)  
        return df_result


    '''
        An alternative standardization is scaling features to lie 
        between a given minimum and maximum value, 
        often between zero and one, 
        or so that the maximum absolute value of each feature 
        is scaled to unit size.
        
        Robust method and prevent zero entries
        input: dataframe
        output:
    '''

    def standardization(self,vm):
        columns_to_standardize = vm.columns[2:]
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(vm[columns_to_standardize])
        vm[columns_to_standardize] = standardized_data
        return vm


    def statistical_extraction(self,activity,vm):
        results = []
        data_processed = self.standardization(vm)
        
        mean = np.mean(data_processed)
        std = np.std(data_processed)
        variance = np.var(data_processed)
        minimum = np.min(data_processed)
        maximum = np.max(data_processed)
        result = {
                'activity': f'{activity}',
                'mean': mean,
                'std': std,
                'variance': variance,
                'minimum': minimum,
                'maximum': maximum
            }

        return result


class NoiseFilter:

    def __init__(self,vm,cutoff_freq,order,sample_rate):
        self.arr = vm
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.sample_rate = sample_rate

    def _butter_lowpass(self):
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self):
        """
        Apply a low-pass Butterworth filter to the data.

        Parameters:
        ----------
        data_array : numpy.ndarray
            Input data to be filtered. If 1D, filters that array. 
            If 2D, filters along the specified columns/indices.

        Returns:
        ----------
        numpy.ndarray
            Filtered data.
        """

        if len(self.arr.shape) == 1:
            return filtfilt(b, a, self.arr)
        b,a = self._butter_lowpass()
        filtered_data = np.empty_like(self.arr)
        for idx in range(self.arr.shape[1]):
            filtered_data[:, idx] = filtfilt(b, a, self.arr[:, idx])
            
        
        return filtered_data
            
    

class Segementation:
    def __init__(self,data,window_size,overlap,sampling_rate):
        self.data = data
        self.ws = window_size
        self.overlap = overlap
        self.sample = sampling_rate
    
    def segmentation(self):
        # Calculate the number of data points in each window
        window_length = int(self.ws[0] * self.sample)

        # Calculate the number of data points to shift the window by for the given overlap
        shift_length = int(window_length * self.overlap)

        # Initialize an empty list to store the segmented data
        segmented_data = []

        # Iterate over the data using a sliding window
        start_index = 0
        df = pd.DataFrame(data = self.data)
        while start_index + window_length <= len(df):
            end_index = start_index + window_length
            segment = df.iloc[start_index:end_index]
            segmented_data.append(segment)
            start_index += shift_length

        # Concatenate the segmented data into a new DataFrame
        df_seg = pd.concat(segmented_data)

        # Reset the index of the segmented DataFrame
        df_seg.reset_index(drop=True, inplace=True)

        return df_seg







