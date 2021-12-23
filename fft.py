from logging import log
import os
import re
import csv
from pprint import pprint

import pandas as pd
import numpy as np
from pandas.core import base
import matplotlib.pyplot as plt


def get_and_split(filepath, header):
    df = pd.read_csv(filepath, sep=",", header=0)
    
    return df[0:header-1], df[header:-1]

def index_to_frame(value, interval):
    return float(value)*interval

def str_to_float(value):
    return float(value)

def parse_sec_unit(text):
    m = re.match(r'([0-9]+\.[0-9]+)(\D+)', text)
    return float(str(m.group(1))), m.group(2)


# data_df.plot(x="frame",y="CH1", color="r", label="CH1", ax=ax)
# plt.show()

# CH1 の周波数をfftによる最大強度周波数とする
# スペクトルデータに1周期分のデータがないと不正確
def primary_freq(df, interval, interval_unit, data_type="raw"):
    if data_type=="raw":
        if interval_unit == "μs":
            interval_unit = float(10**-6)
        elif interval_unit == "ms":
            interval_unit = float(10**-3)
        elif interval_unit == "uS":
            interval_unit = float(10**-6)
        F = np.fft.fft(df["CH1"]) # 変換結果
        N = len(df["CH1"])
        dt = interval*interval_unit
        freq = np.fft.fftfreq(N, d=dt) # 周波数
        Amp = np.abs(F/(N/2)) # 振幅
        
        fft_list = [[freq[i], Amp[i]] for i in range(len(freq))]
        df_fft = pd.DataFrame(fft_list, columns=["freq", "amp"])
        df_fft = df_fft[df_fft['freq'] > 0.0]
        freq = df_fft.at[df_fft['amp'].idxmax(), "freq"]
        return f"{freq:.3E}"
    
    elif data_type == "fft":
        df= df[df['freq'] > 0.0]
        freq = df.at[df['amp'].idxmax(), "freq"]
        return f"{freq:.3E}"
    
def fft(filepath, scale="log"):
    if scale=="log":
        setup_df, data_df = get_and_split(filepath, 7)

        data_df = data_df.set_axis(["frame", "CH1", "CH2"],axis="columns").reset_index().reset_index().drop(columns=["index", "frame"]).set_axis(["index", "CH1", "CH2"],axis="columns").copy()
        interval, unit_of_interval = parse_sec_unit(setup_df.loc[5,"CH1"])
        data_df["frame"] = data_df["index"].apply(index_to_frame, interval=interval)
        data_df = data_df.drop(columns="index")
        data_df["CH1"] = data_df["CH1"].apply(str_to_float)
        data_df["CH2"] = data_df["CH2"].apply(str_to_float)
        print("-"*50)
        pprint(setup_df)
        
        
        freq_ch1 = str(setup_df.loc[0, "CH1"])
        
        if "?" not in freq_ch1:
            print(f'freq(CH1): {freq_ch1}')
            primary_freq_ch1 = primary_freq(data_df, interval=interval, interval_unit=unit_of_interval, data_type="raw")
            print(f'freq(CH1) by fft: {primary_freq_ch1}')
        if "?" in freq_ch1:
            print(f'freq(CH1): {freq_ch1}')
            primary_freq_ch1 = primary_freq(data_df, interval=interval, interval_unit=unit_of_interval, data_type="raw")
            print(f'freq(CH1) by fft: {primary_freq_ch1}')
        
        
        
        # ax = data_df.plot(kind="scatter", x="frame",y="CH2", color="b", label="CH2", s=1)
        # data_df.plot.scatter(x="frame", y="CH1", color="g", label="CH1", s=1, ax=ax)
        # plt.xlabel(f"Time ({unit_of_interval})")

        F = np.fft.fft(data_df["CH2"]) # 変換結果
        N = len(data_df["CH2"])
        dt = interval*10**-6
        freq = np.fft.fftfreq(N, d=dt) # 周波数
        # fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,6))
        # ax[0].plot(F.real, label="Real part")
        # ax[0].legend()
        # ax[1].plot(F.imag, label="Imaginary part")
        # ax[1].legend()
        # ax[2].plot(freq, label="Frequency")
        # ax[2].legend()
        # ax[2].set_xlabel("Number of data")

        # plt.show()

        Amp = np.abs(F/(N/2)) # 振幅
        
        fft_list = [[freq[i], Amp[i]] for i in range(len(freq))]
        df_fft = pd.DataFrame(fft_list, columns=["freq", "amp"])
        primary_freq_ch2 = primary_freq(df_fft, interval=interval, interval_unit=unit_of_interval, data_type="fft")
        print(f"Peak freq(CH2): {primary_freq_ch2}")

        fig, ax = plt.subplots()
        # ax.plot(freq[1:int(N/2)], Amp[1:int(N/2)])
        ax.plot(freq[1:int(N/2)], Amp[1:int(N/2)])
        plt.xscale("log")
        ax.set_xlabel("Freqency [Hz]")
        ax.set_ylabel("Amplitude")
        ax.grid()
        plt.show()
        
    elif scale=="linear":
        setup_df, data_df = get_and_split(base_url + endpoint, 7)

        data_df = data_df.set_axis(["frame", "CH1", "CH2"],axis="columns").reset_index().reset_index().drop(columns=["index", "frame"]).set_axis(["index", "CH1", "CH2"],axis="columns").copy()
        interval, unit_of_interval = parse_sec_unit(setup_df.loc[5,"CH1"])
        data_df["frame"] = data_df["index"].apply(index_to_frame, interval=interval)
        data_df = data_df.drop(columns="index")
        data_df["CH1"] = data_df["CH1"].apply(str_to_float)
        data_df["CH2"] = data_df["CH2"].apply(str_to_float)
        print("-"*50)
        pprint(setup_df)
        ax = data_df.plot(kind="scatter", x="frame",y="CH2", color="b", label="CH2", s=1)
        plt.xlabel(f"Time ({unit_of_interval})")

        F = np.fft.fft(data_df["CH2"]) # 変換結果
        N = len(data_df["CH2"])
        dt = interval*10**-6
        freq = np.fft.fftfreq(N, d=dt) # 周波数
        # fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,6))
        # ax[0].plot(F.real, label="Real part")
        # ax[0].legend()
        # ax[1].plot(F.imag, label="Imaginary part")
        # ax[1].legend()
        # ax[2].plot(freq, label="Frequency")
        # ax[2].legend()
        # ax[2].set_xlabel("Number of data")

        plt.show()

        Amp = np.abs(F/(N/2)) # 振幅

        fig, ax = plt.subplots()
        # ax.plot(freq[1:int(N/2)], Amp[1:int(N/2)])
        ax.plot(freq[0:int(N/2)], Amp[0:int(N/2)])
        ax.set_xlabel("Freqency [Hz]")
        ax.set_ylabel("Amplitude")
        ax.grid()
        plt.show()
        
def sequense_fft(dir_path, scale="log"):
    
    import glob, os
    csv_path_list=glob.glob("spectrum/*.csv")
        
    if scale=="log":
        for i in range(len(csv_path_list)):
            print(csv_path_list[i])
            fft(csv_path_list[i], scale=scale)

if __name__ == "__main__":
    base_url = "spectrum/"
    endpoint = "data_42_005.csv"
    output_url = ""
    sequense_fft(base_url)
    
    