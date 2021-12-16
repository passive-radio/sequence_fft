from matplotlib.pyplot import axis
import openpyxl as xl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import numpy as np

def excel_to_plot(wb, base):
    sheets = []
    dfs = []
    for sh in wb.worksheets:
        sim_df = pd.DataFrame()
        
        title = sh.title
        
        sheets.append(sh)
        df = pd.DataFrame(sh.values)
        # df = df.drop(df.columns[[-1]], axis=1)
        header = df.iloc[0]
        
        # df.drop(df.iloc[0], axis=0)
        df = df.drop(index=df.index[[0]])
        df = df.set_axis(header, axis="columns")
        dfs.append(df)
        
        print(df["Vout/Vin_exp"].iloc[0])
        
        plt.clf()
        
        # sns.set(font_scale=0.5)
        # sns.set_style("whitegrid", {'grid.linestyle': '--'})
        sns.set_context("paper", 1.5, {"lines.linewidth": 1.5})
        # sns.set_palette(sns.hls_palette(24))
        
        # sns.hls_palette(24)
        f, ax = plt.subplots(figsize=(12, 8))
        ax.set(xscale="log", yscale="log")
        sns.set_style('ticks')
                
        #sns.set(palette="bright")
        
        plt.xlim(10, 10**6)
        plt.ylim(0.01, 100)
        
        
        s1 = sns.scatterplot(df["freq"],df["Vout/Vin_exp"], ax=ax, s=20)
        gain_true = df["Vout/Vin_th"].iloc[0]
        
        sim_df = asc_to_df("sim/"+ title + ".txt", headers=0, delimineter="[,\t]")
        
        p = sns.scatterplot(sim_df["freq"],sim_df["Vout/Vin"], ax=ax, s=20)
        p.set_xlabel("freq [Hz]", fontsize=20)
        p.set_ylabel("amp ratio", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.minorticks_on()
                
        # mil_y = ticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=999) # 対数目盛り
        # ax.yaxis.set_minor_locator(mil_y)
        # p.yaxis.set_minor_locator(mil_y)
        # s1.yaxis.set_minor_locator(mil_y)
        
        # # ax.tick_params(direction = "in", length = 5, colors = "blue")
        
        # mil_x = ticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=999) # 対数目盛り
        # ax.xaxis.set_minor_locator(mil_x)
        # p.xaxis.set_minor_locator(mil_x)
        # s1.xaxis.set_minor_locator(mil_x)
        
        plt.axhline(y=gain_true)
        plt.legend(labels=["Vout/Vin_exp","Vout/Vin_ltspice","1+Rf/Rs"],bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize=18)
        plt.savefig(base + title + ".png")


def pd_arctan(value):
    return np.arctan(value)*180/np.pi

def asc_to_df(file, headers, delimineter):
    data = []
    df = pd.read_csv(file, sep = "[  ,	, , ]", engine='python', header=None, skiprows=1, encoding="shift_jis", error_bad_lines=False)
    
    header = ["freq", "Vout_re", "Vout_im", "Vin_re", "Vin_im"]
    df = df.set_axis(header, axis="columns")
    
    
    df["Vout"] = (df["Vout_re"]**2 + df["Vout_im"]**2)**0.5
    df["Vin"] = (df["Vin_re"]**2 + df["Vout_im"]**2)**0.5
    
    df["Vout/Vin"] = df["Vout"]/df["Vin"]
    
    df["Shift"] = df["Vout_im"]/df["Vout_re"]
    df["Shift"] = df["Shift"].apply(pd_arctan)
    
    df = df.drop(columns=["Vout_re", "Vout_im", "Vin_re", "Vin_im"])
    
    # for column_name, item in df.iteritems():
    #     print(column_name)
    #     if type(df[column_name].iloc[0]) != np.float64:
    #         df[column_name] = df[column_name].str.replace("(", "").str.replace(")","").str.replace("ｰ","")
    return df

    df = pd.read_csv('multi_delim.csv', sep='Delim_first|Delim_second|[|]', 
                engine='python', header=None)
    
    with open(file, "r", encoding="utf-8", errors="ignore", newline='') as f:
        reader = csv.reader(f, delimiter=delimineter)
        for i, row in enumerate(reader):
            if headers < i:
                data.append([float(row[0].replace('\n', '').replace(' ','')), float(row[1].replace('\n', '').replace(' ',''))])
                
    # data = pd.DataFrame(data, columns=['x', 'y'])
    return data

if __name__ == "__main__":

    # sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
    
    wb = xl.load_workbook(r'new_amplifier.xlsx', data_only=True) #data_only: convert math type of thing to only its value
    
    base = "plot_fig/"
    
    excel_to_plot(wb, base)
    
    # base = "sim/"
    # title = wb.worksheets[0].title
    
    # print(title)
    # data = asc_to_df(base+title+".txt", headers=0, delimineter="[,\t]")
    
    # print(data.columns)
    
    # print(data.head(3))
    # print(data.tail(3))
