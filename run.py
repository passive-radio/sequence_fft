from fft import sequense_fft, fft


if __name__ == "__main__":
    
    base_path = "spectrum/"
    output_path = "fft_results/"
    
    # txt_to_csv("spectrum_txt/", output_path)
    # print( pd.read_csv(base_path+endpoint).columns)
    
    sequense_fft(base_path, output_dir=output_path, header=7)
    