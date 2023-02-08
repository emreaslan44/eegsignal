import pandas as pd 
".csv file to .json format saved in subfolder"
df= pd.read_csv(r"C:\Users\yasar\Masaüstü\SDP_Final_Codes\Common_Codes\adgEyeDeneme4.csv")
data = df.to_json(r"C:\Users\yasar\Masaüstü\SDP_Final_Codes\Common_Codes\testdata.json")