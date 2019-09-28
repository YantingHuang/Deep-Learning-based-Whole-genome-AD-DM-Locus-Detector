import pandas as pd 
import sys, os 

data_dir = "/home/ec2-user/yanting/AD_Deeplearning_regression/training_data"
complete_train_data_fns = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir) if "complete" in fn]


def check_cpgs_consistency(complete_train_data_fns):
	'''
	check if all traits have same CpGs
	'''
	first_file = True
	for fn in complete_train_data_fns:
		print(fn)
		if first_file:
			df = pd.read_csv(fn, usecols=[0, 1, 2, 3, 4, 5])
			standard_cpgs = df["CPG.Labels"]
			print(df.shape)
			first_file = False
		else:
			df_curr = pd.read_csv(fn, usecols=[0, 1, 2, 3, 4, 5])
			consistent_cpgs = df["CPG.Labels"]
			print((consistent_cpgs==standard_cpgs).sum())

def combine_data(complete_train_data_fns):
	'''
	combine data from multiple traits into one
	'''
	first_file = True
	for fn in complete_train_data_fns:
		print(fn)
		trait = fn.split('/')[-1].split('_')[0]
		if first_file:
			df = pd.read_csv(fn)
			df.rename({"P.value": "%s_pval"%trait}, axis="columns", inplace=True)
			df = pd.concat([df.iloc[:, :3], df.iloc[:, 4:], df.iloc[:, 3].to_frame()], axis=1)
			first_file = False

		else:
			df_curr = pd.read_csv(fn, usecols=[3])
			df_curr.rename({"P.value": "%s_pval"%trait}, axis="columns", inplace=True)
			df = pd.concat([df, df_curr], axis=1, ignore_index=False)
	return df

merged_df = combine_data(complete_train_data_fns)
merged_df.to_csv(os.path.join(data_dir, "merged_complete_training_data.csv"), index=False)

