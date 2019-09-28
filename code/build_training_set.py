import sys, os
import pandas as pd 
import numpy as np


def get_target_vars(meta_analysis_dir, trait, target_col="P.value"):
	'''
	This function is intended to extract the labels for training
	3 possible labels to use: "P.value", "gc.p.value", "T.statistic"
	default option is: "P.value"
	'''
	print("now processing trait: %s" % trait)
	meta_res_fn = os.path.join(meta_analysis_dir, "ROSMAP_%s_ewas_wcelltype_educ.csv"%trait)
	meta_res_df = pd.read_csv(meta_res_fn)
	target_df = meta_res_df[["CPG.Labels", target_col]]
	if (target_col=="P.value" or target_col=="gc.p.value"):
		target_df[target_col] = -np.log10(target_df[target_col])
	return target_df

def mark_genomic_coord_winid(whole_genome_wins_fn, position_meta_data_fn, target_df):
	position_df = pd.read_csv(position_meta_data_fn, sep='\t')
	wins_df = pd.read_csv(whole_genome_wins_fn, sep=" ")
	wins_df['winid'] = wins_df.index + 1
	target_wcoor_df = pd.merge(target_df, position_df[['TargetID', 'CHR', 'MAPINFO']], left_on='CPG.Labels', right_on='TargetID', how='left')
	target_wcoor_df.dropna(axis=0, subset=["TargetID"], inplace=True)
	target_wcoor_df['win_start'] = (target_wcoor_df['MAPINFO']/200).astype(int)*200+1
	target_wcoor_df['CHR'] = 'chr' + target_wcoor_df['CHR'].astype(int).astype(str)
	target_wcoor_df = pd.merge(target_wcoor_df, wins_df[["winid", "seqnames", "start"]], left_on=['CHR', 'win_start'], right_on=["seqnames", "start"])
	target_wcoor_df.drop(labels=['TargetID', "seqnames", "start"], axis=1, inplace=True)
	target_wcoor_df.sort_values("winid", ascending=True, inplace=True)
	return target_wcoor_df

def match_features(data_common_dir, target_wcoor_df):
	'''
	match features from hdf5 files
	feature types include: ATAC-seq(66), RNA-seq(243), WGBS(127)
	'''
	atac_seq_fn = os.path.join(data_common_dir, "atac_seq.h5")
	rna_seq_fn = os.path.join(data_common_dir, "RNA_seq.h5")
	wgbs_fn = os.path.join(data_common_dir, "WGBS_single_H5S")

	target_winids = (target_wcoor_df['winid']-1).tolist()
	target_wcoor_df.rename({"CHR": "chr", "MAPINFO": "coordinate"}, axis="columns", inplace=True)
	### atac-seq part
	print("###start maching atac-seq features###")
	with pd.HDFStore(atac_seq_fn, 'r') as h5_atac:
		for k in h5_atac.keys():
			print(k)
			sys.stdout.flush()
			feat_name = k.strip('/') + "_atacseq_counts"
			target_wcoor_df[feat_name] = h5_atac[k].iloc[target_winids, 1].values
	print(target_wcoor_df.shape)
	### rna-seq part
	print("###start maching rna-seq features###")
	with pd.HDFStore(rna_seq_fn, 'r') as h5_rna:
		for k in h5_rna.keys():
			print(k)
			sys.stdout.flush()
			feat_name = k.strip('/') + "_rnaseq_counts"
			target_wcoor_df[feat_name] = h5_rna[k].iloc[target_winids, 1].values
	print(target_wcoor_df.shape)
	### WGBS part
	print("###start maching WGBS features###")
	target_wcoor_df.set_index(['chr', 'coordinate'], inplace=True)
	with pd.HDFStore(wgbs_fn, 'r') as h5_wgbs:
		for k in h5_wgbs.keys():
			print(k)
			sys.stdout.flush()
			tmp_single_exp_df = h5_wgbs[k]
			tmp_single_exp_df['chr'] = 'chr'+tmp_single_exp_df['chr'].astype(int).astype(str)
			tmp_single_exp_df.set_index(['chr', 'coordinate'], inplace=True)
			target_wcoor_df = target_wcoor_df.join(tmp_single_exp_df, how='left')
			#target_wcoor_df = pd.merge(target_wcoor_df, tmp_single_exp_df, on=['chr', 'coordinate'], how="left")
	print(target_wcoor_df.shape)
	target_wcoor_df.reset_index(inplace=True)
	return target_wcoor_df

if __name__ == '__main__':
	### global variables specification
	trait_names = ["amyloid", "braak", "cerad", "cogdec", "cerad_asfactor", "gpath", "tangles"]
	thomas_meta_analysis_res_dir = "/home/ec2-user/thomasLab_ewas_result_20181221"
	position_meta_data_fn = "/home/ec2-user/volume/git/EnsembleCpG/data/AD_CpG/ROSMAP_arrayMethylation_metaData.tsv"
	data_common_dir = "/home/ec2-user/volume/git/EnsembleCpG/data/commons"
	training_data_dir = "/home/ec2-user/yanting/AD_Deeplearning_regression/training_data"
	whole_genome_wins_fn = os.path.join(data_common_dir, "wins.txt")

	trait = "amyloid"
	trait_names = ["braak", "cerad", "gpath", "tangles"]
	for trait in trait_names:		
		target_df = get_target_vars(thomas_meta_analysis_res_dir, trait)
		target_wcoor_df = mark_genomic_coord_winid(whole_genome_wins_fn, position_meta_data_fn, target_df)
		target_wcoor_df = match_features(data_common_dir, target_wcoor_df)
		print("Saving to disk...")
		target_wcoor_df.to_csv(os.path.join(training_data_dir, "%s_training_data.csv"%trait), index=False)
		print("Completed...")
