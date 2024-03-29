import tensorflow as tf 
print(tf.__version__) #2.0.0-rc1
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import time
from tensorflow import keras
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import cohen_kappa_score
print(sklearn.__version__)

### path specification
root_dir = "/home/ec2-user/yanting/AD_Deeplearning_regression"
data_dir = os.path.join(root_dir, "training_data")
data_fn = os.path.join(data_dir, "merged_complete_training_data.csv")
plot_dir = os.path.join(root_dir, "plots")
lr_curves_dir = os.path.join(root_dir, "learning_curves")
model_architecture_dir = os.path.join(root_dir, "model_architecture")

### other global variables
num_traits = 5
num_classes = 10
subset_train_experiment = True # for comparison purpose
outcome_discretization = False
post_outcome_discretization = False
model_type = "regression" # ordinal_clf/regression

def load_data(data_fn, num_traits, colnum_meta=5):
	print("loading data from disk...")
	all_data_df = pd.read_csv(data_fn)
	all_data_df.fillna(0, inplace=True)
	meta_data = all_data_df.iloc[:, :colnum_meta]
	features = all_data_df.iloc[:, colnum_meta: -num_traits]
	targets = all_data_df.iloc[:, -num_traits:]
	return meta_data, features, targets

def discretize_targets(targets_df, levels=10):
	print("discretize targets into %i bins..." % levels)
	dicretized_labels = []
	for target_col in targets_df:
		dicretized_label = pd.qcut(targets[target_col], q=10, labels=range(0, 10)).astype(int)
		dicretized_labels += [dicretized_label]
	dicretized_labels_df = pd.concat(dicretized_labels, axis=1)
	print("correlation between dicretized p-values:")
	print(dicretized_labels_df.corr())
	return dicretized_labels_df

def train_valid_test_split(features, targets, stratified=True):
	"""
	resulting: 72% train, 18% valid, 10% test
	"""
	if stratified:
		print("performing stratified train-test-valid split according to mean DM pvalue levels...")
		num_traits = targets.shape[1]
		mean_targets_levels = pd.qcut(targets.mean(axis=1), q=10, labels=range(1, 11)).astype(int)
		feats_targets_df = pd.concat([features, targets], axis=1)
		features = feats_targets_df.copy()
		targets = mean_targets_levels.copy()
	print("start doing train-test-valid split...")
	# split the whole dataset into train-valid and test
	x_train_all, x_test, y_train_all, y_test = train_test_split(features, targets, random_state=7, train_size=0.9)
	# split train-valid into train and valid 
	x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11, train_size=0.8)

	print("the distribution of targets values in train-valid-test: ")
	print(y_train.value_counts()/y_train.shape[0])
	print(y_valid.value_counts()/y_valid.shape[0])
	print(y_test.value_counts()/y_test.shape[0])

	x_train, y_train = x_train.iloc[:, :-num_traits], x_train.iloc[:, -num_traits:]
	x_valid, y_valid = x_valid.iloc[:, :-num_traits], x_valid.iloc[:, -num_traits:]
	x_test, y_test = x_test.iloc[:, :-num_traits], x_test.iloc[:, -num_traits:]
	print("The shape of train features and targets:")
	print(x_train.shape, y_train.shape)
	print("The shape of validation features and targets:")
	print(x_valid.shape, y_valid.shape)
	print("The shape of test features and targets:")
	print(x_test.shape, y_test.shape)
	print("finished...")
	return (x_train, y_train, x_valid, y_valid, x_test, y_test)

def plot_target_distribution(targets, prefix, plot_dir=plot_dir):
	"""
	targets should be passed as a pandas series
	"""
	out_fn = os.path.join(plot_dir, "%s_target_density.png"%prefix)
	ax = sns.distplot(targets, hist=True, kde=True, 
		color = 'darkblue', 
		hist_kws={'edgecolor':'black'},
		kde_kws={'linewidth': 4})
	ax.figure.savefig(out_fn)
	print("save plot to %s" % out_fn)

def standardization(x_train, x_valid, x_test):
	"""
	"""
	print("start doing standardization...")
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_valid_scaled = scaler.transform(x_valid)
	x_test_scaled = scaler.transform(x_test)
	print("finished...")
	return (x_train_scaled, x_valid_scaled, x_test_scaled)

def ordinal_target_encoder(x, num_classes=10):
	# #class=10
	# encode label 3 as [1 1 1 1 0 0 0 0 0 0]; [1 1 1 0 0 0 0 0 0]
	ordinal_encoded_targets = np.ones(num_classes)
	ordinal_encoded_targets[x+1:] = 0
	return ordinal_encoded_targets

def generate_multiple_outputs_list(target, model_type):
	outputs_list = []
	if (model_type=="regression"):
		regression = 1
	else:
		regression = 0
	for i in range(target.shape[1]):
		if regression:
			outputs_list += [target[:, i]]
		else:
			outputs_list += [np.row_stack([ordinal_target_encoder(x) for x in target[:, i]])]
	return outputs_list

def data_preparation_blocked_nn(scaled_feat_mat, feat_group_start_end_dict):
	"""
	scaled_feat_mat: with all features, which is the input of fully connected NN
	feat_group_start_end_dict: records the start, end of each group of features
	"""
	scaled_atac_mat = scaled_feat_mat[:, feat_group_start_end_dict['atac'][0]:feat_group_start_end_dict['atac'][1]+1]
	scaled_rnaseq_mat = scaled_feat_mat[:, feat_group_start_end_dict['rnaseq'][0]:feat_group_start_end_dict['rnaseq'][1]+1]
	scaled_wgbs_mat = scaled_feat_mat[:, feat_group_start_end_dict['wgbs'][0]:feat_group_start_end_dict['wgbs'][1]+1]
	scaled_divan_mat = scaled_feat_mat[:, feat_group_start_end_dict['wgbs'][1]+1:]
	blocked_data_list = [scaled_atac_mat, scaled_rnaseq_mat, scaled_wgbs_mat, scaled_divan_mat]
	return blocked_data_list
	
def custom_r2_metric(y_true, y_pred):
	eps = 1e-8 #avoid zero division problem
	squared_res = keras.backend.sum(keras.backend.square(y_true-y_pred))
	squared_total = keras.backend.sum(keras.backend.square(y_true-keras.backend.mean(y_true)))
	return(1-squared_res/(squared_total+eps))

def build_all_connected_models(feature_dim, output_scheme="regression", num_classes=num_classes):
	"""
	keras model with multiple outputs
	AlphaDropout: 1. mean and std stay the same 2. normalization property stay the same
	output can be either regression mode or classification mode (output_scheme="regression" or "ordinal_clf")
	If in classification mode, the strategy used is in the following paper:
	Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network approach to ordinal regression." 
	2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). IEEE, 2008.
	"""
	input_layer = keras.layers.Input(shape=[feature_dim])
	hidden1 = keras.layers.Dense(800, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(input_layer)
	#bn1 = keras.layers.BatchNormalization()(hidden1)
	dropout1 = keras.layers.AlphaDropout(rate=0.5)(hidden1)
	#hidden2 = keras.layers.Dense(500, activation='relu', 
	#	kernel_regularizer=keras.regularizers.l2(0.01), 
	#	kernel_initializer="he_uniform")(dropout1)
	#bn2 = keras.layers.BatchNormalization()(hidden2)
	#dropout2 = keras.layers.AlphaDropout(rate=0.5)(hidden2)
	hidden3 = keras.layers.Dense(200, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(dropout1)
	dropout3 = keras.layers.AlphaDropout(rate=0.3)(hidden3)
	# trait-specific network
	hidden4_1 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(dropout3)
	hidden4_2 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(dropout3)
	hidden4_3 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(dropout3)
	hidden4_4 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(dropout3)
	hidden4_5 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(dropout3)
	#bn3 = keras.layers.BatchNormalization()(hidden3)
	if (output_scheme=="regression"):
		#outputs = [keras.layers.Dense(1, hidden3) for i in range(5)] #
		out1 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out1")(hidden4_1)
		out2 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out2")(hidden4_2)
		out3 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out3")(hidden4_3)
		out4 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out4")(hidden4_4)
		out5 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out5")(hidden4_5)
	elif (output_scheme == "ordinal_clf"):
		out1 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out1")(hidden4_1)
		out2 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out2")(hidden4_2)
		out3 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out3")(hidden4_3)
		out4 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out4")(hidden4_4)
		out5 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out5")(hidden4_5)
	model = keras.models.Model(inputs=[input_layer],
							   outputs=[out1, out2, out3, out4, out5])
	if (output_scheme == "regression"):
		model.compile(loss="mse", 
					  optimizer=tf.keras.optimizers.Adam(0.0001),
					  metrics=['mae', custom_r2_metric])
	elif (output_scheme == "ordinal_clf"):
		model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
					  optimizer=tf.keras.optimizers.Adam(0.001))
	print(model.summary())
	return model

def build_multi_blocks_models(feature_dim, feat_group_start_end_dict, output_scheme="regression", num_classes=num_classes):
	"""
	keras model with multiple outputs
	AlphaDropout: 1. mean and std stay the same 2. normalization property stay the same
	output can be either regression mode or classification mode (output_scheme="regression" or "ordinal_clf")
	If in classification mode, the strategy used is in the following paper:
	Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network approach to ordinal regression." 
	2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). IEEE, 2008.
	"""
	divan_feat_dim = feature_dim
	for k in feat_group_start_end_dict:
		divan_feat_dim = divan_feat_dim - feat_group_start_end_dict[k][2]
	input_atac = keras.layers.Input(shape=[feat_group_start_end_dict['atac'][2]], name="atac_input")
	input_rnaseq = keras.layers.Input(shape=[feat_group_start_end_dict['rnaseq'][2]], name="rnaseq_input")
	input_wgbs = keras.layers.Input(shape=[feat_group_start_end_dict['wgbs'][2]], name="wgbs_input")
	input_divan = keras.layers.Input(shape=[divan_feat_dim], name="divan_input")

	hidden_atac = keras.layers.Dense(30, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(input_atac)
	hidden_rnaseq = keras.layers.Dense(100, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(input_rnaseq)
	hidden_wgbs = keras.layers.Dense(60, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(input_wgbs)
	hidden_divan_1 = keras.layers.Dense(800, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(input_divan)
	dropout1 = keras.layers.AlphaDropout(rate=0.5)(hidden_divan_1)
	hidden_divan_2 = keras.layers.Dense(200, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(dropout1)
	concat = keras.layers.concatenate([hidden_atac, hidden_rnaseq, hidden_wgbs, hidden_divan_2])
	hidden3 = keras.layers.Dense(200, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(dropout1)
	# trait-specific network
	hidden_trait_1 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(concat)
	hidden_trait_2 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(concat)
	hidden_trait_3 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(concat)
	hidden_trait_4 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(concat)
	hidden_trait_5 = keras.layers.Dense(50, activation='relu', 
		kernel_regularizer=keras.regularizers.l2(0.01), 
		kernel_initializer="he_uniform")(concat)
	#bn3 = keras.layers.BatchNormalization()(hidden3)
	if (output_scheme=="regression"):
		#outputs = [keras.layers.Dense(1, hidden3) for i in range(5)] #
		out1 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out1")(hidden_trait_1)
		out2 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out2")(hidden_trait_2)
		out3 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out3")(hidden_trait_3)
		out4 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out4")(hidden_trait_4)
		out5 = keras.layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=num_classes-1.0), 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out5")(hidden_trait_5)
	elif (output_scheme == "ordinal_clf"):
		out1 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out1")(hidden_trait_1)
		out2 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out2")(hidden_trait_2)
		out3 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out3")(hidden_trait_3)
		out4 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out4")(hidden_trait_4)
		out5 = keras.layers.Dense(num_classes, activation="sigmoid", 
			kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer="he_uniform", 
			name="out5")(hidden_trait_5)
	model = keras.models.Model(inputs=[input_atac, input_rnaseq, input_wgbs, input_divan],
							   outputs=[out1, out2, out3, out4, out5])
	if (output_scheme == "regression"):
		model.compile(loss="mse", 
					  optimizer=tf.keras.optimizers.Adam(0.0001),
					  metrics=['mae', custom_r2_metric])
	elif (output_scheme == "ordinal_clf"):
		model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
					  optimizer=tf.keras.optimizers.Adam(0.001))
	print(model.summary())
	return model

def generate_loss_weight_criteria(num_traits, num_classes=num_classes, weighted_loss=True, top_only=None):
	"""
	define class weights to put more focus on highly DM locus
	param top_only takes into effect only when weighted_loss=True
	"""
	if (weighted_loss):
		print("use different weight for each class")
		print("put more focus on locus with more significant DM p-values")
		if (top_only):
			print("weighted loss for the top class only...")
			class_weight_each_output_dict = dict([(i, 1) for i in range(0, num_classes-1)])
			class_weight_each_output_dict[num_classes-1] = num_classes
			class_weight_dict = dict([("out%i"%i, class_weight_each_output_dict) for i in range(1, num_traits+1)])
		else:
			class_weight_each_output_dict = dict([(i, i+1) for i in range(0, num_classes)])
			class_weight_dict = dict([("out%i"%i, class_weight_each_output_dict) for i in range(1, num_traits+1)])
	else:
		print("use balanced weight for each class")
		class_weight_each_output_dict = dict([(i, 1) for i in range(0, num_classes)])
		class_weight_dict = dict([("out%i"%i, class_weight_each_output_dict) for i in range(1, num_traits+1)])
	return class_weight_dict

def generate_loss_weight_criteria(num_traits, num_classes=num_classes, weighted_loss=True, top_only=None):
	"""
	define class weights to put more focus on highly DM locus
	This leads to higher penalization on False Negative cases
	param top_only takes into effect only when weighted_loss=True
	"""
	if (weighted_loss):
		print("use different weight for each class")
		print("put more focus on locus with more significant DM p-values")
		if (top_only):
			print("weighted loss for the top class only...")
			class_weight_each_output_dict = dict([(i, 1) for i in range(0, num_classes-1)])
			class_weight_each_output_dict[num_classes-1] = num_classes
			class_weight_dict = dict([("out%i"%i, class_weight_each_output_dict) for i in range(1, num_traits+1)])
		else:
			class_weight_each_output_dict = dict([(i, i+1) for i in range(0, num_classes)])
			class_weight_dict = dict([("out%i"%i, class_weight_each_output_dict) for i in range(1, num_traits+1)])
	else:
		print("use balanced weight for each class")
		class_weight_each_output_dict = dict([(i, 1) for i in range(0, num_classes)])
		class_weight_dict = dict([("out%i"%i, class_weight_each_output_dict) for i in range(1, num_traits+1)])
	return class_weight_dict

def custom_lower_fp_cost_func(y_pred, y_true):
	curr_w = tf.math.floor(y_pred)
	weighted_mse_loss = tf.tensordot(curr_w, tf.math.square(y_pred - y_true))
	return weighted_mse_loss


def plot_learning_curves(history, lr_curves_dir=lr_curves_dir):
	"""
	history: tensorflow.python.keras.callbacks.History
	history.history: python dictionary
	"""
	loss_cols = ['loss', 'val_loss']
	mae_col = [k for k in history.history if "mae" in k]
	loss_dict = dict((custom_col, history.history[custom_col]) for custom_col in loss_cols)
	mae_dict = dict((custom_col, history.history[custom_col]) for custom_col in mae_col)

	fig = plt.figure(figsize=(16, 10))
	ax1 = fig.add_subplot(211)
	pd.DataFrame(loss_dict).plot(ax=ax1)
	ax1.xaxis.set_ticks(range(0, pd.DataFrame(loss_dict).shape[0]))
	ax1.xaxis.set_label_text("epochs")
	ax1.set_title("loss(MSE)")
	plt.grid(True)

	ax2 = fig.add_subplot(212)
	pd.DataFrame(mae_dict).plot(ax=ax2)
	ax2.xaxis.set_ticks(range(0, pd.DataFrame(mae_dict).shape[0]))
	ax2.xaxis.set_label_text("epochs")
	ax2.set_title("MAE")
	plt.grid(True)

	fig.subplots_adjust(hspace=0.5)

	lr_fn = os.path.join(lr_curves_dir, "loss.png")
	plt.savefig(lr_fn)
	print("learning curve saved to %s..." % lr_fn)

def report_mean_mae_across_traits(history, dataset_type):
	"""
	history: tensorflow.python.keras.callbacks.History
	history.history: python dictionary
	No return values in this funciton
	"""
	if dataset_type=="train":
		mae_cols = [key for key in history.history if ("mae" in key) and ("val" not in key)]
	elif dataset_type == "valid":
		mae_cols = [key for key in history.history if ("mae" in key) and ("val" in key)]
	else:
		print("Please specify train/valid for the dataset type")
		return
	mae_df = pd.DataFrame(dict((mae_col, history.history[mae_col]) for mae_col in mae_cols))
	mae_df.loc[:, "mean_mae"] = mae_df.mean(axis=1)
	print("minimum mean MAE for %s occurs in epoch %i: %.3f" % (dataset_type, mae_df["mean_mae"].argmin(), mae_df["mean_mae"].min()))

def generate_discretized_predictions(trained_model, x_train, x_valid, x_test, num_classes=10, model_type="regression", threshold=None):
	"""
	generate 3 numpy arrays:
	discretized training/validation/testing predictions
	"""
	print("generating discretized predictions...")
	if (model_type == "regression"):
		train_preds = np.concatenate(model.predict(x_train), axis=1)
		valid_preds = np.concatenate(model.predict(x_valid), axis=1)
		test_preds = np.concatenate(model.predict(x_test), axis=1)
	elif (model_type == "ordinal_clf"):
		if (threshold is None):
			sys.exit("please specify a threshold for ordinal classification")

		train_preds = np.concatenate([(1-preds[0][:, 0]).reshape(preds[0].shape[0], -1), -np.diff(preds[0], axis=1), preds[0][:, -1].reshape(preds[0].shape[0], -1)], axis=1)
		train_preds = np.column_stack([np.argmax(preds<threshold, axis=1)-1 for preds in model.predict(x_train)])
		valid_preds = np.column_stack([np.argmax(preds<threshold, axis=1)-1 for preds in model.predict(x_valid)])
		test_preds = np.column_stack([np.argmax(preds<threshold, axis=1)-1 for preds in model.predict(x_test)])
		return (train_preds, valid_preds, test_preds)
	discretized_train_preds = []
	discretized_valid_preds = []
	discretized_test_preds = []
	all_traits_bin_levels = []
	for col_ind in range(train_preds.shape[1]):
		train_preds_single_trait, bin_levels = pd.qcut(train_preds[:, col_ind], q=num_classes, labels=range(0, num_classes), retbins=True)	
		print(bin_levels)
		discretized_train_preds += [train_preds_single_trait]
		all_traits_bin_levels += [bin_levels]
		discretized_valid_preds += [np.digitize(valid_preds[:, col_ind], bins=bin_levels[1:], right=True)]
		discretized_test_preds += [np.digitize(test_preds[:, col_ind], bins=bin_levels[1:], right=True)]
	discretized_train_preds = np.column_stack(discretized_train_preds)
	discretized_valid_preds = np.column_stack(discretized_valid_preds)
	discretized_test_preds = np.column_stack(discretized_test_preds)
	print("completed...")
	return (discretized_train_preds, discretized_valid_preds, discretized_test_preds)

def evaluate_mae_by_stratification(trained_model, y_mat, straitification_col, prefix, traits_info_list, x_mat=None, preds=None):
	"""
	Two modes available:
	specify straitification_col="labels" or straitification_col="preds"
	1) evaluate by labels: evaluate MAE when true label is 1, 2, 3, ..., 9
	2) evaluate by predictions: evaluate MAE when predictions are 0-1, 1-2, ..., 9-10

	evaluate by predictions is a preferred way since the goal is to annotate unknown locus across the genome
	we want our predictions be as accurate as possible
	"""
	if ((x_mat is None) and (preds is None)):
		print("provide at design matrix(X) or predictions")
		return

	if (not x_mat is None):
		print("predicting on %s set by %s..." % (prefix, straitification_col))
		preds = np.concatenate(model.predict(x_mat), axis=1)
	else:
		print("using provided predictions for evaluation")
	result_dict = {}
	for i, trait in enumerate(traits_info_list):
		trait_preds_label_df = pd.DataFrame({"preds": preds[:, i], "labels": y_mat[:, i]})
		#random simulation
		#mean_absolute_error(np.random.uniform(0, 10, 33447), trait_preds_label_df['labels'])
		if (straitification_col=="labels"):
			merging_colname = "labels"
			mae_by_label = trait_preds_label_df.groupby("labels").apply(lambda x: mean_absolute_error(x['labels'], x['preds'])).to_frame().reset_index()
			mae_by_label.columns = [merging_colname, "%s_%s_MAE"%(prefix, trait)]
		elif (straitification_col=="preds"):
			merging_colname = "preds_group"
			trait_preds_label_df.loc[:, "preds_group"] = np.floor(trait_preds_label_df['preds'])
			mae_by_label = trait_preds_label_df.groupby("preds_group").apply(lambda x: mean_absolute_error(x['labels'], x['preds'])).to_frame().reset_index()
			mae_by_label.columns = [merging_colname, "%s_%s_MAE"%(prefix, trait)]
		else:
			sys.exit("wrong stratification evaluation criteria specified")
		if (i==0):
			res_mae_df = mae_by_label.copy()
		else:
			res_mae_df = res_mae_df.merge(mae_by_label, on=merging_colname, how="left")
	res_mae_df.sort_values(by=merging_colname, ascending=True, inplace=True)
	res_mae_df = res_mae_df.round(3)
	return res_mae_df

### load data and discretize targets
meta_data, features, targets = load_data(data_fn, num_traits)

### get the start and end indices of each feature group 
atac_col_index = []
rnaseq_col_index = []
wgbs_col_index = []
col_ind_count = 0
for f in features:
	if "atacseq" in f:
		atac_col_index.append(col_ind_count)
	elif "rna" in f:
		rnaseq_col_index.append(col_ind_count)
	elif "WGBS" in f:
		wgbs_col_index.append(col_ind_count)
	col_ind_count += 1
feat_group_start_end_dict = {
	"atac": [min(atac_col_index), max(atac_col_index), max(atac_col_index)-min(atac_col_index)+1],
	"rnaseq": [min(rnaseq_col_index), max(rnaseq_col_index), max(rnaseq_col_index)-min(rnaseq_col_index)+1],
	"wgbs": [min(wgbs_col_index), max(wgbs_col_index), max(wgbs_col_index)-min(wgbs_col_index)+1]
}


# log2 transformation
#features = np.log2(features + 1)
if (outcome_discretization):
	targets = discretize_targets(targets)
traits_info_list = [col.split("_")[0] for col in targets.columns]

(x_train, y_train, x_valid, y_valid, x_test, y_test) = train_valid_test_split(features, targets)
### experiment on whether training size impact the results
if (subset_train_experiment):
	train_size = x_train.shape[0]
	subset_ratio = 1.00
	subset_train_size = int(train_size*subset_ratio)
	print("subset training set size: %i" % subset_train_size)
	shuffled_train_index = np.arange(0, train_size)
	np.random.shuffle(shuffled_train_index)
	x_train = x_train.iloc[shuffled_train_index[:subset_train_size], :]
	y_train = y_train.iloc[shuffled_train_index[:subset_train_size], :]
(x_train_scaled, x_valid_scaled, x_test_scaled) = standardization(x_train, x_valid, x_test)

#print(np.isnan(x_train_scaled).sum())
### process targets in the need of kears model
y_train = y_train.values
y_valid = y_valid.values
y_test = y_test.values

### generate target variables fitted for keras model specification
y_train_multout = generate_multiple_outputs_list(y_train, model_type=model_type)
y_valid_multout = generate_multiple_outputs_list(y_valid, model_type=model_type)
y_test_multout  = generate_multiple_outputs_list(y_test, model_type=model_type)

### build the model
#model = build_all_connected_models(feature_dim=x_train.shape[1], output_scheme=model_type, num_classes=num_classes)
model = build_multi_blocks_models(feature_dim=x_train.shape[1], feat_group_start_end_dict=feat_group_start_end_dict, output_scheme=model_type, num_classes=num_classes)
keras.utils.plot_model(model, os.path.join(model_architecture_dir, "model_with_shape_info.png"), show_shapes=True)

### training
# define callbacks
# TensorBoard, Earlystopping, ModelCheckpoint
callback_logdir = os.path.join(root_dir,"callback_log")
if not os.path.exists(callback_logdir):
	os.mkdir(callback_logdir)
output_model_fn = os.path.join(callback_logdir, "AD_regression_best_model.h5")
callbacks = [
	keras.callbacks.TensorBoard(callback_logdir),
	keras.callbacks.ModelCheckpoint(output_model_fn, save_best_only=True),
	keras.callbacks.EarlyStopping(patience=10, min_delta=1e-3)
]

# decide weight strategy
class_weight_dict = generate_loss_weight_criteria(num_traits=5, num_classes=num_classes, weighted_loss=False, top_only=False)

### for blocked NN
x_train_list = data_preparation_blocked_nn(x_train_scaled, feat_group_start_end_dict)
x_valid_list = data_preparation_blocked_nn(x_valid_scaled, feat_group_start_end_dict)
x_test_list = data_preparation_blocked_nn(x_test_scaled, feat_group_start_end_dict)

"""
history = model.fit(x_train_scaled, y_train_multout,
	validation_data=(x_valid_scaled, y_valid_multout),
	batch_size=1024,
	epochs=1000,
	class_weight=class_weight_dict,
	callbacks=callbacks)
model.evaluate(x_test_scaled, y_test_multout)
"""
history = model.fit(x_train_list, y_train_multout,
	validation_data=(x_valid_list, y_valid_multout),
	batch_size=1024,
	epochs=1000,
	class_weight=class_weight_dict,
	callbacks=callbacks)
model.evaluate(x_test_list, y_test_multout)

print(history.history['loss'])
print(history.history['val_loss'])     
plot_learning_curves(history)

# report mean MAE for train and validation set
report_mean_mae_across_traits(history, dataset_type="train")
report_mean_mae_across_traits(history, dataset_type="valid")

# regression mode:
# generate stratified predictions based on quantiles
# i.e., lower 0-10% percentile with label 0; lower 10-20% percentile with label 1 and etc.
"""
d_train_preds, d_valid_preds, d_test_preds=generate_discretized_predictions(model, 
																			x_train=x_train_scaled,
																			x_valid=x_valid_scaled,
																			x_test=x_test_scaled,
																			num_classes=num_classes,
																			model_type=model_type, 
																			threshold=0.2)
"""
if (post_outcome_discretization):
	d_train_preds, d_valid_preds, d_test_preds=generate_discretized_predictions(model, 
																			x_train=x_train_list,
																			x_valid=x_valid_list,
																			x_test=x_test_list,
																			num_classes=num_classes,
																			model_type=model_type)


### evaluation by labels
#print(evaluate_by_label(model, y_train, straitification_col="labels", prefix="train", traits_info_list=traits_info_list, x_mat=x_train_scaled))
#print(evaluate_by_label(model, y_valid, straitification_col="labels",prefix="valid", traits_info_list=traits_info_list, x_mat=x_valid_scaled))
#print(evaluate_by_label(model, y_test, straitification_col="labels", prefix="test", traits_info_list=traits_info_list, x_mat=x_test_scaled))


### evaluation by preds
"""
print(evaluate_mae_by_stratification(model, y_train, straitification_col="preds", prefix="train", traits_info_list=traits_info_list, x_mat=x_train_scaled))
print(evaluate_mae_by_stratification(model, y_valid, straitification_col="preds",prefix="valid", traits_info_list=traits_info_list, x_mat=x_valid_scaled))
print(evaluate_mae_by_stratification(model, y_test, straitification_col="preds", prefix="test", traits_info_list=traits_info_list, x_mat=x_test_scaled))
"""
print(evaluate_mae_by_stratification(model, y_train, straitification_col="preds", prefix="train", traits_info_list=traits_info_list, x_mat=x_train_list))
print(evaluate_mae_by_stratification(model, y_valid, straitification_col="preds",prefix="valid", traits_info_list=traits_info_list, x_mat=x_valid_list))
print(evaluate_mae_by_stratification(model, y_test, straitification_col="preds", prefix="test", traits_info_list=traits_info_list, x_mat=x_test_list))

### evaluation by quantilized preds
if (post_outcome_discretization):
	print(evaluate_mae_by_stratification(model, y_train, straitification_col="preds", prefix="train", traits_info_list=traits_info_list, preds=d_train_preds))
	print(evaluate_mae_by_stratification(model, y_valid, straitification_col="preds",prefix="valid", traits_info_list=traits_info_list, preds=d_valid_preds))
	print(evaluate_mae_by_stratification(model, y_test, straitification_col="preds", prefix="test", traits_info_list=traits_info_list, preds=d_test_preds))

"""
### Kappa statistics
for i in range(5):
	print(cohen_kappa_score(d_valid_preds[:, i], y_valid[:, i]))
for i in range(5):
	print(cohen_kappa_score(d_test_preds[:, i], y_test[:, i]))
"""

### simulation
"""
sim_preds = np.random.uniform(1, 10, size=(33447, ))
sim_preds[sim_preds>=9] = 9
trait_preds_label_df = pd.DataFrame({"preds": sim_preds, "labels": y_test[:, 0]})
trait_preds_label_df.loc[:, "preds_group"] = np.floor(trait_preds_label_df['preds'])
trait_preds_label_df.groupby("preds_group").apply(lambda x: mean_absolute_error(x['labels'], x['preds'])).to_frame().reset_index()
"""
