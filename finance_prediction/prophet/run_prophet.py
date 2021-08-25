import datetime
import math

import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error


class ProphetModel (object):
	def __init__(self, data: pd.DataFrame):
		"""
		初始化模型数据
		:param data: 模型输入数据包含ds和y
		"""
		self.data = data

	def predict(self, pred_periods=1):
		model = Prophet ()
		model.fit (self.data)
		future = model.make_future_dataframe (periods=pred_periods)
		forecast = model.predict (future)
		pred_res = [round (x, 2) for x in forecast['yhat'].to_list ()]
		train_res = pred_res[0:self.data.shape[0]]
		test_res = pred_res[self.data.shape[0]:]
		return train_res, test_res


def cal_rmse(y, y_hat):
	"""
	计算RMSE
	:param y: 真实值
	:param y_hat: 预测值
	:return:
	"""
	rmse = round (math.sqrt (mean_squared_error (y, y_hat)), 6)
	return rmse



def prophet_process(stock_file: str):
	raw_data = pd.read_csv (stock_file)
	raw_data['Date'] = raw_data['Date'].apply (lambda x: datetime.datetime.strptime (x, '%d-%m-%Y'))
	# raw_data['Date'] = pd.to_datetime (raw_data['Date'])
	raw_data.rename (columns={'Date': 'ds', 'Adjusted Close': 'y'}, inplace=True)
	raw_data = raw_data[['ds', 'y']]
	raw_data = raw_data[(raw_data['ds'] >= datetime.datetime (2012, 1, 3))
	                      & (raw_data['ds'] <= datetime.datetime (2019, 12, 31))].copy ()
	raw_data.sort_values (by='ds', inplace=True)
	raw_data.reset_index (drop=True, inplace=True)

	# # 选择训练数据和测试数据。方式1： 直接按照选择的日期选择数据
	# train_data = raw_data[(raw_data['ds'] >= datetime.datetime (2017, 1, 1))
	#                       & (raw_data['ds'] <= datetime.datetime (2021, 3, 15))].copy ()
	# train_data.sort_values (by='ds', inplace=True)
	# train_data.reset_index (drop=True, inplace=True)
	# test_data = raw_data[(raw_data['ds'] >= datetime.datetime (2021, 3, 15))
	#                      & (raw_data['ds'] <= datetime.datetime (2021, 6, 15))].copy ()
	# test_data.sort_values (by='ds', inplace=True)
	# test_data.reset_index (drop=True, inplace=True)

	# # 选择训练数据和测试数据.方式2：设置测试数据比例
	# test_ratio = 0.1
	# train_ratio = 1 - test_ratio
	# train_size = int (raw_data.shape[0] * train_ratio)
	# train_data = raw_data.loc[0: train_size].copy ()
	# test_data = raw_data.loc[train_size:].copy ()

	# 选择训练数据和测试数据.方式3：设置测试数据的周期
	test_size = 96
	train_size = raw_data.shape[0] - test_size
	train_data = raw_data.loc[0: train_size].copy ()
	test_data = raw_data.loc[train_size:].copy ()


	# 生成用于拟合的符合prophet入参要求的数据
	data = pd.DataFrame ()
	data['ds'] = pd.date_range (start=train_data['ds'].min (), periods=train_data.shape[0])
	data['y'] = train_data['y']

	# 预测
	pm = ProphetModel (data)
	train_res, test_res = pm.predict (pred_periods=test_data.shape[0])
	train_data['y_hat'] = train_res
	test_data['y_hat'] = test_res

	# 模型评估
	rmse = cal_rmse (test_data['y'].tolist (), test_res)
	print ('rmse：{0}'.format (rmse))

	# 最终结果存入文件
	final_res = pd.concat ([train_data, test_data], axis=0)
	final_res['rmse'] = rmse
	final_res.sort_values (by='ds', inplace=True)
	final_res.reset_index (drop=True, inplace=True)
	final_res.to_excel ('result.xlsx')

	# 绘图并存储
	plt.figure (1)
	plt.plot (final_res['ds'], final_res['y'], label='true')
	plt.plot (train_data['ds'], train_data['y_hat'], color='y', label="train")
	plt.plot (test_data['ds'], test_data['y_hat'], color='r', label='test')
	plt.legend ()
	# plt.title ('[Big plot] company:{0}, rmse:{1}'.format(stock_file.split('.')[0], rmse))
	plt.xlabel ('Date')
	plt.ylabel ('Close Price')
	plt.title ('Close Price Fitting')
	plt.gcf ().autofmt_xdate ()
	plt.savefig ('./test1.jpg')
	plt.show()

	# 放大画后面的数据：测试数据占1/4
	plt_test_ratio = 0.25
	true_size = int (test_data.shape[0] / plt_test_ratio)
	train_size = true_size - test_data.shape[0]
	final_res = final_res.loc[(final_res.shape[0] - true_size):]
	train_data = train_data.loc[(train_data.shape[0] - train_size):]
	plt.figure (2)
	plt.plot (final_res['ds'], final_res['y'], label='true')
	plt.plot (train_data['ds'], train_data['y_hat'], color='y', label="train")
	plt.plot (test_data['ds'], test_data['y_hat'], color='r', label='test')
	plt.legend ()
	# plt.title (stock_file.split ('.')[0])
	plt.xlabel ('Date')
	plt.ylabel ('Close Price')
	plt.title ('Close Price Fitting')
	plt.gcf ().autofmt_xdate ()
	plt.savefig ('./test2.jpg')
	plt.show ()


if __name__ == "__main__":
	prophet_process ("AMZN.csv")
