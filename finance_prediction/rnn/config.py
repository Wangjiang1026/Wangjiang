import os
import torch

device = "cuda" if  torch.cuda.is_available() else "cpu"
test_days = 96
batch_size = 128
n_steps = 64

columns = ['Volume', 'Close']
# columns = ['Close']


# symbol = "AMZN"
symbol = "GOOG"


START_DATE = '2012-01-03'
END_DATE = '2019-12-31'


DATASETS_PATH = os.path.join(os.path.dirname(__file__), '../datasets')

# GOOG: 2004-08-19, 2021-08-16
# AMZN: 1997-05-15, 2021-08-16

model_path = os.path.join(os.path.dirname(__file__), 'checkpoints') + "_" + symbol