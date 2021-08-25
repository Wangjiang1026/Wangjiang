import torch
import numpy as np
from rnn.models import network
import matplotlib.pyplot as plt
from rnn.data import dataset, data_prepare
from rnn.models.transform import Normalization
from config import *

SHOW_DETAIL = False

company_data = data_prepare.read_data(symbol)
obj_data = company_data[columns].to_numpy()

train_data = obj_data[:-test_days]
scaler = Normalization()
train_data_norm = scaler.fit_transform(train_data)

obj_data_norm = scaler.transform(obj_data)
test_loader = dataset.build_loader(obj_data_norm, batch_size=1, n_steps=n_steps, shuffle=False)

model = network.LSTMPriceModel(obj_data.shape[1], obj_data.shape[1])
print(model)

model_path = model_path + '_lstm'
ckpt_filename = f'{model_path}/model_4950.pt'
state_dict = torch.load(ckpt_filename)
model.load_state_dict(state_dict)
model.eval()
model=model.to(device)


################ test ################
targets = []
preds = []
for input, target in test_loader:
    input = input.to(device)
    with torch.no_grad():
        model.init_hidden(1, device)
        output, _ = model(input)
    preds.append(output.detach().cpu().numpy())
    targets.append(target.detach().cpu().numpy())

preds = scaler.inverse_transform(np.concatenate(preds, axis=0))[:, -1]
targets = scaler.inverse_transform(np.concatenate(targets, axis=0))[:, -1]
index = company_data.index[n_steps:]

if SHOW_DETAIL:
    show_len = test_days * 4
    preds = preds[-show_len:]
    targets = targets[-show_len:]
    index = index[-show_len:]

# true
plt.plot(index, targets, label="true")

# pred
plt.plot(index[:-test_days], preds[:-test_days], color='y', label="train")
plt.plot(index[-test_days:], preds[-test_days:], color='r', label='test')
plt.gcf().autofmt_xdate()
plt.legend()
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Close Price Fitting')
plt.show()

from sklearn.metrics import mean_squared_error

error = mean_squared_error(targets[-test_days:], preds[-test_days:], squared=False)
print(error)