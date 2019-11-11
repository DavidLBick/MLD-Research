from dataloader import * 
from sklearn.naive_bayes import GaussianNB
import numpy as np 


# print('DATA SAMPLES:')
# print(data[0, 0, 0, 0])
# print(data[0, 0, 0, 1])
# print(data[0, 0, 0, 2])
# train_data = data[:, :16, :, :500]
# X = train_data.reshape(60*16, 500*306)
# print('X VALS:')
# print(X[0, 0])
# print(X[0, 1])
# print(X[0, 2])

# a = []
# for i in range(60):
# 	for j in range(16):
# 		a.append(i)
# Y = np.array(a)

# gnb = GaussianNB()
# gnb.fit(X, Y)

# test_data = data[:, 16:, :, :500]
# test_X = train_data.reshape(60*4, 500*306)
# test_predictions = gnb.predict(test_X)

# b = []
# for i in range(60):
# 	for j in range(4):
# 		b.append(i)
# test_Y = np.array(b)

end_vals = []
for channel in range(306):
	for ms in range(500):
		train_data = data[:, :16, channel, ms]
		X = train_data.reshape(60*16, 1)

		a = []
		for i in range(60):
			for j in range(16):
				a.append(i)
		Y = np.array(a)

		gnb = GaussianNB()
		gnb.fit(X, Y)

		test_data = data[:, 16:, channel, ms]
		test_X = test_data.reshape(60*4, 1)
		test_predictions = gnb.predict(test_X)

		b = []
		for i in range(60):
			for j in range(4):
				b.append(i)
		test_labels = np.array(b)

		num_correct = np.sum((test_predictions == test_labels))
		end_vals.append((channel, ms, num_correct))

end_vals.sort(key=lambda x: x[2], reverse=True)
print(end_vals)
