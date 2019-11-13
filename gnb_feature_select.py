from dataloader import * 
from sklearn.naive_bayes import GaussianNB
import numpy as np 


# end_vals = []
# for channel in range(306):
# 	for ms in range(500):
# 		train_data = data[:, :16, channel, ms]
# 		X = train_data.reshape(60*16, 1)

# 		a = []
# 		for i in range(60):
# 			for j in range(16):
# 				a.append(i)
# 		Y = np.array(a)

# 		gnb = GaussianNB()
# 		gnb.fit(X, Y)

# 		test_data = data[:, 16:, channel, ms]
# 		test_X = test_data.reshape(60*4, 1)
# 		test_predictions = gnb.predict(test_X)

# 		b = []
# 		for i in range(60):
# 			for j in range(4):
# 				b.append(i)
# 		test_labels = np.array(b)

# 		num_correct = np.sum((test_predictions == test_labels))
# 		end_vals.append((channel, ms, num_correct))

# end_vals.sort(key=lambda x: x[2], reverse=True)
# print(end_vals)

end_vals = []
for channel in range(306):
	train_data = data[:, :16, channel, :500]
	X = train_data.reshape(60*16, 500)

	a = []
	for i in range(60):
		for j in range(16):
			a.append(i)
	Y = np.array(a)

	gnb = GaussianNB()
	gnb.fit(X, Y)

	test_data = data[:, 16:, channel, :500]
	test_X = test_data.reshape(60*4, 500)
	test_predictions = gnb.predict(test_X)

	b = []
	for i in range(60):
		for j in range(4):
			b.append(i)
	test_labels = np.array(b)

	num_correct = np.sum((test_predictions == test_labels))
	end_vals.append((channel, num_correct))

end_vals.sort(key=lambda x: x[1], reverse=True)
print(end_vals)