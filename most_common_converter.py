import numpy as np
import matplotlib.pyplot as plt

a = np.load('gnb_accuracies.npy')


# Y = list(map(lambda x: x[0], a))
# X = list(map(lambda x: x[1], a))
# W = list(map(lambda x: x[2], a))

# for (channel, ms, num_correct) in a:
# 	X.append(channel)
# 	Y.append(ms)
# 	W.append(num_correct)

# heatmap, xedges, yedges = np.histogram2d(X, Y, bins=[500, 306], weights=W)
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# plt.clf()
# plt.imshow(heatmap.T, extent=extent, origin='lower')
# plt.show()



res = [0 for i in range(306)]
for (channel, ms, num_correct) in a:
	res[channel] += num_correct

Y = []
X = []
W = [0 for i in range(500*306)]
for channel in range(306):
	for ms in range(500):
		W[channel*500 + ms] = res[channel]
		Y.append(channel)
		X.append(ms)


heatmap, xedges, yedges = np.histogram2d(X, Y, bins=[500, 306], weights=W)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()


print(sorted(enumerate(res), key=lambda x: x[1], reverse=True))



# res = [(i,0) for i in range(306)]
# for (channel, ms, num_correct) in a:
# 	(idx, count) = res[channel]
# 	res[channel] = (idx, count+num_correct)

# res.sort(key=lambda x: x[1], reverse=True) # A list of 
# print(res)
# res = np.array(res)
# np.save('most_common_channels.npy', res)