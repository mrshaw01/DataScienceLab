import numpy as np


def readfile(filename):
    listdata = []
    myfile = open(filename, "r", encoding= "utf-8")
    for line in myfile:
        data = line.split()
        if data == []:
            break
        listdata.append(data)
    myfile.close()
    return listdata


def normalize_and_add_ones(X):
	X_max = np.array([[np.amax(X[:, column_id]) for column_id in range(X.shape[1])] for _ in range(X.shape[0])])
	X_min = np.array([[np.amin(X[:, column_id]) for column_id in range(X.shape[1])] for _ in range(X.shape[0])])
	X_normalized = (X-X_min)/(X_max-X_min)
	ones = np.array([[1] for _ in range(X_normalized.shape[0])])
	return np.column_stack((ones, X_normalized))


class RidgeRegression:
	def __init__(self):
		return

	def fit(self, X_train, Y_train, LAMBDA):
		assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
		W = np.linalg.inv(
			X_train.transpose().dot(X_train) + LAMBDA * np.identity(X_train.shape[1])
		).dot(X_train.transpose()).dot(Y_train)
		return W

	def fit_gradident_descent(self, X_train, Y_train, LAMBDA, learning_rate, max_num_epoch = 100, batch_size = 20):
		W = np.random.randn(X_train.shape[1])
		W = np.expand_dims(W, axis = 1)
		last_loss = 1e9
		for ep in range(max_num_epoch):
			arr = np.array(range(X_train.shape[0]))
			np.random.shuffle(arr)
			X_train = X_train[arr]
			Y_train = Y_train[arr]
			total_minibatch = int(np.ceil(X_train.shape[0]/ batch_size))
			for i in range(total_minibatch):
				index = i*batch_size
				X_train_sub = X_train[index: min(index+batch_size, X_train.shape[0])]
				Y_train_sub = Y_train[index: min(index+batch_size, X_train.shape[0])]
				grad = X_train_sub.T.dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA*W
				W = W - learning_rate*grad
			new_loss = self.compute_RSS(self.predict(W, X_train), Y_train)
			if np.abs(new_loss - last_loss) < 1e-2:
				break
			last_loss = new_loss
		return W

	def predict(self, W, X_new):
		X_new = np.array(X_new)
		Y_new = X_new.dot(W)
		return Y_new

	def compute_RSS(self, Y_new, Y_predicted):
		loss = 1. / Y_new.shape[0] * np.sum((Y_new - Y_predicted)**2)
		return loss

	def get_the_best_LAMBDA(self, X_train, Y_train):
		def cross_validation(num_folds, LAMBDA):
			row_ids = np.array(range(X_train.shape[0]))
			# Redundant
			ending_ids = len(row_ids)-len(row_ids)%num_folds
			# Standard
			valid_ids = np.split(row_ids[ :ending_ids], num_folds)
			# Add redundant to last part
			valid_ids[-1] = np.append(valid_ids[-1], row_ids[ending_ids: ])
			# Create trainning parts
			train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
			total_RSS = 0
			for i in range(num_folds):
				valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
				train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
				W = self.fit(X_train=train_part['X'], Y_train=train_part['Y'], LAMBDA=LAMBDA)
				# W = self.fit_gradident_descent(X_train=train_part['X'], Y_train=train_part['Y'], LAMBDA=LAMBDA, learning_rate=1e-3)
				Y_predicted = self.predict(W, valid_part['X'])
				total_RSS += self.compute_RSS(valid_part['Y'], Y_predicted)
			return total_RSS/num_folds
			
		def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
			for current_LAMBDA in LAMBDA_values:
				aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
				if aver_RSS < minimum_RSS:
					best_LAMBDA = current_LAMBDA
					minimum_RSS = aver_RSS
				print(f"LAMBDA: {current_LAMBDA}\tRSS: {aver_RSS}\tBest LAMBDA: {best_LAMBDA}\tMinimum RSS: {minimum_RSS}")
			return best_LAMBDA, minimum_RSS
		
		# Initialize
		best_LAMBDA = 0
		minimum_RSS = 1e10

		# Scan with long steps
		LAMBDA_values = range(50)
		best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA, minimum_RSS=minimum_RSS, LAMBDA_values=LAMBDA_values)

		# Scan with short steps
		LAMBDA_values = np.array(range(max(0, (best_LAMBDA-1)*1000), (best_LAMBDA+1)*1000))/1000
		best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA, minimum_RSS=minimum_RSS, LAMBDA_values=LAMBDA_values)

		# Return
		return best_LAMBDA

if __name__ == "__main__":
	# Load & process data
	data = np.array(readfile("x28.txt")).astype('float')
	print(f"Original data shape: {data.shape}")
	X = data[:,1: data.shape[1]-1]
	print(f"Original X shape: {X.shape}")
	Y = data[:,data.shape[1]-1: ]
	print(f"Original Y shape: {Y.shape}")
	X = normalize_and_add_ones(X)
	print(f"Normalized X shape: {X.shape}")

	# 50 data points for trainning, 10 data points for testing
	X_train, Y_train = X[:50], Y[:50]
	X_test, Y_test = X[50:], Y[50:]	

	# Ridge regression
	ridge_regression = RidgeRegression()

	# Get best lambda for ridge
	best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)	
	print('Best LAMBDA:',best_LAMBDA)

	# Learn the weight
	W_learned = ridge_regression.fit(X_train = X_train, Y_train=Y_train,LAMBDA=best_LAMBDA)

	# Testing
	Y_predicted = ridge_regression.predict(W = W_learned, X_new = X_test)

	# RSS computation
	print("RSS:", ridge_regression.compute_RSS(Y_new = Y_test, Y_predicted = Y_predicted))
