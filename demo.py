from numpy import *
# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
	#initialize error at 0
    totalError = 0
	#for every data point
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
	    #compute error as the sum of distances of each data point from the line squared
	    #equation error(m,b) = 1/N Sigma(i:N) (y_i - (m * x_i + b_i))^2
        totalError += (y - (m * x + b)) ** 2
	#get average distance of points from line
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
	#starting points for gradient
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
		#direction with respect to b and m
		#computing partial derivatives of our error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    #update b and m values using partial derivatives 
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	#starting b and m
    b = starting_b
    m = starting_m
	#gradient descent
    for i in range(num_iterations):
		#update b and m with new more accurate b and m by performing this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
	#Step 1 - collect data
    points = genfromtxt("data.csv", delimiter=",")
	#Step 2 - define hyperparameters (i.e. tuning knobs)
	#how fast should the model converge? If learning rate is too large the error function may not decrease
    learning_rate = 0.0001
	# slope formula: y = mx + b
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
	#how much should the model be trained. Our dataset is small
    num_iterations = 1000
	#Step 3 - train model
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

if __name__ == '__main__':
    run()
