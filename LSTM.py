# LSTM implementation import copy, numpy as np np.random.seed(0)
# compute sigmoid nonlinearity
def sigmoid(x):
output = 1/(1+np.exp(-x)) return output
# convert output of sigmoid function to its derivative def sigmoid_output_to_derivative(output):
return output*(1-output)
# hypertangent activation function
def tanh(x):
return np.tanh(x)
# convert output of hypertangent function to its derivative
def dtanh(y):
return 1 - y * y
# create function in which we get binary representation of the
integers up to 2^50
binary_dim = 50
def int2binary(num) :
ans = list()
iter = 0
while iter < binary_dim :
if num%2 == 0 : ans.append(0) if num%2 == 1 : ans.append(1)
num = (num - num%2)/2
iter = iter + 1 ans.reverse()
return np.array(ans)
largest_number = 2**binary_dim
# Initial setup
alpha = 0.1 # learning rate input_dim = 2
hidden_dim = 15
output_dim = 1 # = sum of two inputs
# initialize neural network weights
# np.random.random returns floats in [0,1).
synapse_f = 2*np.random.random((input_dim + hidden_dim,output_dim)) -1
synapse_v = 2*np.random.random((hidden_dim,output_dim)) - 1
 synapse_i = 2*np.random.random((input_dim + -1
synapse_c = 2*np.random.random((input_dim + -1
synapse_o = 2*np.random.random((input_dim + -1
hidden_dim,output_dim)) hidden_dim,output_dim)) hidden_dim,hidden_dim))
synapse_f_update
synapse_v_update
synapse_i_update
synapse_c_update
synapse_o_update
= np.zeros_like(synapse_f) = np.zeros_like(synapse_v) = np.zeros_like(synapse_i) = np.zeros_like(synapse_c) = np.zeros_like(synapse_o)
# training logic
for j in range(10000):
# generate a simple addition problem (a + b = c)
a_int = np.random.randint(largest_number/2) a = int2binary(a_int) # binary encoding
b_int = np.random.randint(largest_number/2) b = int2binary(b_int) # binary encoding
# true answer
c_int = a_int + b_int c = int2binary(c_int)
    # where we'll store our best guess (binary encoded);
hat(c)
d = np.zeros_like(c)
    overallError = 0
layer_2_deltas = list()
layer_h_values = list() layer_h_values.append(np.zeros(hidden_dim)) layer_c_values = list() layer_c_values.append(np.zeros(output_dim)) layer_f_values = list() layer_f_values.append(np.zeros(output_dim)) layer_i_values = list() layer_i_values.append(np.zeros(output_dim)) layer_cbar_values = list() layer_cbar_values.append(np.zeros(output_dim)) layer_o_values = list() layer_o_values.append(np.zeros(hidden_dim))
    # moving along the positions in the binary encoding
    # forward
# int version
# int version
d =

for position in range(binary_dim):
         # generate input and output
X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
y = np.array([[c[binary_dim - position - 1]]]).T
# concatenation of prev_hidden and input
Z = np.concatenate((np.atleast_2d(layer_h_values[-1]), X),axis =1)
# computation of hidden layer
layer_f = sigmoid(np.dot(Z,synapse_f))
layer_i = sigmoid(np.dot(Z,synapse_i))
layer_cbar = np.tanh(np.dot(Z,synapse_c))
layer_c = layer_f *layer_c_values[-1] + layer_i * layer_cbar layer_o = sigmoid(np.dot(Z,synapse_o))
layer_h = np.tanh(layer_c) * layer_o
# output layer (new binary representation) layer_2 = sigmoid(np.dot(layer_h,synapse_v))
# did we miss?... if so, by how much? layer_2_error = layer_2 - y
layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(l ayer_2))
overallError += np.abs(layer_2_error[0])
        # decode estimate so we can print it out
d[binary_dim - position - 1] = np.round(layer_2[0][0]) # np.round_(in_array, decimals = 3) = arrays rounded with 3 effective decimals
# store hidden layer so we can use it in the next timestep layer_c_values.append(copy.deepcopy(layer_c)) layer_h_values.append(copy.deepcopy(layer_h)) layer_f_values.append(copy.deepcopy(layer_f)) layer_i_values.append(copy.deepcopy(layer_i)) layer_cbar_values.append(copy.deepcopy(layer_cbar)) layer_o_values.append(copy.deepcopy(layer_o))
future_layer_h_delta = np.zeros(hidden_dim) future_layer_c_delta = np.zeros(output_dim)
# backward propagation step
for position in range(binary_dim):
# initial setup of current position
X = np.array([[a[position],b[position]]])

layer_h = layer_h_values[-position-1] layer_c = layer_c_values[-position-1] layer_f = layer_f_values[-position-1] layer_i = layer_i_values[-position-1] layer_cbar = layer_cbar_values[-position-1] layer_o = layer_o_values[-position-1]
Z = np.concatenate((np.atleast_2d(layer_h_values[- position-2]), X),axis = 1)
# error at output layer
layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
h_delta = future_layer_h_delta + layer_2_delta.dot(synapse_v.T)
o_delta = h_delta * np.tanh(layer_c)
c_delta = future_layer_c_delta + np.sum(h_delta*layer_o)*(1- np.tanh(layer_c)**2)
cbar_delta = c_delta * layer_i
i_delta = c_delta * layer_cbar
f_delta = c_delta * layer_c_values[-position-2]
        # propagation crossing the activation
f_delta_prime = sigmoid_output_to_derivative(layer_f)*f_delta
i_delta_prime = sigmoid_output_to_derivative(layer_i)*i_delta
cbar_delta_prime = dtanh(layer_cbar)*cbar_delta
o_delta_prime = sigmoid_output_to_derivative(layer_o)*o_delta
z_delta = synapse_f.T*f_delta_prime + synapse_i.T*i_delta_prime +
synapse_c.T*cbar_delta_prime+o_delta_prime.dot(synapse_o.T)
        # let's update all our weights so we can try again
synapse_v_update += np.atleast_2d(layer_h).T.dot(layer_2_delta)
synapse_f_update += np.atleast_2d(Z).T.dot(f_delta_prime) synapse_i_update += np.atleast_2d(Z).T.dot(i_delta_prime) synapse_c_update += np.atleast_2d(Z).T.dot(cbar_delta_prime) synapse_o_update += np.atleast_2d(Z).T.dot(o_delta_prime)
future_layer_h_delta = np.array([list(z_delta)[0][:-2]]) future_layer_c_delta = layer_f*c_delta
# synapse updating
synapse_f -= synapse_f_update * alpha synapse_v -= synapse_v_update * alpha synapse_i -= synapse_i_update * alpha synapse_c -= synapse_c_update * alpha synapse_o -= synapse_o_update * alpha
 
#initialize synapse_f_update *= 0 synapse_v_update *= 0 synapse_i_update *= 0 synapse_c_update *= 0 synapse_o_update *= 0
    # print out progress
    if(j % 1000 == 0):
        print ("Error:" + str(overallError))
        print ("Pred:" + str(d))
        print ("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print (str(a_int) + " + " + str(b_int) + " = " + str(out)) 
        print ("------------")
