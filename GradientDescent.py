#!/usr/bin/env python
import numpy as np
import checkversion

class GradientDescent:
    def get_mini_batches(X, y, batch_size):
        random_idxs = random.choice(len(y), len(y), replace=False)
        X_shuffled = X[random_idxs,:]
        y_shuffled = y[random_idxs]
        mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                       i in range(0, len(y), batch_size)]
        return mini_batches

    def train_nn_MBGD(nn_structure, X, y, bs=100, iter_num=3000, alpha=0.25, lamb=0.000):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        mini_batches = get_mini_batches(X, y, bs)
        for mb in mini_batches:
            X_mb = mb[0]
            y_mb = mb[1]
            # pdb.set_trace()
            for i in range(len(y_mb)):
                delta = {}
                # perform the feed forward pass and return the stored h and z values, 
                # to be used in the gradient descent step
                h, z = feed_forward(X_mb[i, :], W, b)
                # loop from nl-1 to 1 backpropagating the errors
                for l in range(len(nn_structure), 0, -1):
                    if l == len(nn_structure):
                        delta[l] = calculate_out_layer_delta(y_mb[i,:], h[l], z[l])
                        avg_cost += np.linalg.norm((y_mb[i,:]-h[l]))
                    else:
                        if l > 1:
                            delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                        # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                        tri_W[l] += np.dot(delta[l+1][:,np.newaxis], 
                                          np.transpose(h[l][:,np.newaxis])) 
                        # trib^(l) = trib^(l) + delta^(l+1)
                        tri_b[l] += delta[l+1]
            # perform the gradient descent step for the weights in each layer
            for l in range(len(nn_structure) - 1, 0, -1):
                W[l] += -alpha * (1.0/bs * tri_W[l] + lamb * W[l])
                b[l] += -alpha * (1.0/bs * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func