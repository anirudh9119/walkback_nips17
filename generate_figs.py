#!/usr/bin/env python
import argparse
from collections import OrderedDict

import numpy
import theano
from blocks.serialization import load
from matplotlib import pyplot
from theano import tensor


def plot(main_loop):
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    model_brick, = main_loop.model.top_bricks
    training_examples = numpy.vstack(
        [batch for (batch,) in main_loop.data_stream.get_epoch_iterator()])

    figure, axes = pyplot.subplots(nrows=2, ncols=2)

    # Plot energy curves
    axis = axes[0, 0]
    X1_coord, X2_coord = numpy.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    x1_value, x2_value = X1_coord.flatten(), X2_coord.flatten()

    v = tensor.matrix('v')
    v_data = tensor.zeros_like(v)
    v_0 = tensor.zeros_like(v)

    F = model_brick.total_energy(v, v_data, v_0, 0, 0, 1)
    energy_function = theano.function([v], F)
    F_value = energy_function(numpy.vstack([x1_value, x2_value]).T)
    C = F_value.reshape((100, 100))

    curves = axis.contour(X1_coord, X2_coord, C,
                          levels=numpy.linspace(C.min(), C.max(), 40))
    figure.colorbar(curves, ax=axis)
    axis.scatter(training_examples[:, 0], training_examples[:, 1])
    axis.set_xlim([xmin, xmax])
    axis.set_ylim([ymin, ymax])
    axis.set_title('Energy curves')
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')
    axis.set_aspect('equal')

    # Plot trajectories
    axis = axes[0, 1]
    v_data = tensor.matrix('v_data')
    v_data_value = training_examples[:100]

    v_0, v_0_updates = model_brick.langevin_trajectory(
        v_data, v_data * tensor.ones_like(v_data), 0, 0, 0.0001,
        model_brick.num_steps)
    pos_v, pos_updates = model_brick.langevin_trajectory(
        v_data, v_0, model_brick.beta, model_brick.gamma, 1,
        model_brick.num_steps)
    neg_v, neg_updates = model_brick.langevin_trajectory(
        v_data, v_0, 0, model_brick.gamma, 1, model_brick.num_steps)

    updates = OrderedDict()
    updates.update(v_0_updates)
    updates.update(pos_updates)
    updates.update(neg_updates)

    function = theano.function(
        [v_data], [v_0, pos_v, neg_v], updates=updates)
    v_0_value, pos_v_value, neg_v_value = function(v_data_value)

    axis.scatter(v_data_value[:, 0], v_data_value[:, 1], c='b')
    axis.scatter(v_0_value[:, 0], v_0_value[:, 1], c='k')
    axis.scatter(pos_v_value[:, 0], v_0_value[:, 1], c='g')
    axis.scatter(neg_v_value[:, 0], v_0_value[:, 1], c='r')
    axis.set_title('Trajectories')
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')
    axis.set_aspect('equal')

    # Plot samples
    axis = axes[1, 0]
    v_data_value = training_examples[:1000]

    v_0 = model_brick.theano_rng.normal(size=(1000, 2))
    v, updates = model_brick.langevin_trajectory(
        v_0, v_0 * tensor.ones_like(v_0), 0, 0, 1.0, 1000)

    function = theano.function([], v, updates=updates)
    v_value = function()

    axis.scatter(v_data_value[:, 0], v_data_value[:, 1], c='b')
    axis.scatter(v_value[:, 0], v_value[:, 1], c='r')
    axis.set_title('Samples')
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')
    axis.set_aspect('equal')

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot energy curves, trajectories and samples.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    args = parser.parse_args()
    with open(args.main_loop_path, 'rb') as src:
        main_loop = load(src)
    plot(main_loop)
