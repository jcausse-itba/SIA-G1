import contextlib
import io
import unittest

import numpy as np

from tp_3.engine.activation_functions.identity import Identity
from tp_3.engine.activation_functions.relu import ReLU
from tp_3.engine.activation_functions.step import Step
from tp_3.engine.activation_functions.tanh import Tanh
from tp_3.engine.layer import Layer
from tp_3.engine.loss_functions.mse import MeanSquaredError
from tp_3.engine.models.mlp import MLP
from tp_3.engine.optimizers.sgd import SGD


class PerceptronScenarioTests(unittest.TestCase):
    def setUp(self):
        self.xor_inputs = np.array(
            [[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]
        )
        self.xor_targets = np.array([[1.0], [1.0], [-1.0], [-1.0]])

    def fit_quietly(self, model, inputs, targets, epochs, learning_rate):
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(
                inputs,
                targets,
                epochs=epochs,
                lr=learning_rate,
                print_every=epochs + 1,
            )

    def test_step_perceptron_classifies_and(self):
        inputs = np.array(
            [[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]
        )
        expected = np.array([-1.0, -1.0, -1.0, 1.0])

        perceptron = Layer(2, 1, Step())
        perceptron.W = np.array([[1.0], [1.0]])
        perceptron.b = np.array([-1.0])

        actual = perceptron.feed_forward(inputs).ravel()

        np.testing.assert_array_equal(actual, expected)

    def test_simple_linear_perceptron_fits_linear_samples(self):
        inputs = np.linspace(-1.0, 1.0, 50).reshape(-1, 1)
        expected = inputs.copy()
        np.random.seed(7)
        model = MLP([1, 1], Identity(), Identity(), MeanSquaredError(), SGD)

        self.fit_quietly(model, inputs, expected, epochs=1000, learning_rate=0.05)

        np.testing.assert_allclose(model.forward(inputs), expected, atol=0.02)

    def test_simple_nonlinear_perceptron_fits_tanh_samples(self):
        inputs = np.linspace(-2.0, 2.0, 50).reshape(-1, 1)
        expected = np.tanh(inputs)
        np.random.seed(11)
        model = MLP([1, 1], Tanh(), Tanh(), MeanSquaredError(), SGD)

        self.fit_quietly(model, inputs, expected, epochs=2000, learning_rate=0.05)

        np.testing.assert_allclose(model.forward(inputs), expected, atol=0.05)

    def test_mlp_fit_uses_minibatches_and_records_learning_curve(self):
        class CountingSGD(SGD):
            update_count = 0

            def update(self, params, gradients):
                type(self).update_count += 1
                return super().update(params, gradients)

        inputs = np.arange(5.0).reshape(-1, 1)
        targets = inputs.copy()
        CountingSGD.update_count = 0
        model = MLP([1, 1], Identity(), Identity(), MeanSquaredError(), CountingSGD)
        initial_loss = model.loss_fn.compute(model.forward(inputs), targets)

        with contextlib.redirect_stdout(io.StringIO()):
            history = model.fit(
                inputs,
                targets,
                epochs=1,
                lr=0.01,
                print_every=2,
                batch_size=2,
            )

        self.assertEqual(CountingSGD.update_count, 6)
        self.assertEqual(history[0], initial_loss)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1], model.loss_fn.compute(model.forward(inputs), targets))

    def test_mlp_xor_for_requested_architectures(self):
        architectures = ([2, 2, 1], [2, 3, 2, 1])
        for architecture in architectures:
            with self.subTest(architecture=architecture):
                model = MLP(
                    list(architecture),
                    ReLU(),
                    Identity(),
                    MeanSquaredError(),
                    SGD,
                )
                model.layers[0].W = np.array(
                    [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]
                )[:, :architecture[1]]
                model.layers[0].b = np.zeros(architecture[1])

                if architecture == [2, 2, 1]:
                    model.layers[1].W = np.ones((2, 1))
                    model.layers[1].b = np.array([-1.0])
                else:
                    model.layers[1].W = np.array(
                        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
                    )
                    model.layers[1].b = np.zeros(2)
                    model.layers[2].W = np.ones((2, 1))
                    model.layers[2].b = np.array([-1.0])

                actual = model.forward(self.xor_inputs)

                np.testing.assert_allclose(actual, self.xor_targets, atol=1e-12)

    def test_single_step_perceptron_cannot_separate_xor(self):
        positive_inputs = self.xor_inputs[self.xor_targets.ravel() == 1.0]
        negative_inputs = self.xor_inputs[self.xor_targets.ravel() == -1.0]
        positive_affine_mean = np.column_stack(
            [positive_inputs, np.ones(len(positive_inputs))]
        ).mean(axis=0)
        negative_affine_mean = np.column_stack(
            [negative_inputs, np.ones(len(negative_inputs))]
        ).mean(axis=0)

        np.testing.assert_array_equal(positive_affine_mean, negative_affine_mean)


if __name__ == "__main__":
    unittest.main()