package NeuralNets;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;

public class BackProp extends NeuralNet {

	// not part of program, just homework stuff
	// public static void main(String[] args) {
	// double x = 2;
	// System.out.println(-3 * (x * x) + 10 * x >= -2);
	// }

	public double learningRate = .7;

	public BackProp(Random r, int[] layers) {
		super(r, layers);
	}

	public BackProp(File saveLoc) throws FileNotFoundException {
		super(saveLoc);
	}

	public static void main(String[] args) {
		double[][] input = new double[4][];
		double[][] expected = new double[4][];

		input[0] = new double[] { 0, 0 };
		input[1] = new double[] { 1, 0 };
		input[2] = new double[] { 0, 1 };
		input[3] = new double[] { 1, 1 };

		expected[0] = new double[] { 0 };
		expected[1] = new double[] { 1 };
		expected[2] = new double[] { 1 };
		expected[3] = new double[] { 1 };

		int[] layers = new int[] { 2, 3, 3, 1 };
		BackProp n = new BackProp(new Random(12), layers);

		for (int i = 0; i < input.length; i++) {
			System.out.print("input: ");
			printArray(input[i]);
			System.out.println("output: " + n.activate(input[i])[0] + " expected " + expected[i][0]);
		}
		System.out.println("error: " + n.batchTrain(input, expected));
		double error = 0;
		for (int i = 0; i < 200000; i++) {
			error = n.batchTrain(input, expected);
			n.initializeMatrices();
		}
		System.out.println("error: " + error);
		for (int i = 0; i < input.length; i++) {
			System.out.print("input: ");
			printArray(input[i]);
			System.out.println("output: " + n.activate(input[i])[0] + " : " + expected[i][0]);
			System.out.println();
		}
		printArray(n.activate(input[0]));
	}

	public static int batchSize = 16;

	// finds the gradients for a batch of training data, then averages them and
	// applies them
	public double batchTrain(double[][] input, double[][] expected) {

		// the added cost for the whole batch, is averaged and returned later
		double batchCost = 0;

		// fills gradient arrays with 0
		double[][][] weightGradient = new double[weights.length][][];
		double[][] biasGradient = new double[biases.length][];

		// initializing the gradient arrays
		for (int i = 0; i < weights.length; i++) {
			weightGradient[i] = new double[weights[i].length][];
			for (int j = 0; j < layers[i]; j++)
				weightGradient[i][j] = new double[weights[i][j].length];
			biasGradient[i] = new double[biases[i].length];
		}

		// getting gradients for individual inputs
		for (int i = 0; i < input.length; i++) {
			// z is activations without sigmoid
			double[][] z = new double[layers.length - 1][];
			double[][] activations = new double[layers.length - 1][];
			z[0] = getZ(input[i], 0);
			activations[0] = sigmoid(z[0]);
			for (int j = 0; j < z[0].length; j++) {
				z[0][j] = sigmoidPrime(z[0][j]);
			}

			for (int j = 1; j < z.length; j++) {
				z[j] = getZ(activations[j - 1], j);
				activations[j] = sigmoid(z[j]);

				for (int k = 0; k < z[j].length; k++) {
					z[j][k] = sigmoidPrime(z[j][k]);
				}
			}

			// cost is placed directly after initialization of activation
			/* double cost = cost(activations[activations.length - 1], expected[i]); */
			batchCost += cost(activations[activations.length - 1], expected[i]);

			double[][] activationGradients = new double[activations.length][];

			// activations
			// [-1]- 0, 0
			//
			// [0] - 0, 0, 0
			// [1] - 0, 0, 0
			// [2] - 0

			// activation gradients
			// 0, 0
			// [0] - 0, 0, 0
			// [1] - 0, 0, 0
			// [2] - found

			// weights[layer, initialNeuron, outgoingNeuron];

			// initialize gradients of output layer
			int finalLayerIndex = activations.length - 1;
			activationGradients[finalLayerIndex] = new double[activations[finalLayerIndex].length];
			for (int j = 0; j < activationGradients[finalLayerIndex].length; j++) {
				activationGradients[finalLayerIndex][j] = 2 * (activations[finalLayerIndex][j] - expected[i][j]);
			}

			// intializatino of gradient of the rest of the layers
			// j is index of layer
			for (int j = finalLayerIndex - 1; j >= 0; j--) {
				// System.out.println("\nj: " + j + "\tactivation lengths: " +
				// activations[j].length);
				activationGradients[j] = new double[activations[j].length];

				// k is index of current activation neural
				for (int k = 0; k < activationGradients[j].length; k++) {
					// l is index of activation in layer ahead
					for (int l = 0; l < activations[j + 1].length; l++) {
						// stem.out.println("k: " + k + "\tl: " + l);
						activationGradients[j][k] += weights[j + 1][k][l] * z[j + 1][l] * activationGradients[j + 1][l];
					}
				}
			}

			// k is initial neuron
			for (int k = 0; k < weightGradient[0].length; k++) {
				// j is outgoing neuron
				for (int j = 0; j < weightGradient[0][k].length; j++) {
					weightGradient[0][k][j] += input[i][k] * z[0][j] * activationGradients[0][j];
				}
			}

			for (int l = 1; l < biasGradient.length; l++) {
				for (int j = 0; j < biasGradient[l].length; j++) {
					biasGradient[l][j] += z[l][j] * activationGradients[l][j];
				}
			}

			// l is layer
			for (int l = 1; l < weightGradient.length; l++) {
				// k is initial neuron
				for (int k = 0; k < weightGradient[l].length; k++) {
					// j is outgoing neuron
					for (int j = 0; j < weightGradient[l][k].length; j++) {
						// weightGradient[l][k][j] = 0 * z[0][j] * activationGradients[0][j];
						weightGradient[l][k][j] += activations[l - 1][k] * z[l][j] * activationGradients[l][j];
					}
				}
			}

		}

		// multiplication is faster than division
		// den is found so i only have to divide once
		double den = 1 / (double) (input.length);

		// applies the average of the gradients to weights/biases array
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++)
				for (int k = 0; k < weights[i][j].length; k++)
					weights[i][j][k] += weightGradient[i][j][k] * den * -learningRate;
			for (int j = 0; j < biases[i].length; j++)
				biases[i][j] += biasGradient[i][j] * den * -learningRate;
		}

		return batchCost * den;
	}

	public double[] getZ(double[] input, int layer) {
		return toArray(toMatrix(input).mult(weightsMatrix[layer]).plus(biasMatrix[layer]));
	}

	public static double cost(double[] output, double[] expected) {
		double sum = 0;
		for (int i = 0; i < output.length; i++)
			sum += Math.pow((expected[i] - output[i]), 2);
		return sum;
	}

	public static double[] initializeArray(int length, double input) {
		double[] arr = new double[length];
		for (int i = 0; i < arr.length; i++)
			arr[i] = input;
		return arr;
	}

	public static double[][] sigmoid(double[][] x) {
		double[][] output = new double[x.length][];
		for (int i = 0; i < output.length; i++)
			output[i] = sigmoid(x[i]);
		return output;
	}

	public static double sigmoidPrime(double x) {
		return Math.pow(Math.E, -x) / Math.pow(1 + Math.pow(Math.E, -x), 2);
	}

}
