package NeuralNets;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

public class NeuralNet {

	// number of neurons per layer
	public final int[] layers;

	public enum Types {
		backProp
	};

	public double[][][] weights;
	public double[][] biases;
	// these are seperated to allow for ease in backProp and activation
	public SimpleMatrix[] weightsMatrix;
	public SimpleMatrix[] biasMatrix;

	public static void main(String[] args) {
		// NeuralNet n = new NeuralNet(new Random(12), new int[] { 2, 3, 1 });

	}

	public NeuralNet(File f) throws FileNotFoundException {
		Scanner sc = new Scanner(f);
		String[] str = sc.nextLine().split(" ");
		this.layers = new int[str.length];
		for (int i = 0; i < str.length; i++) {
			layers[i] = Integer.parseInt(str[i]);
		}

		weights = new double[layers.length - 1][][];
		biases = new double[layers.length - 1][];
		weightsMatrix = new SimpleMatrix[layers.length - 1];
		biasMatrix = new SimpleMatrix[layers.length - 1];

		for (int i = 0; i < layers.length - 1; i++) {
			weights[i] = new double[layers[i]][];
			for (int j = 0; j < layers[i]; j++) {
				weights[i][j] = new double[layers[i + 1]];
				for (int k = 0; k < weights[i][j].length; k++) {
					weights[i][j][k] = sc.nextDouble();
				}
			}
		}

		for (int i = 0; i < layers.length - 1; i++) {
			biases[i] = new double[layers[i + 1]];
			for (int j = 0; j < biases[i].length; j++)
				biases[i][j] = sc.nextDouble();
			weightsMatrix[i] = toMatrix(weights[i]);
			biasMatrix[i] = toMatrix(biases[i]);
		}
		sc.close();
	}

	public NeuralNet(Random r, int[] layers) {
		// random is seed for gaussian
		this.layers = layers;

		// rows are weights from activation[k] to neuron[n] weights[l, k, n] is the
		// weight from neuron[k] in layer[l] to neuron[n] in layer[l + 1]

		// weights[layer, initialNeuron, outgoingNeuron];

		// bias isn't 3 dimensional, because it only has neuron leading to the next
		// layer

		weights = new double[layers.length - 1][][];
		biases = new double[layers.length - 1][];
		weightsMatrix = new SimpleMatrix[layers.length - 1];
		biasMatrix = new SimpleMatrix[layers.length - 1];

		for (int i = 0; i < layers.length - 1; i++) {
			// initialization of dimension 2 and 3 of array
			weights[i] = new double[layers[i]][];
			for (int j = 0; j < layers[i]; j++) {
				weights[i][j] = initializeArray(r, layers[i + 1]);
			}
			// biases is initialized directly because there it's only 2 dimensional
			biases[i] = initializeArray(r, layers[i + 1]);

			// rows is length of layers[i]
			// column is length of layers[i + 1]
			weightsMatrix[i] = toMatrix(weights[i]);
			// rows is set to 1
			// columns is set to length of numbers in array
			biasMatrix[i] = toMatrix(biases[i]);
		}
		// rows and columns of matrices must be exact for multiplication and addition

		// System.out.println("weights");
		// printArray(weights);
		// System.out.println("biases");
		// printArray(biases);

		// double[] input = { 0, 1 };
		//
		// new ETime();
		// printArray(activate(input));
		// ETime.printMilli();

		// System.out.println("biases");
		// printArray(biases);
	}

	// loops through all layers, and applies activateLayer
	public double[] activate(double[] input) {
		double[] output = activateLayer(input, 0);
		for (int i = 1; i < layers.length - 1; i++)
			// passes output of previous layer to activateLayer() to get output of next
			// layer
			output = activateLayer(output, i);
		return output;
	}

	public double[] activateLayer(double[] input, int layer) {
		// turns input to matrix
		// multiplies input by weight matrix
		// adds resulting matrix with bias
		// turns result back to array
		// applies sigmoid to array

		return sigmoid(toArray(toMatrix(input).mult(weightsMatrix[layer]).plus(biasMatrix[layer])));
		// input is turned back into array because it makes backProp easier when you
		// need input from indvidual layers and also to apply sigmoid
		// for lengthier weight lengths, it might slow down training, probably not too
		// much though
	}

	public void initializeMatrices() {
		for (int i = 0; i < layers.length - 1; i++) {
			weightsMatrix[i] = toMatrix(weights[i]);
			biasMatrix[i] = toMatrix(biases[i]);
		}
	}

	// returns array of "gaussians"
	public static double[] initializeArray(Random r, int length) {
		double[] arr = new double[length];
		for (int i = 0; i < arr.length; i++)
			arr[i] = gaussian(r);
		return arr;
	}

	// applies sigmoid to whole array
	public static double[] sigmoid(double[] x) {
		double[] output = new double[x.length];
		for (int i = 0; i < output.length; i++)
			output[i] = sigmoid(x[i]);
		return output;
	}

	public static double sigmoid(double x) {
		// greater than/less than check prevents errors in backProp which occur if x is
		// too high
		if (x < -50)
			return 0;
		if (x > 50)
			return 1;
		return 1 / (1 + Math.pow(Math.E, -x));
	}

	// self explanitory
	public static double[] toArray(SimpleMatrix matrix) {
		double[] output = new double[matrix.numCols()];
		for (int i = 0; i < output.length; i++)
			output[i] = matrix.get(i);
		return output;
	}

	// used for weights
	public static SimpleMatrix toMatrix(double[][] array) {
		// array.length is length of specified layer, l
		// array[0].length is length of l + 1
		SimpleMatrix m = new SimpleMatrix(array.length, array[0].length);
		// refer above for weightsMatrix shape
		for (int i = 0; i < array.length; i++)
			m.setRow(i, 0, array[i]);
		return m;
	}

	// used for inputs and biases
	public static SimpleMatrix toMatrix(double[] array) {
		// new matrix with row size 1 and column to length of array
		SimpleMatrix m = new SimpleMatrix(1, array.length);
		// fills the only row with contents of array
		m.setRow(0, 0, array);
		return m;
	}

	public static void printArray(double[][][] array) {
		for (double[][] d : array)
			printArray(d);
	}

	public static void printArray(double[][] array) {
		for (double[] d : array)
			printArray(d);
		System.out.println();
	}

	public static void printArray(int[] array) {
		for (int d : array)
			System.out.print(d + " ");
		System.out.println();
	}

	public static void printArray(double[] array) {
		for (double d : array)
			System.out.print(d + " ");
		System.out.println();
	}

	// not typical "gaussian"
	// returns a random from -1 to 1
	public static double gaussian(Random r) {
		return r.nextDouble() * 2 - 1;
	}

	public void saveToFile(File f) throws IOException {
		if (f.exists()) {
			f.delete();
			f.createNewFile();
		}

		PrintWriter pw = new PrintWriter(f);
		for (int i = 0; i < layers.length; i++) {
			pw.print(layers[i] + " ");
		}
		pw.println();
		pw.println();

		for (int i = 0; i < weights.length; i++)
			for (int j = 0; j < weights[i].length; j++)
				for (int k = 0; k < weights[i][j].length; k++)
					pw.print(weights[i][j][k] + " ");
		pw.println();

		for (int i = 0; i < biases.length; i++)
			for (int j = 0; j < biases[i].length; j++)
				pw.print(biases[i][j] + " ");
		pw.println();

		pw.close();
	}

	public void saveToFile(String str) throws IOException {
		saveToFile(new File(str));
	}

}
