package IndividualNetworks;

import java.util.Random;

import NeuralNets.NeuralNet;

public class Evolution {

	public double[] data;
	public int generationSize = 10;

	public static void main(String[] args) {
		NeuralNet n = new NeuralNet(new Random(12), new int[] { 2, 3, 1 });

		NeuralNet.printArray(n.weights);
		NeuralNet.printArray(n.biases);
		NeuralNet.printArray(combine(flatten(n.weights), flatten(n.biases)));
	}

	public static double[] flatten(double[][] array) {
		double[] output = new double[getLength(array)];
		int index = 0;
		for (int i = 0; i < array.length; i++)
			for (int j = 0; j < array[i].length; j++)
				output[index++] = array[i][j];
		return output;
	}

	public static double[] flatten(double[][][] array) {
		double[] output = new double[getLength(array)];
		int index = 0;
		for (int i = 0; i < array.length; i++)
			for (int j = 0; j < array[i].length; j++)
				for (int k = 0; k < array[i][j].length; k++)
					output[index++] = array[i][j][k];
		return output;
	}

	public static int getLength(double[][][] array) {
		int sum = 0;
		for (int i = 0; i < array.length; i++)
			sum += getLength(array[i]);
		return sum;
	}

	public static int getLength(double[][] array) {
		int sum = 0;
		for (int i = 0; i < array.length; i++)
			sum += array[i].length;
		return sum;
	}

	public static double[] combine(double[] array1, double[] array2) {
		double[] output = new double[array1.length + array2.length];
		int index = 0;
		for (int i = 0; i < array1.length; i++)
			output[index++] = array1[i];
		for (int i = 0; i < array2.length; i++)
			output[index++] = array2[i];
		return output;
	}

}
