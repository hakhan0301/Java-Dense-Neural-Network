package IndividualNetworks;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Scanner;

import NeuralNets.BackProp;

public class MNIST {

	//location of MNIST Dataset
	public static String fileStart = "MNIST Dataset/Original/";
	public static String imagesFile = fileStart + "train-images.idx3-ubyte";
	public static String labelsFile = fileStart + "train-labels.idx1-ubyte";

	public File saveFile;
	public BackProp n;

	public double[][] images;
	public double[][] labels;

	public MNIST(String saveLoc) throws FileNotFoundException {
		this(new File(saveLoc));
	}

	public MNIST(File saveLoc) throws FileNotFoundException {
		saveFile = saveLoc;
		this.n = new BackProp(saveLoc);
	}

	public void train(int batchSize, double trainingRate) {
		int index = 0;
		for (int i = 0; i < labels.length; i += batchSize) {
			double[][] imageBatch = new double[batchSize][];
			double[][] labelsBatch = new double[batchSize][];
			for (int j = 0; j < batchSize; j++) {
				imageBatch[j] = images[index];
				labelsBatch[j] = labels[index++];
			}
			System.out.println(n.batchTrain(imageBatch, labelsBatch));
		}
	}

	public static void downloadFile(String location) throws IOException {
		downloadFile(new File(location));
	}

	public void loadFile(File location) throws IOException {
		Scanner sc = new Scanner(location);
		int length = sc.nextInt();
		sc.nextLine();

		images = new double[length][];
		labels = new double[length][];
		for (int i = 0; i < length; i++) {
			String[] pixels = sc.nextLine().split(" ");
			images[i] = new double[784];
			labels[i] = new double[10];
			for (int j = 0; j < images[i].length; j++) {
				images[i][j] = Double.parseDouble(pixels[j]);
			}
			int labelNumber = Integer.parseInt(sc.nextLine());
			for (int j = 0; j < labels[i].length; j++) {
				labels[i][j] = (j == labelNumber) ? 1 : 0;
			}
		}
		n.saveToFile(saveFile);
		sc.close();
	}

	public static void downloadFile(File location) throws IOException {
		if (location.exists()) {
			location.delete();
			location.createNewFile();
		}
		PrintWriter pw = new PrintWriter(location);

		int[] labels = MnistReader.getLabels(labelsFile);
		List<int[][]> images = MnistReader.getImages(imagesFile);

		System.out.println(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());

		pw.println(images.size());

		for (int i = 0; i < labels.length; i++) {
			for (int j = 0; j < images.get(i).length; j++) {
				for (int k = 0; k < images.get(i)[j].length; k++)
					pw.print(images.get(i)[j][k] + " ");
			}
			pw.println();
			pw.println(labels[i]);
		}
		pw.close();
	}

	public static void train(BackProp n, File f) {

	}

	public static void main(String[] args) throws IOException {
		File file = new File("C:/Users/s592100/Desktop/7 - Independant Study/MNIST Dataset/output.txt");
		// downloadFile(file);

		File saveFile = new File("C:/Users/s592100/Desktop/7 - Independant Study/MNIST Dataset/NeuralnetSaveFile.txt");

		MNIST mnist = new MNIST(saveFile);

		mnist.loadFile(file);

		mnist.train(20, .07);
		// mnist.n.saveToFile(saveFile);
	}
}
