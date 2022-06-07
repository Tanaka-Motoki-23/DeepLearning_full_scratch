import java.text.NumberFormat;

public class Main {
	
	public static void main(String[] args) {
		
		int batchSize = 100;
		Dataset trainDataset = new Dataset("./trainImg.csv","./trainLabel.csv", 10, 10000);
		Dataset testDataset = new Dataset("./testImg.csv","./testLabel.csv", 10, 1000);
		MnistNet network = new MnistNet(trainDataset.getImgShape(), trainDataset.getClasses());
		network.fit(trainDataset, batchSize, 1);
		
		System.out.println("Calculating train accuracy...");
		NumberFormat nfPer = NumberFormat.getPercentInstance();
	    System.out.println("train accuracy = " + nfPer.format(network.calcDatasetAccuracy(trainDataset, batchSize)));
		
		System.out.println("Calculating test accuracy...");
	    System.out.println("test accuracy = " + nfPer.format(network.calcDatasetAccuracy(testDataset, batchSize)));
		
	}	
}
