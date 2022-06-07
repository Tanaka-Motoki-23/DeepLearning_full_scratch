import java.util.ArrayList;
public abstract class NetworkWithBackPropagation {
	protected int classes;
	protected int inputColumnSize;
	
	public NetworkWithBackPropagation(int inputColumnSize, int classes) {
		this.inputColumnSize = inputColumnSize;
		this.classes = classes;
	}

	public abstract void initNetwork();
	
	public abstract LinerArray predict(LinerArray X);
	
	public abstract LinerArray forward(LinerArray xBatch, ArrayList<LinerArray> tBatch);
	
	public abstract void gradient(LinerArray x, ArrayList<LinerArray> t);
	
	public abstract void gradientDescent(LinerArray xBatch, ArrayList<LinerArray> tBatch, int stepNum);
	
	public void fit(Dataset dataset, int batchSize, int epochs) {
		dataset.shuffle();
		LinerArray xBatch = new LinerArray(LinerArray.initShape(batchSize, dataset.getImgShape()));
		ArrayList<LinerArray> tBatch;
		
		System.out.println("Learning...");
		for(int nbEpoch=0; nbEpoch<epochs; nbEpoch++) {
			dataset.shuffle();
			for(int step=0; step<dataset.getNumData(); step+=batchSize) {
				System.out.println("Loss = "+ this.calcDatasetLoss(dataset));
				tBatch = new ArrayList<LinerArray>();
				this.extractBatch(xBatch, tBatch, dataset, step, batchSize);
				System.out.println("Step => " + step + "/" + dataset.getNumData());			
				this.gradientDescent(xBatch, tBatch, 100);
			}
		}
	}

	public void extractBatch(LinerArray xBatch, ArrayList<LinerArray> tBatch, Dataset dataset, int step, int batchSize) {
		for(int i=0; i<batchSize; i++) {
			tBatch.add(dataset.getData(step+i).getLabel());
			for(int element=0; element<dataset.getImgShape(); element++) {
				xBatch.setElement(i, element, dataset.getData(step+i).getImage().getElement(0,element));
			}
		}
	}
	
	public int[] getReturnShape(LinerArray input) {
		return LinerArray.initShape(input.getRowSize(), this.classes);
	}
	
	public double calcAccuracy(LinerArray yBatch, ArrayList<LinerArray> tBatch) {
		double accuracy = 0.0d;
		for(int i=0; i<yBatch.getRowSize(); i++) {
			if(yBatch.maxIndex(i) == tBatch.get(i).maxIndex(0)) {
				accuracy += 1.0d;
			};
		}
		return accuracy /= (double)yBatch.getRowSize();
	}
	
	public double calcDatasetAccuracy(Dataset dataset, int batchSize) {
		double accuracy = 0.0d;
		dataset.shuffle();
		LinerArray xBatch = new LinerArray(LinerArray.initShape(batchSize, dataset.getImgShape()));
		LinerArray yBatch = new LinerArray(this.getReturnShape(xBatch));
		ArrayList<LinerArray> tBatch;
		
		for(int step=0; step<dataset.getNumData(); step+=batchSize) {
			tBatch = new ArrayList<LinerArray>();
			this.extractBatch(xBatch, tBatch, dataset, step, batchSize);
			System.out.println("Step => " + step + "/" + dataset.getNumData());
			yBatch = this.predict(xBatch);
			accuracy += this.calcAccuracy(yBatch, tBatch);	
		}
	accuracy /= ((double)dataset.getNumData()/(double)batchSize);
		return accuracy;
	}
	
	public double calcDatasetLoss(Dataset dataset) {
		double loss = 0.0d;
		LinerArray x = new LinerArray(LinerArray.initShape(dataset.getNumData(), dataset.getImgShape()));
		ArrayList<LinerArray> t = new ArrayList<LinerArray>();
		for(int i=0; i<dataset.getNumData(); i++) {
			t.add(dataset.getData(i).getLabel());
			for(int element=0; element<dataset.getImgShape(); element++) {
				x.setElement(i, element, dataset.getData(i).getImage().getElement(0,element));
			}
		}
		loss = this.forward(x, t).getElement(0, 0);
		return loss;
	}
	
	public int getClasses() {
		return this.classes;
	}
	
	public int getInputColumnSize() {
		return this.inputColumnSize;
	}
}

