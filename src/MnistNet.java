
import java.util.ArrayList;

public class MnistNet extends NetworkWithBackPropagation{
	private LinerArray W1, b1, W2, b2;	
	private LinerArray gradW1, gradb1, gradW2, gradb2;
	private Affine affine1, affine2;
	private ReLU relu1;
	private SoftmaxWithLoss softmaxWithLoss;
	
	public MnistNet(int inputColumnSize, int classes) {
		super(inputColumnSize, classes);
		this.initNetwork();
	};
	
	@Override
	public void initNetwork() {
		
		int neuron1 = 100;
		int neuron2 = super.getClasses();
		
		this.W1 = new LinerArray(LinerArray.initShape(super.getInputColumnSize(), neuron1));
		this.W1 = LinerArray.initParameter(this.W1, Math.sqrt(2.0d/(double)this.W1.getRowSize()));
		this.b1 = new LinerArray(LinerArray.initShape(1, neuron1));	
		this.b1 = LinerArray.initParameter(this.b1, Math.sqrt(2.0d/(double)this.b1.getRowSize()));
		
		this.W2 = new LinerArray(LinerArray.initShape(neuron1, neuron2));
		this.W2 = LinerArray.initParameter(this.W2, Math.sqrt(2.0d/(double)this.W2.getRowSize()));
		this.b2 = new LinerArray(LinerArray.initShape(1, neuron2));
		this.b2 = LinerArray.initParameter(this.b2, Math.sqrt(2.0d/(double)this.b2.getRowSize()));
	
		this.affine1 = new Affine(this.W1, this.b1);
		this.affine2 = new Affine(this.W2, this.b2);
		this.relu1 = new ReLU();
		this.softmaxWithLoss = new SoftmaxWithLoss();
	};
	
	@Override
	public LinerArray predict(LinerArray X) { 
		LinerArray tmp = LinerArray.copy(X);
		tmp = this.affine1.forward(tmp);
		tmp = this.relu1.forward(tmp);

		tmp = this.affine2.forward(tmp);
		tmp = this.softmaxWithLoss.calcScore(tmp);
		return tmp;
	}
	
	@Override
	public LinerArray forward(LinerArray xBatch, ArrayList<LinerArray> tBatch) {
		LinerArray tmp = LinerArray.copy(xBatch);
		this.softmaxWithLoss.setT(tBatch);
		
		tmp = this.affine1.forward(LinerArray.copy(tmp));
		tmp = this.relu1.forward(LinerArray.copy(tmp));
	
		tmp = this.affine2.forward(LinerArray.copy(tmp));
		tmp = this.softmaxWithLoss.forward(LinerArray.copy(tmp));
		return tmp;
	}
	
	@Override
	public void gradient(LinerArray x, ArrayList<LinerArray> t) {
		LinerArray dOut = new LinerArray(x.getShape());
		for(int row=0; row<dOut.getRowSize(); row++) {
			for(int column=0; column<dOut.getColumnSize(); column++) {
				dOut.setElement(row, column, 1.0d);
			}
		}
		this.softmaxWithLoss.backward(LinerArray.copy(dOut));
		dOut = this.softmaxWithLoss.getDx();
		this.affine2.backward(LinerArray.copy(dOut));
		dOut = this.affine2.getDx();		
	
		this.relu1.backward(LinerArray.copy(dOut));
		dOut = this.relu1.getDx();
		this.affine1.backward(LinerArray.copy(dOut));
		
		this.gradW1 = this.affine1.getDW();
		this.gradb1 = this.affine1.getDb();
		
		this.gradW2 = this.affine2.getDW();
		this.gradb2 = this.affine2.getDb();
			
	}
	
	@Override
	public void gradientDescent(LinerArray xBatch, ArrayList<LinerArray> tBatch, int stepNum) {
		Optimizer adaGradW1 = new AdaGrad(0.1d, this.W1.getShape());
		Optimizer adaGradb1 = new AdaGrad(0.1d, this.b1.getShape());
	
		Optimizer adaGradW2 = new AdaGrad(0.1d, this.W2.getShape());
		Optimizer adaGradb2 = new AdaGrad(0.1d, this.b2.getShape());
		
		for(int i=0; i<stepNum; i++) {	
			this.forward(xBatch, tBatch);
			this.gradient(xBatch, tBatch);
			adaGradW1.update(this.W1, this.gradW1);
			adaGradb1.update(this.b1, this.gradb1);
			
			adaGradW2.update(this.W2, this.gradW2);
			adaGradb2.update(this.b2, this.gradb2);
			
			this.affine1 = new Affine(this.W1, this.b1);
			this.affine2 = new Affine(this.W2, this.b2);
		}
	}
}

