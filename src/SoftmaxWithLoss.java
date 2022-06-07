import java.util.ArrayList;

public class SoftmaxWithLoss implements Layer{
	private LinerArray x, y;
	private ArrayList<LinerArray> t;
	private LinerArray dx;
	
	public SoftmaxWithLoss() {};
	
	@Override
	public LinerArray forward(LinerArray x){
		this.x = x;
		this.y = new LinerArray(this.x.getShape());
		LinerArray loss = new LinerArray(LinerArray.initShape(1, 1));
		ActivateFunctions.softmax(this.x, this.y);
		loss.setElement(0, 0, LossFunctions.crossEntropyError(this.y, this.t));
		return  loss;
	}
	
	@Override
	public void backward(LinerArray dOut) {
		this.dx = new LinerArray(this.x.getShape());
		int batchSize = this.t.size();
		for(int i=0; i<t.size(); i++) {
			for(int column=0; column<y.getColumnSize(); column++) {
				this.dx.setElement(i, column, (this.y.getElement(i, column) - t.get(i).getElement(0, column))/(double)batchSize);
			}
		}
	}
	
	public LinerArray getDx() {
		return this.dx;
	}
	
	public LinerArray calcScore(LinerArray x) {
		this.x = x;
		this.y = new LinerArray(this.x.getShape());
		ActivateFunctions.softmax(this.x, this.y);
		//return LinerArray.copy(this.y);
		return this.y;
	}
	
	public void setT(ArrayList<LinerArray> t) {
		this.t = t;
	}
	
	public LinerArray getY(){
		return this.y;
	}
	
}
