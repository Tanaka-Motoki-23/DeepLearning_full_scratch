
public class Affine implements Layer{
	private LinerArray x, W ,b;
	private LinerArray dx, dW, db;

	public Affine(LinerArray W, LinerArray b) {
		this.W = W;
		this.b = b;
	}
	
	@Override
	public LinerArray forward(LinerArray x) {
		this.x = x;
		return LinerArray.add(LinerArray.dot(this.x, this.W), this.b);
	}

	@Override
	public void backward(LinerArray dOut) {
		this.dx = LinerArray.dot(dOut, LinerArray.T(this.W));
		this.dW = LinerArray.dot(LinerArray.T(this.x), dOut);
		this.db = this.sumAxis(dOut);
	}
	
	public LinerArray sumAxis(LinerArray dOut) {
		LinerArray sum = new LinerArray(LinerArray.initShape(1, dOut.getColumnSize()));
		double tmp = 0.0d;
		for(int column=0; column<dOut.getColumnSize(); column++) {
			tmp = 0.0d;
			for(int row=0; row<dOut.getRowSize(); row++) {
				tmp += dOut.getElement(row, column);
			}
			sum.setElement(0, column, tmp);
		}
		return sum;
	}
	
	public LinerArray getDx() {
		return this.dx;
	}
	
	public LinerArray getDW() {
		return this.dW;
	}
	
	public LinerArray getDb() {
		return this.db;
	}
}
