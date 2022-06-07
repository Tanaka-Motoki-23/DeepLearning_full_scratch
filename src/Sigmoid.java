
public class Sigmoid implements Layer{
	private LinerArray y;
	private LinerArray x;
	private LinerArray dx;
	
	public Sigmoid() {};
	
	@Override
	public LinerArray forward(LinerArray x) {
		this.x = x;
		LinerArray tmp = new LinerArray(this.x.getShape());
		ActivateFunctions.sigmoid(this.x, tmp);
		this.y = LinerArray.copy(tmp);
		return tmp;
	}
	
	@Override
	public void backward(LinerArray dOut) {
		this.dx = new LinerArray(dOut.getShape());
		for(int row=0; row<dOut.getRowSize(); row++) {
			for(int column=0; column<dOut.getColumnSize(); column++) {
				double tmp = dOut.getElement(row, column) * (1.0d - this.y.getElement(row, column)) * this.y.getElement(row, column);
 				this.dx.setElement(row, column, tmp);
			}
		}
	}
	
	public LinerArray getDx() {
		return this.dx;
	}
}
