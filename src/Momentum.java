
public class Momentum implements Optimizer{
	private double lr;
	private double momentum;
	private LinerArray v;
	
	public Momentum(double lr, double momentum, int[] shape) {
		this.lr = lr;
		this.momentum = momentum;
		this.v = new LinerArray(shape);
		for(int row=0; row<this.v.getRowSize(); row++) {
			for(int column=0; column<this.v.getColumnSize(); column++) {
				this.v.setElement(row, column, 0.0d);
			}
		}
	}
	
	@Override
	public void update(LinerArray array, LinerArray grad) {
		for(int row=0; row<this.v.getRowSize(); row++) {
			for(int column=0; column<this.v.getColumnSize(); column++) {
				this.v.setElement(row, column, (this.v.getElement(row, column)*this.momentum) - (this.lr*grad.getElement(row, column)));
			}
		}
		for(int row=0; row<array.getRowSize(); row++) {
			for(int column=0; column<array.getColumnSize(); column++) {
				array.setElement(row, column, (array.getElement(row, column) + this.v.getElement(row, column)));		
			}
		}
		
	}
}
