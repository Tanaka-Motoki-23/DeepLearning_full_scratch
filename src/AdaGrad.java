
public class AdaGrad implements Optimizer{
	private double lr;
	private LinerArray h;

	public AdaGrad(double lr, int[] shape) {
		this.lr = lr;
		this.h = new LinerArray(shape);
		for(int row=0; row<this.h.getRowSize(); row++) {
			for(int column=0; column<this.h.getColumnSize(); column++) {
				this.h.setElement(row, column, 0.0d);
			}
		}
	}
	
	@Override
	public void update(LinerArray array, LinerArray grad) {
		for(int row=0; row<this.h.getRowSize(); row++) {
			for(int column=0; column<this.h.getColumnSize(); column++) {
				this.h.setElement(row, column, (this.h.getElement(row, column) + (grad.getElement(row, column)*grad.getElement(row, column))));
			}
		}
		for(int row=0; row<array.getRowSize(); row++) {
			for(int column=0; column<array.getColumnSize(); column++) {
				array.setElement(row, column, (array.getElement(row, column) - (this.lr*grad.getElement(row, column))/(Math.sqrt(this.h.getElement(row, column))+1e-7d)));		
			}
		}
		
	}

}
