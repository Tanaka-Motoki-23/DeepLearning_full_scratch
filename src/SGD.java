
public class SGD implements Optimizer{
	private double lr;
	
	public SGD(double lr) {
		this.lr = lr;
	};
	
	@Override
	public void update(LinerArray array, LinerArray grad) {
		for(int row=0; row<array.getRowSize(); row++) {
			for(int column=0; column<array.getColumnSize(); column++) {
				array.setElement(row, column, (array.getElement(row, column) - lr*grad.getElement(row, column)));		
			}
		}
	}
	
}
