
public class ReLU implements Layer{
	private LinerArray mask;
	private LinerArray x;
	private LinerArray dx;
	
	public ReLU() {};
	
	public void initMask() {
		this.mask = new LinerArray(this.x.getShape());
		for(int row=0; row<this.x.getRowSize(); row++) {
			for(int column=0; column<this.x.getColumnSize(); column++) {
				if(this.x.getElement(row, column) > 0) {
					this.mask.setElement(row, column, 0.0d);
				}else {
					this.mask.setElement(row, column, 1.0d);
				}		
			}
		}
	}
	
	@Override
	public LinerArray forward(LinerArray x){
		this.x = x;
		LinerArray out = new LinerArray(this.x.getShape());
		this.initMask();
		ActivateFunctions.ReLU(this.x, out);
		return out;
	}
	
	@Override
	public void backward(LinerArray dOut) {
		this.dx = new LinerArray(dOut.getShape());
		for(int row=0; row<dOut.getRowSize(); row++) {
			for(int column=0; column<dOut.getColumnSize(); column++) {
				if(mask.getElement(row, column) == 0.0d) {
					this.dx.setElement(row, column, (dOut.getElement(row, column)));
				}else {
					this.dx.setElement(row, column, 0.0d);
				}
			}
		}
	}
	
	public LinerArray getDx() {
		return this.dx;
	}
}
