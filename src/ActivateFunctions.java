
public class ActivateFunctions {
	private ActivateFunctions() {};
	
	public static double identity_function(double x) {
		return x;
	}
	
	public static void identity_function(LinerArray input, LinerArray output) {
		for(int row=0; row<input.getRowSize(); row++) {
			for(int column=0; column<input.getColumnSize(); column++) {
				output.setElement(row, column, input.getElement(row, column));		
			}
		}
	}
	
	public static double step_function(double x) {
		if(x > 0) {
			return 1.0d;
		}else {
			return 0.0d;
		}
	}
	
	public static void step_function(LinerArray input, LinerArray output) {
		for(int row=0; row<input.getRowSize(); row++) {
			for(int column=0; column<input.getColumnSize(); column++) {
				if(input.getElement(row, column) > 0) {
					output.setElement(row, column, 1.0d);
				}else {
					output.setElement(row, column, 0.0d);
				}
			}
		}
	}
	
	public static double sigmoid(double x) {
		return 1.0 / (1.0d + Math.exp(-1*x));
	}
	
	public static void sigmoid(LinerArray input, LinerArray output) {
		for(int row=0; row<input.getRowSize(); row++) {
			for(int column=0; column<input.getColumnSize(); column++) {
				output.setElement(row, column, 1.0d / (1.0d + Math.exp(-1*(input.getElement(row, column)))));		
			}
		}
	}
	
	public static double ReLU(double x) {
		if(x > 0) {
			return x;
		}else {
			return 0.0d;
		}
	}
	
	public static void ReLU(LinerArray input, LinerArray output) {
		for(int row=0; row<input.getRowSize(); row++) {
			for(int column=0; column<input.getColumnSize(); column++) {
				if(input.getElement(row, column) > 0) {
					output.setElement(row, column, input.getElement(row, column));
				}else {
					output.setElement(row, column, 0.0d);
				}
			}
		}
	}
	
	public static void softmax(LinerArray input, LinerArray output) {
		double sum_exp = 0.0d;
		double C = input.max();
		for(int row=0; row<input.getRowSize(); row++) {
			sum_exp = 0.0d;
			for(int column=0; column<input.getColumnSize(); column++) {
					sum_exp += Math.exp(input.getElement(row, column) - C);
			}
			for(int column=0; column<input.getColumnSize(); column++) {
				output.setElement(row, column, Math.exp(input.getElement(row, column) - C)/sum_exp);
			}
		}
	}
}
