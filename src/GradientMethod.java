import java.util.ArrayList;

public class GradientMethod {
	private GradientMethod() {};
	
	public static double numericalDiff(double x) {
		double h = 1e-4d;
		return (GradientMethod.function(x+h)-GradientMethod.function(x-h)) / (2.0d*h);
	}
	
	public static double function(double x) {
		return x*x + 3.0*3.0;
	}
	
	public static double function(LinerArray input, ArrayList<LinerArray> t) {
		double tmp = 0.0d;
		for(int row=0; row<input.getRowSize(); row++) {
			for(int column=0; column<input.getColumnSize(); column++) {
				tmp += LossFunctions.crossEntropyError(input, t);
			}
		}
		return tmp;
	}
	
	public static LinerArray numericalGradient(LinerArray array, ArrayList<LinerArray> t) {
		double h = 1e-4d;
		double fxh1 = 0.0d, fxh2 = 0.0d;
		double tmpVal = 0.0d;
		LinerArray tmp = LinerArray.copy(array);
		LinerArray grad = new LinerArray(array.getShape());
		
		for(int row=0; row<array.getRowSize(); row++) {
			for(int column=0; column<array.getColumnSize(); column++) {
				tmpVal = array.getElement(row, column);
				tmp.setElement(row, column, tmpVal + h);
				fxh1 = GradientMethod.function(tmp, t);
				tmp.setElement(row, column, tmpVal - h);
				fxh2 = GradientMethod.function(tmp, t);
				System.out.println(tmpVal + h);
				grad.setElement(row, column, (fxh1-fxh2)/(2.0d*h));
				tmp.setElement(row, column, tmpVal);
			}
		}
		return grad;
	}
	
	public static LinerArray gradientDescent(LinerArray input,ArrayList<LinerArray> t, double lr, int stepNum) {
		LinerArray tmp = LinerArray.copy(input);
		LinerArray grad = LinerArray.copy(input);
		
		for(int i=0; i<stepNum; i++) {
			grad = GradientMethod.numericalGradient(tmp, t);
		
			for(int row=0; row<input.getRowSize(); row++) {
				for(int column=0; column<input.getColumnSize(); column++) {
					tmp.setElement(row, column, (tmp.getElement(row, column) - lr*grad.getElement(row, column)));
				}
			}
		}
		return tmp;
	}
	
	public static void printDiff(double x) {
		System.out.println(GradientMethod.numericalDiff(x));
	}

}


