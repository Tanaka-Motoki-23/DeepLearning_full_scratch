import java.util.ArrayList;
import java.util.Random;

public class LinerArray {
	protected ArrayList<ArrayList<Double>> array = new ArrayList<ArrayList<Double>>();
	protected int[] shape;
	
	public LinerArray(int[] shape) {
		this.shape = shape;
		Random initRand = new Random();
		initRand.setSeed(0L);
		for(int i=0; i<shape.length; i++) {
		this.array.add(new ArrayList<Double>());
			for(int j=0; j<shape[i]; j++) {
				this.array.get(i).add(initRand.nextGaussian());
			}
		}
	}
	
	public ArrayList<Double> getRow(int row){
		return this.array.get(row);
	}
	
	public int getRowSize() {
		return this.shape.length;
	}
	
	public int getColumnSize() {
		return this.shape[0];
	}
	
	public int[] getShape() {
		return this.shape;
	}
	
	public void printArray() {
		System.out.println("[ ");
		for(ArrayList<Double> arr : this.array) {
			System.out.print("\t" + "[ ");
			for(double element : arr) {
				System.out.print(String.format("%f", element) + "\t");
			}
			System.out.println("]");
		}
		System.out.print("\t");
		this.printShape();
		System.out.println("]");
	}
	
	public void printShape() {
		System.out.print("[ ");
		System.out.print("Row:" + this.getRowSize() + ",");
		System.out.print("Column:" + this.getColumnSize());	
		System.out.println(" ]");
	}
	
	public void setElement(int row, int column, double element) {
		this.array.get(row).set(column, element);
	}
	
	public double getElement(int row, int column) {
		return 	this.array.get(row).get(column);
	}
	
	public double max() {
		double max = this.array.get(0).get(0);
		for(int row=0; row<this.getRowSize(); row++) {
			for(int column=0; column<this.getColumnSize(); column++) {
				if ( this.getElement(row, column) > max) {
					max = this.getElement(row, column);
				}
			}
		}
		return max;
	}
	
	public int maxIndex(int row) {
		int maxIndex = 0;
		double maxValue = this.array.get(row).get(0);
		for(int column=0; column<this.getColumnSize(); column++) {
			if ( this.getElement(row, column) > maxValue) {
				maxValue = this.getElement(row, column);
				maxIndex = column;
			}
		}
		return maxIndex;
	}
	
	public double sum() {
		double sum = 0.0d;
		for(int row=0; row<this.getRowSize(); row++) {
			for(int column=0; column<this.getColumnSize(); column++) {
					sum += this.getElement(row, column);
			}
		}
		return sum;
	}
	
	public static LinerArray add(LinerArray A, LinerArray B) {
		LinerArray C = new LinerArray(A.getShape());
		if(LinerArray.addCheckShape(A, B, C)) {
			if(A.getRowSize() == B.getRowSize()) {
				for(int row=0; row<A.getRowSize(); row++) {
					for(int column=0; column<A.getColumnSize(); column++) {
						C.setElement(row, column, A.getElement(row, column) + B.getElement(row, column));
					}
				}
			}else if(B.getRowSize() == 1){
				for(int row=0; row<A.getRowSize(); row++) {
					for(int column=0; column<A.getColumnSize(); column++) {
						C.setElement(row, column, A.getElement(row, column) + B.getElement(0, column));
					}
				}
			}
		}else {
			System.out.println("can't add");
			A.printShape();
			B.printShape();
		}
		return C;
	}
	
	public static boolean addCheckShape(LinerArray A, LinerArray B, LinerArray C) {
		if(B.getRowSize() == 1 && (A.getColumnSize() == B.getColumnSize())) {
			return true;
		}
		if(A.getRowSize() != B.getRowSize() || A.getRowSize() != C.getRowSize() || B.getRowSize() != C.getRowSize()) {
			return false;
		}else if(A.getColumnSize() != B.getColumnSize() || A.getColumnSize() != C.getColumnSize() || B.getColumnSize() != C.getColumnSize()) {
			return false;
		}
		return true;
	}
	
	public static LinerArray dot(LinerArray A, LinerArray B) {
		LinerArray C = new LinerArray(LinerArray.initShape(A.getRowSize(), B.getColumnSize()));
		if(LinerArray.dotCheckShape(A, B, C)) {
			double tmp = 0.0d;
			for(int row=0; row<A.getRowSize(); row++) {
				for(int column=0; column<B.getColumnSize(); column++) {
					tmp = 0.0d;
					for(int index=0; index<B.getRowSize(); index++) {
						tmp += A.getElement(row, index) * B.getElement(index, column);
					}
					C.setElement(row, column, tmp);
				}
			}
		}else {
			System.out.println("can't dot");
			A.printShape();
			B.printShape();
			
		}	
		return C;
	}
	
	public static boolean dotCheckShape(LinerArray A, LinerArray B, LinerArray C) {
		if(A.getColumnSize() != B.getRowSize() || A.getRowSize() != C.getRowSize() || B.getColumnSize() != C.getColumnSize()) {
			return false;
		}
		return true;
	}

	public static LinerArray T(LinerArray A) {
		LinerArray AT = new LinerArray(LinerArray.initShape(A.getColumnSize(), A.getRowSize()));
		for(int row=0; row<A.getRowSize(); row++) {
			for(int column=0; column<A.getColumnSize(); column++) {
				AT.setElement(column, row, A.getElement(row, column));
			}
		}
		return AT;
	}
	
	public static int[] getOutputShape(LinerArray A, LinerArray B) {
		int[] tmp = new int[A.getRowSize()];
		for(int i=0; i<A.getRowSize(); i++) {
			tmp[i] = B.getColumnSize();
		}
		return tmp;
	}
	
	public static int[] initShape(int row, int column) {
		int[] tmp = new int[row];
		for(int i=0; i<row; i++) {
			tmp[i] = column;
		}
		return tmp;
	}
	
	public static LinerArray copy(LinerArray input) {
		LinerArray tmp = new LinerArray(input.getShape());
			for(int row=0; row<input.getRowSize(); row++) {
				for(int column=0; column<input.getColumnSize(); column++) {
					tmp.setElement(row, column, input.getElement(row, column));		
				}
			}
		return tmp;
	}

	public static LinerArray initParameter(LinerArray parameter, double std) {
		LinerArray tmp = new LinerArray(parameter.getShape());
		Random initRand = new Random();
		initRand.setSeed(0L);
		for(int row=0; row<tmp.getRowSize(); row++) {
			for(int column=0; column<tmp.getColumnSize(); column++) {
				tmp.setElement(row, column, initRand.nextGaussian()*std);
			}
		}
		return tmp;
	}
}

