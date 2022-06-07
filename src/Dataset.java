import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class Dataset {
	private ArrayList<LinerArray> img = new ArrayList<LinerArray>();
	private ArrayList<LinerArray> label = new ArrayList<LinerArray>();
	protected ArrayList<Data> data = new ArrayList<Data>();
	protected int classes, numData;
	
	public Dataset(String imagePath, String labelPath, int classes, int numData) {
		this.classes = classes;
		this.numData = numData;
		this.setImage(imagePath, this.img, LinerArray.initShape(1, 784), numData, true);
		this.setLabel(labelPath, this.label, LinerArray.initShape(1, classes), numData);
		for(int index=0; index<numData; index++) {
			Data tmp = new Data(this.img.get(index), this.label.get(index));
			this.data.add(tmp);
		}
	}
	
	public void setImage(String filePath, ArrayList<LinerArray> output, int[] shape, int numData, boolean normalize) {
		File f;
		BufferedReader br;
		LinerArray tmp;
		
		f = new File(filePath);
		try {
			br = new BufferedReader(new FileReader(f));
			String line;
			for (int nd=0; nd<numData; nd++) {
				tmp = new LinerArray(shape);
				for(int row=0; row<shape.length; row++) {
					line = br.readLine();
					String[] img = line.split(",");
					for (int column=0; column<img.length; column++){
						if(normalize) {
							tmp.setElement(row, column, Double.parseDouble(img[column]) / 255.0d);
						}else {
							tmp.setElement(row, column, Double.parseDouble(img[column]));
						}
			        }
				}
				if(nd % 100 == 0) {
					System.out.println("Loading... " + filePath + " => "+ nd + "/" + numData);
				}			
				output.add(tmp);
			}
			System.out.println("Finish.");
			br.close();
				
			} catch (IOException e) {
				e.printStackTrace();
		};
	}
	
	public void setLabel(String filePath, ArrayList<LinerArray> output, int[] shape, int numData) {
		File f;
		BufferedReader br;
		LinerArray tmp;
		
		f = new File(filePath);
		try {
			br = new BufferedReader(new FileReader(f));
			String line;
			for (int nd=0; nd<numData; nd++) {
				tmp = new LinerArray(shape);
				for(int row=0; row<shape.length; row++) {
					line = br.readLine();
					String[] img = line.split(",");
					for (int column=0; column<shape[0]; column++){
						if(column == Integer.parseInt(img[0])) {
							tmp.setElement(row, column, 1.0d);
						}else {
							tmp.setElement(row, column, 0.0d);
						}
					}
			     }
				if(nd % 100 == 0) {
					System.out.println("Loading... " + filePath + " => "+ nd + "/" + numData);
				}			
				output.add(tmp);
			}
			System.out.println("Finish.");
			br.close();
				
			} catch (IOException e) {
				e.printStackTrace();
		};
	}
	
	public void shuffle() {
		Collections.shuffle(this.data);
	}
	
	public int getNumData() {

		return this.numData;
	}
	
	public Data getData(int index) {
		return this.data.get(index);
	}
	
	public ArrayList<Data> getAllData() {
		return this.data;
	}
	
	public Data getAllLabel(int index) {
		return this.data.get(index);
	}
	
	public int getImgShape() {
		return this.data.get(0).getImage().getColumnSize();
	}
	
	public void printImg() {
		for(Data data : this.data) {
			data.getImage().printArray();
		}
	}
	
	public void printLabel() {
		for(Data data : this.data) {
			data.getLabel().printArray();
		}
	}
	
	public int getClasses() {
		return this.classes;
	}
	
	public static void printLabel(ArrayList<LinerArray> tBatch) {
		for(LinerArray label : tBatch) {
			label.printArray();
		}
		System.out.println("\t" + "[ " + "Label_Num = " + tBatch.size() + " ]");
	} 
	
}
