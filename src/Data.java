
public class Data {
	protected LinerArray image;
	protected LinerArray label;
	
	public Data(LinerArray image, LinerArray label) {
		this.image = image;
		this.label = label;
	}
	
	public LinerArray getImage() {
		return this.image;
	}
	
	public LinerArray getLabel() {
		return this.label;
	}
	
}
