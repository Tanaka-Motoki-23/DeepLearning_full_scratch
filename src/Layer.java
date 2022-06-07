
public interface Layer {
	LinerArray forward(LinerArray x);
	void backward(LinerArray dOut);
}
