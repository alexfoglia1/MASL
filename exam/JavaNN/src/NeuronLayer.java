
public class NeuronLayer {
	
	public Neuron[] neurons;

	public NeuronLayer(int n, int inputsPerNeuron) {
		this.neurons = new Neuron[n];
		for (int i = 0; i < n; i++) {
			this.neurons[i] = (new Neuron(inputsPerNeuron));
		}
	}

	public double[] feedForward(double[] inputs) {
		double[] outputs = new double[neurons.length];
		for (int i = 0; i < neurons.length; i++) {
			outputs[i] = (neurons[i].predict(inputs));
		}
		return outputs;
	}

	public double[] getOutputs() {
		double[] outputs = new double[neurons.length];
		for (int i = 0; i < neurons.length; i++) {
			outputs[i] = neurons[i].output;
		}
		return outputs;
	}

	
}
