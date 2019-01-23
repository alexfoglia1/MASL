import java.util.Arrays;

public class NeuralNetwork {
	private int n_in;
	private int n_hid;
	private int n_out;
	private NeuronLayer hidden;
	private NeuronLayer output;

	public NeuralNetwork(int inputNeurons, int hiddenNeurons, int numberOfOutputs) {
		n_in = inputNeurons;
		n_hid = hiddenNeurons;
		n_out = numberOfOutputs;
		hidden = new NeuronLayer(n_hid, n_in);
		output = new NeuronLayer(n_out, n_hid);
		initInputToHidden();
		initHiddenToOut();
	}

	public void initInputToHidden() {
		for (int i = 0; i < n_hid; i++) {
			for (int j = 0; j < n_in; j++) {
				hidden.neurons[i].weights[j] = Math.random();
			}
		}
	}

	public void initHiddenToOut() {
		for (int i = 0; i < n_out; i++) {
			for (int j = 0; j < n_hid; j++) {
				output.neurons[i].weights[j] = Math.random();
			}
		}
	}

	public double[] feedForward(double[] inputs) {
		double[] hidden_outputs = hidden.feedForward(inputs);
		return output.feedForward(hidden_outputs);
	}

	public void train(double[] training_in, double[] training_out) {
		feedForward(training_in);
		double[] deltaWrtOut = new double[n_out];
		for (int i = 0; i < n_out; i++) {
			double target_output = training_out[i];
			double actual_output = output.neurons[i].output;
			double deltaWrtInput = -(target_output - actual_output) * actual_output * (1 - actual_output);
			deltaWrtOut[i] = deltaWrtInput;
		}
		double[] deltaWrtHid = new double[n_hid];
		for(int i=0; i<n_hid;i++) {
			double deltaWrtHiddenOut = 0;
			for(int j=0; j<n_out;j++) {
				deltaWrtHiddenOut+=deltaWrtOut[j] * output.neurons[j].weights[i];
			}
			double actual_output = hidden.neurons[i].output;
			double deltaWrtIn = actual_output * (1 - actual_output);
			deltaWrtHid[i] = deltaWrtHiddenOut * deltaWrtIn;
		}
		for (int i = 0; i < n_out; i++) {
			for (int j = 0; j < n_hid; j++) {
				double act_input = output.neurons[i].inputs[j];
				double deltaWrtWeight = deltaWrtOut[i] * act_input;
				output.neurons[i].weights[j] -= deltaWrtWeight;
			}
		}
		for (int i = 0; i < n_hid; i++) {
			for (int j = 0; j < n_in; j++) {
				double act_input = hidden.neurons[i].inputs[j];
				double deltaWrtWeight = deltaWrtHid[i] * act_input;
				hidden.neurons[i].weights[j] -= deltaWrtWeight;
			}
		}
	}

	public double totalError(double[][][] training_sets) {
		double err = 0;
		for (int i = 0; i < training_sets.length; i++) {
			double[] t_in = training_sets[i][0];
			double[] t_out = training_sets[i][1];
			double[] act_out = feedForward(t_in);
			for (int j = 0; j < act_out.length; j++) {
				double target_output = t_out[j];
				double actual_output = output.neurons[j].output;
				double squareError = 0.5 * Math.pow(target_output - actual_output, 2);
				err += squareError;
			}
		}
		return err;
	}
	
	public static void main(String[] args) {
    	double[][][] training_sets = {
                {{0, 0}, {0}},
                {{0, 1}, {1}},
                {{1, 0}, {1}},
                {{1, 1}, {0}}
				};

		NeuralNetwork nn = new NeuralNetwork(2, 5, 1);
		double[] prediction = nn.feedForward(new double[] {0,0});
		System.out.println("Before Training:\n\n0 xor 0 = " + Arrays.toString(prediction));
		prediction = nn.feedForward(new double[] {0,1});
		System.out.println("0 xor 1 = " + Arrays.toString(prediction));
		prediction = nn.feedForward(new double[] {1,0});
		System.out.println("1 xor 0 = " + Arrays.toString(prediction));
		prediction = nn.feedForward(new double[] {1,1});
		System.out.println("1 xor 1 = " + Arrays.toString(prediction)+"\n\nAfter Training:\n");
		for (int i = 0; i < 20000; i++) {
			int randIndex = (int) (Math.random() * training_sets.length);
			double[] t_in = training_sets[randIndex][0];
			double[] t_out = training_sets[randIndex][1];
			nn.train(t_in, t_out);
			
		}
		prediction = nn.feedForward(new double[] {0,0});
		System.out.println("0 xor 0 = " + Arrays.toString(prediction));
		prediction = nn.feedForward(new double[] {0,1});
		System.out.println("0 xor 1 = " + Arrays.toString(prediction));
		prediction = nn.feedForward(new double[] {1,0});
		System.out.println("1 xor 0 = " + Arrays.toString(prediction));
		prediction = nn.feedForward(new double[] {1,1});
		System.out.println("1 xor 1 = " + Arrays.toString(prediction));
	}
	
}
