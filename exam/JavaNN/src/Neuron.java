
public class Neuron{
	public double[] weights;
	public double[] inputs;
	public double output;
    public Neuron(int n_inputs) {
        weights = new double[n_inputs];
    }
    private double logistic(double x) {
    	return 1/(1+Math.exp(-x));
    }
    public double predict(double[] inputs) {
        this.inputs = inputs;
        double sum = 0;
        for(int i=0; i<inputs.length;i++) {
        	sum += inputs[i] * weights[i];
        }
        this.output = logistic(sum);
        return output;
    }
}