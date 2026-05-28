// Verifies backprop by comparing analytic weight gradients to finite-difference estimates.
// Run from the sketch with the 'b' key; results print to the Processing console.
class ReinforceSmokeTest {
    static final float EPS = 1e-4f;
    static final float REL_TOL = 0.05f;

    boolean run() {
        randomSeed(42);
        NeuralPolicy policy = new NeuralPolicy();
        float[] state = fixedState();

        println("========== BACKPROP SMOKE TEST ==========");
        println("Press 'b' in the sketch to re-run this test.");
        println("Checks that analytic gradients match finite-difference estimates.");
        println("");

        policy.forward(state);
        float curl0 = policy.predictFromCache().curl;
        println("Baseline curl (forward): " + nf(curl0, 0, 6));

        policy.zeroGrads();
        policy.backwardFromShotGrad(1, 0, 0);

        int passed = 0;
        int total = 0;

        total++;
        if (checkWeightGrad(policy, state, "output curl neuron, weight[0]",
                              policy.outputLayer.neurons[0], 0)) passed++;

        total++;
        if (checkWeightGrad(policy, state, "output curl neuron, bias",
                              policy.outputLayer.neurons[0], -1)) passed++;

        total++;
        if (checkWeightGrad(policy, state, "hidden neuron[0], weight[0]",
                              policy.hiddenLayer.neurons[0], 0)) passed++;

        total++;
        if (checkDirection(policy, state, policy.outputLayer.neurons[0], 0)) passed++;

        println("");
        if (passed == total) {
            println("RESULT: PASS (" + passed + "/" + total + " checks)");
            println("Backprop looks correct — safe to build PolicyGradientTraining.");
        } else {
            println("RESULT: FAIL (" + passed + "/" + total + " checks)");
            println("Fix Neuron/NeuronLayer/NeuralPolicy backward before REINFORCE training.");
        }
        println("=========================================");
        return passed == total;
    }

    float[] fixedState() {
        float[] state = new float[TOTAL_STONES * 4 + 2];
        for (int i = 0; i < state.length; i++) {
            state[i] = -0.5f + (i * 0.13f) % 1.0f;
        }
        return state;
    }

    boolean checkWeightGrad(NeuralPolicy policy, float[] state,
                            String label, Neuron neuron, int weightIndex) {
        float analytic = (weightIndex >= 0)
            ? neuron.weightGrads[weightIndex]
            : neuron.biasGrad;
        float numerical = numericalCurlGrad(policy, state, neuron, weightIndex);
        return reportGradCheck(label, analytic, numerical);
    }

    float numericalCurlGrad(NeuralPolicy policy, float[] state,
                              Neuron neuron, int weightIndex) {
        float plus, minus;
        if (weightIndex >= 0) {
            float saved = neuron.weights[weightIndex];
            neuron.weights[weightIndex] = saved + EPS;
            policy.forward(state);
            plus = policy.predictFromCache().curl;

            neuron.weights[weightIndex] = saved - EPS;
            policy.forward(state);
            minus = policy.predictFromCache().curl;

            neuron.weights[weightIndex] = saved;
        } else {
            float saved = neuron.bias;
            neuron.bias = saved + EPS;
            policy.forward(state);
            plus = policy.predictFromCache().curl;

            neuron.bias = saved - EPS;
            policy.forward(state);
            minus = policy.predictFromCache().curl;

            neuron.bias = saved;
        }
        return (plus - minus) / (2 * EPS);
    }

    boolean checkDirection(NeuralPolicy policy, float[] state,
                           Neuron neuron, int weightIndex) {
        float grad = neuron.weightGrads[weightIndex];
        float saved = neuron.weights[weightIndex];

        policy.forward(state);
        float curlBefore = policy.predictFromCache().curl;

        neuron.weights[weightIndex] = saved + EPS;
        policy.forward(state);
        float curlAfter = policy.predictFromCache().curl;

        neuron.weights[weightIndex] = saved;

        float delta = curlAfter - curlBefore;
        boolean pass;
        if (abs(grad) < 1e-6f) {
            pass = abs(delta) < 1e-3f;
        } else {
            pass = (grad > 0 && delta > 0) || (grad < 0 && delta < 0);
        }

        println("-- direction: output curl weight[0] --");
        println("  analytic grad=" + nf(grad, 0, 6)
                + "  curl delta (w+eps)=" + nf(delta, 0, 6)
                + "  " + (pass ? "PASS" : "FAIL"));
        return pass;
    }

    boolean reportGradCheck(String label, float analytic, float numerical) {
        float err = abs(analytic - numerical);
        float scale = max(max(abs(analytic), abs(numerical)), 1e-6f);
        float relErr = err / scale;
        boolean pass = relErr <= REL_TOL;

        println("-- " + label + " --");
        println("  analytic  d(curl)/dparam = " + nf(analytic, 0, 6));
        println("  numerical d(curl)/dparam = " + nf(numerical, 0, 6));
        println("  relative error = " + nf(relErr * 100, 0, 2) + "%  "
                + (pass ? "PASS" : "FAIL"));
        return pass;
    }
}
