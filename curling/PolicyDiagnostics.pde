// Terminal diagnostics for AI policy inspection. Call from test mode only.
class PolicyDiagnostics {

  void logTestShot(NeuralPolicy policy, ArrayList<Stone> layout,
                          int stonesLeft, int lastStoneTeam) {
    if (policy == null || layout == null) return;

    float[] state = policy.convertState(layout, stonesLeft, lastStoneTeam);
    float[] hidden  = policy.hiddenLayer.feedForward(state);
    float[] outputs = policy.outputLayer.feedForward(hidden);
    Shot shot = policy.predict(state);

    println("========== AI TEST DIAGNOSTICS ==========");
    logLayout(layout);
    logStateVector(state);
    logActivations("hidden", hidden);
    logActivations("output", outputs);
    logShot(outputs, shot);
    logLayerStats("hidden", policy.hiddenLayer);
    logLayerStats("output", policy.outputLayer);
    logHiddenHealth(hidden, "hidden", policy.HIDDEN_ACTIVATION);
    logOutputHealth(outputs, policy.OUTPUT_ACTIVATION);
    println("=========================================");
  }

  void logLayout(ArrayList<Stone> layout) {
    println("-- layout (" + layout.size() + " stones) --");
    for (int i = 0; i < layout.size(); i++) {
      Stone s = layout.get(i);
      String team = s.team == TEAM_RED ? "RED" : "YELLOW";
      println("  slot " + i + ": " + team
              + "  x=" + nf(s.pos.x, 0, 2)
              + "  y=" + nf(s.pos.y, 0, 2));
    }
  }

  void logStateVector(float[] state) {
    println("-- state[" + state.length + "] --");
    println("  " + arraySummary(state));
    int slots = TOTAL_STONES;
    for (int slot = 0; slot < slots; slot++) {
      int base = slot * 4;
      if (base + 3 >= state.length) break;
      String kind = state[base + 3] < 0 ? "empty" : "stone";
      println("  slot " + slot + " (" + kind + "): "
              + "x=" + nf(state[base], 0, 3)
              + " y=" + nf(state[base + 1], 0, 3)
              + " team=" + nf(state[base + 2], 0, 1)
              + " exists=" + nf(state[base + 3], 0, 1));
    }
    if (state.length >= 2) {
      println("  meta: stonesLeft=" + nf(state[state.length - 2], 0, 2)
              + "  lastTeam=" + nf(state[state.length - 1], 0, 1));
    }
  }

  void logActivations(String name, float[] values) {
    println("-- " + name + " activations[" + values.length + "] --");
    println("  " + arraySummary(values));
    StringBuilder line = new StringBuilder("  ");
    for (int i = 0; i < values.length; i++) {
      line.append(nf(values[i], 0, 3));
      if (i < values.length - 1) line.append(", ");
    }
    println(line.toString());
  }

  void logShot(float[] rawOutputs, Shot shot) {
    println("-- shot --");
    println("  raw curl=" + nf(rawOutputs[0], 0, 4)
            + "  raw speed=" + nf(rawOutputs[1], 0, 4)
            + "  raw angle=" + nf(rawOutputs[2], 0, 4));
    println("  curl=" + nf(shot.curl, 0, 4)
            + "  speed=" + nf(shot.speed, 0, 2)
            + "  angleDeg=" + nf(degrees(shot.angle), 0, 2));
  }

  void logLayerStats(String name, NeuronLayer layer) {
    float wMin = Float.MAX_VALUE, wMax = -Float.MAX_VALUE, wSum = 0;
    float bMin = Float.MAX_VALUE, bMax = -Float.MAX_VALUE, bSum = 0;
    int wCount = 0;
    for (Neuron n : layer.neurons) {
      bMin = min(bMin, n.bias);
      bMax = max(bMax, n.bias);
      bSum += n.bias;
      for (float w : n.weights) {
        wMin = min(wMin, w);
        wMax = max(wMax, w);
        wSum += w;
        wCount++;
      }
    }
    println("-- " + name + " layer weights/biases --");
    println("  neurons=" + layer.neurons.length
            + "  weights=" + wCount
            + "  wMin=" + nf(wMin, 0, 4)
            + "  wMax=" + nf(wMax, 0, 4)
            + "  wMean=" + nf(wSum / wCount, 0, 4));
    println("  bMin=" + nf(bMin, 0, 4)
            + "  bMax=" + nf(bMax, 0, 4)
            + "  bMean=" + nf(bSum / layer.neurons.length, 0, 4));
  }

  // Returns [saturationPct, inactivePct] for a hidden activation vector.
  float[] hiddenHealthPcts(float[] activations, ActivationKind hiddenActivation) {
    int saturated = 0;
    int inactive = 0;
    boolean countInactive = (hiddenActivation == ActivationKind.RELU);
    for (float v : activations) {
      if (abs(v) > 0.9f) saturated++;
      if (countInactive && v < Neuron.RELU_INACTIVE_THRESHOLD) inactive++;
    }
    float n = activations.length;
    return new float[]{
      100.0f * saturated / n,
      100.0f * inactive / n
    };
  }

  void logHiddenHealth(float[] activations, String name, ActivationKind hiddenActivation) {
    float[] pcts = hiddenHealthPcts(activations, hiddenActivation);
    float sq = 0;
    for (float v : activations) sq += v * v;
    float rms = sqrt(sq / activations.length);
    println("-- " + name + " health --");
    println("  sat (|a|>0.9): " + nf(pcts[0], 0, 1) + "%"
            + "  rms=" + nf(rms, 0, 3));
    if (hiddenActivation == ActivationKind.RELU) {
      println("  inactive (a<" + nf(Neuron.RELU_INACTIVE_THRESHOLD, 0, 2) + "): "
              + nf(pcts[1], 0, 1) + "%");
    } else {
      println("  inactive: n/a (hidden is not ReLU)");
    }
  }

  // Fraction of output units with |tanh-out| > 0.9. Returns percent.
  float outputSaturationPct(float[] outputs, ActivationKind outputActivation) {
    if (outputActivation != ActivationKind.TANH) return 0;
    int saturated = 0;
    for (float v : outputs) {
      if (abs(v) > 0.9f) saturated++;
    }
    return 100.0f * saturated / outputs.length;
  }

  void logOutputHealth(float[] outputs, ActivationKind outputActivation) {
    println("-- output health --");
    if (outputActivation == ActivationKind.TANH) {
      println("  sat (|out|>0.9): " + nf(outputSaturationPct(outputs, outputActivation), 0, 1) + "%");
    } else {
      println("  sat: n/a (output is not tanh)");
    }
  }

  String arraySummary(float[] values) {
    float vMin = Float.MAX_VALUE, vMax = -Float.MAX_VALUE, sum = 0;
    for (float v : values) {
      vMin = min(vMin, v);
      vMax = max(vMax, v);
      sum += v;
    }
    return "min=" + nf(vMin, 0, 3)
         + " max=" + nf(vMax, 0, 3)
         + " mean=" + nf(sum / values.length, 0, 3);
  }
}
