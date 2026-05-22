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
    logSaturation(hidden, "hidden");
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

  void logSaturation(float[] activations, String name) {
    int saturated = 0;
    for (float v : activations) {
      if (abs(v) > 0.9) saturated++;
    }
    float pct = 100.0f * saturated / activations.length;
    println("-- " + name + " saturation (|a|>0.9): "
            + saturated + "/" + activations.length
            + " (" + nf(pct, 0, 1) + "%)");
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
