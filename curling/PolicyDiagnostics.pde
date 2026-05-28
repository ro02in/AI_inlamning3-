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

  // One expert line for PG inference (matches policy-search ensemble format).
  void logGradientExpertLine(String name, NeuralPolicy policy,
                             float[] hidden, float[] means, float[] logStds,
                             Shot shot, float score,
                             float satPct, float deadPct, float outSatPct) {
    String inactiveStr = (policy.HIDDEN_ACTIVATION == ActivationKind.RELU)
        ? "  inactive=" + nf(deadPct, 0, 1) + "%"
        : "";
    String outSatStr = (policy.OUTPUT_ACTIVATION == ActivationKind.TANH)
        ? "  outSat=" + nf(outSatPct, 0, 1) + "%"
        : "";
    String sigmaStr = "";
    if (policy.meanAndStd && logStds != null) {
      float ls0 = constrain(logStds[0], NeuralPolicy.LOG_STD_MIN, NeuralPolicy.LOG_STD_MAX);
      float ls1 = constrain(logStds[1], NeuralPolicy.LOG_STD_MIN, NeuralPolicy.LOG_STD_MAX);
      float ls2 = constrain(logStds[2], NeuralPolicy.LOG_STD_MIN, NeuralPolicy.LOG_STD_MAX);
      sigmaStr = "  σ=" + nf(exp(ls0), 0, 3) + "/" + nf(exp(ls1), 0, 3)
               + "/" + nf(exp(ls2), 0, 3);
    }
    println("  " + name
            + "  curl=" + nf(shot.curl, 0, 3)
            + "  spd=" + nf(shot.speed, 0, 1)
            + "  ang=" + nf(degrees(shot.angle), 0, 1) + "deg"
            + "  hSat=" + nf(satPct, 0, 1) + "%"
            + inactiveStr
            + outSatStr
            + sigmaStr
            + "  score=" + nf(score, 0, 4));
  }

  // Inference diagnostics for a single meanAndStd policy (test / play with logScores=true).
  void logGradientInference(NeuralPolicy policy, ArrayList<Stone> layout,
                            int stonesLeft, int lastStoneTeam, Heuristic scorer) {
    if (policy == null || layout == null || !policy.meanAndStd) return;

    float[] state   = policy.convertState(layout, stonesLeft, lastStoneTeam);
    float[] hidden  = policy.hiddenLayer.feedForward(state);
    float[] means   = policy.outputLayer.feedForward(hidden);
    float[] logStds = policy.outputLogStd.feedForward(hidden);
    Shot shot = policy.predictMean(state);

    float score = 0;
    if (scorer != null) {
      ShotResult result = scorer.simulate(policy, state, layout, null);
      score = scorer.scoreResult(result);
    }

    float[] health = hiddenHealthPcts(hidden, policy.HIDDEN_ACTIVATION);
    float outSatPct = outputSaturationPct(means, policy.OUTPUT_ACTIVATION);

    println("--- PG expert scores ---");
    logGradientExpertLine("Enkel", policy, hidden, means, logStds, shot, score,
                          health[0], health[1], outSatPct);
    println("Chosen shot: Enkel"
            + "  score=" + nf(score, 0, 3)
            + "  curl=" + nf(shot.curl, 0, 3)
            + "  spd=" + nf(shot.speed, 0, 1)
            + "  ang=" + nf(degrees(shot.angle), 0, 1) + "deg");
  }

  // Periodic training diagnostics (printed every N update steps).
  void logGradientTrainingStep(String expertLabel, NeuralPolicy policy, float[] hidden,
                               int step, float meanReward, float maxReward, float minReward,
                               int eliteCount, float advSum, boolean skipped,
                               float[] means, float[] sigmas, float[] gradMean, float[] gradLogStd) {
    String prefix = expertLabel.length() > 0 ? ("[" + expertLabel + "] ") : "";
    float satPct = 0, deadPct = 0, outSatPct = 0;
    if (policy != null && hidden != null) {
      float[] health = hiddenHealthPcts(hidden, policy.HIDDEN_ACTIVATION);
      satPct = health[0];
      deadPct = health[1];
      if (means != null) {
        outSatPct = outputSaturationPct(means, policy.OUTPUT_ACTIVATION);
      }
    }
    String healthStr = "  hSat=" + nf(satPct, 0, 1) + "%"
                     + "  inactive=" + nf(deadPct, 0, 1) + "%"
                     + "  outSat=" + nf(outSatPct, 0, 1) + "%";

    println("--- PG train step " + step + " " + prefix + "---");
    if (skipped) {
      println("  skipped (all elites below baseline)" + healthStr);
      println("  reward mean=" + nf(meanReward, 0, 3)
              + "  max=" + nf(maxReward, 0, 3)
              + "  min=" + nf(minReward, 0, 3));
      return;
    }
    println("  reward mean=" + nf(meanReward, 0, 3)
            + "  max=" + nf(maxReward, 0, 3)
            + "  min=" + nf(minReward, 0, 3)
            + "  elites=" + eliteCount
            + "  advSum=" + nf(advSum, 0, 3)
            + healthStr);
    println("  μ tanh: curl=" + nf(means[0], 0, 3)
            + "  speed=" + nf(means[1], 0, 3)
            + "  angle=" + nf(means[2], 0, 3));
    println("  σ: curl=" + nf(sigmas[0], 0, 4)
            + "  speed=" + nf(sigmas[1], 0, 4)
            + "  angle=" + nf(sigmas[2], 0, 4));
    println("  grad μ: curl=" + nf(gradMean[0], 0, 4)
            + "  speed=" + nf(gradMean[1], 0, 4)
            + "  angle=" + nf(gradMean[2], 0, 4));
    println("  grad logσ: curl=" + nf(gradLogStd[0], 0, 4)
            + "  speed=" + nf(gradLogStd[1], 0, 4)
            + "  angle=" + nf(gradLogStd[2], 0, 4));
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
