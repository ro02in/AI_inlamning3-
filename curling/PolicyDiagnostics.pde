// Terminal diagnostics for AI policy inspection. Call from test mode only.
class PolicyDiagnostics {
  final float RELU_HIGH_ACTIVATION = 3.0f;

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

  // Full inference log for gradient ensemble + shot-type selector.
  void logEnsembleInference(GradientEnsemble ensemble, float[] probs, int chosen,
                            ArrayList<Stone> layout,
                            int stonesLeft, int lastTeam, Shot shot,
                            String modeLabel, String selectorMode) {
    if (ensemble == null || ensemble.trainers == null) return;
    NeuralPolicy policy = ensemble.trainers[chosen].policy;
    ArrayList<Heuristic> heuristics = ensemble.typeHeuristics[chosen];
    if (policy == null || layout == null) return;

    float[] state  = policy.convertState(layout, stonesLeft, lastTeam);
    float[] hidden = policy.hiddenLayer.feedForward(state);
    float[] means  = policy.outputLayer.feedForward(hidden);
    float[] logStds = policy.meanAndStd ? policy.outputLogStd.feedForward(hidden) : null;

    float[] health = hiddenHealthPcts(hidden, policy.HIDDEN_ACTIVATION);
    float satPct   = health[0];
    float deadPct  = health[1];
    float outSatPct = outputSaturationPct(means, policy.OUTPUT_ACTIVATION);
    float[] scores = ensembleScoreBreakdown(heuristics, shot, layout, stonesLeft);
    String scoreFields = "immediateScore=" + nf(scores[0], 0, 4)
                       + "  typeHeuristic=" + nf(scores[1], 0, 4)
                       + "  rolloutScore=" + nf(scores[2], 0, 4);

    println("========== AI SHOT (" + modeLabel + ") ==========");
    println("selectorMode=" + selectorMode);
    println("-- selector probs --");
    for (int i = 0; i < ensemble.names.length; i++) {
      println("  " + ensemble.names[i] + "  p=" + nf(probs[i], 0, 3)
              + (i == chosen ? "  *CHOSEN*" : ""));
    }
    logRolloutDecision(ensemble, chosen);
    logActionDiversity(ensemble, layout, stonesLeft, lastTeam, chosen, shot);
    println("-- chosen expert: " + ensemble.names[chosen] + " --");
    logGradientExpertLineWithScores(ensemble.names[chosen], policy, hidden, means, logStds,
                                    shot, scoreFields, satPct, deadPct, outSatPct);
    println("=========================================");
  }

  void logRolloutDecision(GradientEnsemble ensemble, int chosen) {
    if (ensemble.lastDecisionScores == null || ensemble.lastAdjustedSelectorProbs == null) return;
    println("-- rollout decision --");
    for (int i = 0; i < ensemble.count; i++) {
      println("  " + ensemble.names[i]
              + "  nnOdds=" + nf(ensemble.lastAdjustedSelectorProbs[i], 0, 3)
              + "  rolloutOdds=" + nf(ensemble.lastRolloutRankProbs[i], 0, 3)
              + "  rollout=" + nf(ensemble.lastRolloutScores[i], 0, 3)
              + "  type=" + nf(ensemble.lastTypeScores[i], 0, 3)
              + "  finalProb=" + nf(ensemble.lastFinalDecisionProbs[i], 0, 3)
              + (i == chosen ? "  *BEST*" : ""));
    }
  }

  // One expert line for PG inference (matches policy-search ensemble format).
  void logGradientExpertLine(String name, NeuralPolicy policy,
                             float[] hidden, float[] means, float[] logStds,
                             Shot shot, float score,
                             float satPct, float deadPct, float outSatPct) {
    logGradientExpertLineWithScores(name, policy, hidden, means, logStds, shot,
                                    "score=" + nf(score, 0, 4),
                                    satPct, deadPct, outSatPct);
  }

  void logGradientExpertLineWithScores(String name, NeuralPolicy policy,
                             float[] hidden, float[] means, float[] logStds,
                             Shot shot, String scoreFields,
                             float satPct, float deadPct, float outSatPct) {
    String outSatStr = (policy.OUTPUT_ACTIVATION == ActivationKind.TANH)
        ? "  outSat=" + nf(outSatPct, 0, 1) + "%"
        : "";
    String sigmaStr = "";
    if (policy.meanAndStd && logStds != null) {
      float s0 = policy.sigmaFromLogStd(0, logStds[0]);
      float s1 = policy.sigmaFromLogStd(1, logStds[1]);
      float s2 = policy.sigmaFromLogStd(2, logStds[2]);
      sigmaStr = "  σ=" + nf(s0, 0, 3) + "/" + nf(s1, 0, 3) + "/" + nf(s2, 0, 3);
    }
    println("  " + name
            + "  curl=" + nf(shot.curl, 0, 3)
            + "  spd=" + nf(shot.speed, 0, 1)
            + "  ang=" + nf(degrees(shot.angle), 0, 1) + "deg"
            + hiddenHealthFields(policy, hidden)
            + outSatStr
            + sigmaStr
            + "  " + scoreFields);
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
                               float[] means, float[] sigmas, float[] gradMean, float[] gradLogStd,
                               float stdAtMaxPct, float stdAtMinPct,
                               float rawLogStdMin, float rawLogStdMax,
                               float minClampGap, float maxClampGap,
                               float rawClipPct, float meanOutSatPct,
                               float rewardSpread, float entropyGrad, float stdRegGrad) {
    String prefix = expertLabel.length() > 0 ? ("[" + expertLabel + "] ") : "";
    float outSatPct = 0;
    if (policy != null && hidden != null) {
      if (means != null) {
        outSatPct = outputSaturationPct(means, policy.OUTPUT_ACTIVATION);
      }
    }
    String healthStr = hiddenHealthFields(policy, hidden)
                     + "  outSat=" + nf(outSatPct, 0, 1) + "%";

    println("--- PG train step " + step + " " + prefix + "---");
    if (skipped) {
      println("  skipped (all elites below baseline)" + healthStr);
      println("  reward mean=" + nf(meanReward, 0, 3)
              + "  max=" + nf(maxReward, 0, 3)
              + "  min=" + nf(minReward, 0, 3)
              + "  rewardSpread=" + nf(rewardSpread, 0, 3));
      println("  stdAtMax=" + nf(stdAtMaxPct, 0, 1) + "%"
              + "  stdAtMin=" + nf(stdAtMinPct, 0, 1) + "%"
              + "  rawLogStd=" + nf(rawLogStdMin, 0, 3) + ".." + nf(rawLogStdMax, 0, 3)
              + "  clampGap=" + nf(minClampGap, 0, 3) + "/" + nf(maxClampGap, 0, 3)
              + "  rawClip=" + nf(rawClipPct, 0, 1) + "%"
              + "  meanOutSat=" + nf(meanOutSatPct, 0, 1) + "%");
      return;
    }
    println("  reward mean=" + nf(meanReward, 0, 3)
            + "  max=" + nf(maxReward, 0, 3)
            + "  min=" + nf(minReward, 0, 3)
            + "  rewardSpread=" + nf(rewardSpread, 0, 3)
            + "  elites=" + eliteCount
            + "  advSum=" + nf(advSum, 0, 3)
            + healthStr);
    println("  stdAtMax=" + nf(stdAtMaxPct, 0, 1) + "%"
            + "  stdAtMin=" + nf(stdAtMinPct, 0, 1) + "%"
            + "  rawLogStd=" + nf(rawLogStdMin, 0, 3) + ".." + nf(rawLogStdMax, 0, 3)
            + "  clampGap=" + nf(minClampGap, 0, 3) + "/" + nf(maxClampGap, 0, 3)
            + "  rawClip=" + nf(rawClipPct, 0, 1) + "%"
            + "  meanOutSat=" + nf(meanOutSatPct, 0, 1) + "%"
            + "  entropyGrad=" + nf(entropyGrad, 0, 4)
            + "  stdRegGrad=" + nf(stdRegGrad, 0, 4));
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

  void logActionDiversity(GradientEnsemble ensemble, ArrayList<Stone> layout,
                          int stonesLeft, int lastTeam, int chosen, Shot chosenShot) {
    Shot[] means = new Shot[ensemble.count];
    println("-- expert means --");
    for (int i = 0; i < ensemble.count; i++) {
      means[i] = ensemble.expertMeanShot(i, layout, stonesLeft, lastTeam);
      println("  " + ensemble.names[i]
              + "  curl=" + nf(means[i].curl, 0, 3)
              + "  spd=" + nf(means[i].speed, 0, 1)
              + "  ang=" + nf(degrees(means[i].angle), 0, 1) + "deg"
              + (i == chosen ? "  *CHOSEN*" : ""));
    }

    float distSum = 0;
    float distMin = Float.POSITIVE_INFINITY;
    int pairs = 0;
    for (int i = 0; i < means.length; i++) {
      for (int j = i + 1; j < means.length; j++) {
        float d = actionDistance(means[i], means[j]);
        distSum += d;
        distMin = min(distMin, d);
        pairs++;
      }
    }
    if (pairs == 0) distMin = 0;

    float speedMean = 0;
    float angleMean = 0;
    for (Shot s : means) {
      speedMean += s.speed;
      angleMean += degrees(s.angle);
    }
    speedMean /= max(1, means.length);
    angleMean /= max(1, means.length);
    float speedVar = 0;
    float angleVar = 0;
    for (Shot s : means) {
      float ds = s.speed - speedMean;
      float da = degrees(s.angle) - angleMean;
      speedVar += ds * ds;
      angleVar += da * da;
    }

    int[] expertBands = speedBands(means);
    int[] selectedBands = new int[5];
    selectedBands[speedBandIndex(chosenShot.speed)]++;
    println("-- action diversity --"
            + " meanPairwiseActionDistance=" + nf(distSum / max(1, pairs), 0, 3)
            + " minPairwiseActionDistance=" + nf(distMin, 0, 3)
            + " speedStdAcrossExperts=" + nf(sqrt(speedVar / max(1, means.length)), 0, 3)
            + " angleStdAcrossExperts=" + nf(sqrt(angleVar / max(1, means.length)), 0, 3));
    println("  expertSpeedBands " + speedBandSummary(expertBands));
    println("  selectedSpeedBand " + speedBandSummary(selectedBands));
  }

  float actionDistance(Shot a, Shot b) {
    float curl = (a.curl - b.curl) / 2.0f;
    float speed = (a.speed - b.speed) / max(1f, UI.SPEED_MAX);
    float angle = degrees(a.angle - b.angle) / 20.0f;
    return sqrt(curl * curl + speed * speed + angle * angle);
  }

  int[] speedBands(Shot[] shots) {
    int[] counts = new int[5];
    for (Shot s : shots) counts[speedBandIndex(s.speed)]++;
    return counts;
  }

  int speedBandIndex(float speed) {
    if (speed < 18f) return 0;
    if (speed < 22f) return 1;
    if (speed < 25f) return 2;
    if (speed < 28f) return 3;
    return 4;
  }

  String speedBandSummary(int[] counts) {
    return "12-18=" + counts[0]
         + " 18-22=" + counts[1]
         + " 22-25=" + counts[2]
         + " 25-28=" + counts[3]
         + " 28+=" + counts[4];
  }

  float[] ensembleScoreBreakdown(ArrayList<Heuristic> heuristics, Shot shot,
                                 ArrayList<Stone> layout, int stonesLeft) {
    ShotResult result = simulateMeanShot(shot, layout,
                                         max(0, (stonesLeft - 1) * 2));
    ScoreResult after = house.scoreEnd(result.stonesAfter);
    float immediateScore = after.diffFrom(result.scoreBefore);
    float typeHeuristic = 0;
    float rolloutScore = 0;

    if (heuristics != null) {
      for (Heuristic h : heuristics) {
        float contributed = h.contribute(h.scoreResult(result));
        if (h instanceof FinalScoreHeuristic) rolloutScore += contributed;
        else typeHeuristic += contributed;
      }
    }
    return new float[]{ immediateScore, typeHeuristic, rolloutScore };
  }

  ShotResult simulateMeanShot(Shot shot, ArrayList<Stone> layout, int shotsRemainingAfter) {
    ArrayList<Stone> before    = copyLayout(layout);
    ArrayList<Stone> simStones = copyLayout(layout);
    ScoreResult scoreBefore = house.scoreEnd(simStones);

    PVector h = sheet.hackWorld();
    Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
    fired.curl = constrain(shot.curl, -1, 1);
    fired.vel.set(sin(shot.angle) * shot.speed, cos(shot.angle) * shot.speed);
    simStones.add(fired);

    for (int step = 0; step < 100000; step++) {
      physics.step(simStones, DT);
      boolean anyMoving = false;
      for (Stone s : simStones) if (s.isMoving()) { anyMoving = true; break; }
      if (!anyMoving) break;
    }

    ShotResult result = new ShotResult(simStones, before, fired, scoreBefore, shot);
    result.shotsRemainingAfter = shotsRemainingAfter;
    return result;
  }

  ArrayList<Stone> copyLayout(ArrayList<Stone> layout) {
    ArrayList<Stone> copy = new ArrayList<Stone>();
    for (Stone s : layout) {
      Stone c = new Stone(s.pos.x, s.pos.y, s.team);
      c.hogPassed = s.hogPassed;
      copy.add(c);
    }
    return copy;
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
      if (hiddenActivation == ActivationKind.RELU) {
        if (v > RELU_HIGH_ACTIVATION) saturated++;
      } else if (abs(v) > 0.9f) {
        saturated++;
      }
      if (countInactive && v < Neuron.RELU_INACTIVE_THRESHOLD) inactive++;
    }
    float n = activations.length;
    return new float[]{
      100.0f * saturated / n,
      100.0f * inactive / n
    };
  }

  String hiddenHealthLabel(NeuralPolicy policy) {
    return policy != null && policy.HIDDEN_ACTIVATION == ActivationKind.RELU ? "hHigh" : "hSat";
  }

  String hiddenHealthFields(NeuralPolicy policy, float[] hidden) {
    if (policy == null) {
      return "  hRms=0.000  zRms=0.000  zMax=0.000  normRms=0.000  inactive=0.0%";
    }
    NeuronLayer layer = policy.hiddenLayer;
    float hRms = hiddenRms(hidden);
    float zRms = hiddenRms(layer != null ? layer.lastRawPreActivation : null);
    float zMax = maxAbs(layer != null ? layer.lastRawPreActivation : null);
    float normRms = hiddenRms(layer != null ? layer.lastPreActivation : null);
    float inactive = hiddenInactivePct(hidden, policy.HIDDEN_ACTIVATION);
    return "  hRms=" + nf(hRms, 0, 3)
         + "  zRms=" + nf(zRms, 0, 3)
         + "  zMax=" + nf(zMax, 0, 3)
         + "  normRms=" + nf(normRms, 0, 3)
         + "  inactive=" + nf(inactive, 0, 1) + "%";
  }

  float hiddenRms(float[] activations) {
    if (activations == null || activations.length == 0) return 0;
    float sq = 0;
    for (float v : activations) sq += v * v;
    return sqrt(sq / activations.length);
  }

  float maxAbs(float[] values) {
    if (values == null || values.length == 0) return 0;
    float result = 0;
    for (float v : values) result = max(result, abs(v));
    return result;
  }

  float hiddenInactivePct(float[] activations, ActivationKind hiddenActivation) {
    if (activations == null || activations.length == 0
        || hiddenActivation != ActivationKind.RELU) return 0;
    int inactive = 0;
    for (float v : activations) {
      if (v < Neuron.RELU_INACTIVE_THRESHOLD) inactive++;
    }
    return 100.0f * inactive / activations.length;
  }

  void logHiddenHealth(float[] activations, String name, ActivationKind hiddenActivation) {
    float[] pcts = hiddenHealthPcts(activations, hiddenActivation);
    float rms = hiddenRms(activations);
    println("-- " + name + " health --");
    if (hiddenActivation == ActivationKind.RELU) {
      println("  high (a>" + nf(RELU_HIGH_ACTIVATION, 0, 1) + "): "
              + nf(pcts[0], 0, 1) + "%"
              + "  rms=" + nf(rms, 0, 3));
      println("  inactive (a<" + nf(Neuron.RELU_INACTIVE_THRESHOLD, 0, 2) + "): "
              + nf(pcts[1], 0, 1) + "%");
    } else {
      println("  sat (|a|>0.9): " + nf(pcts[0], 0, 1) + "%"
              + "  rms=" + nf(rms, 0, 3));
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
