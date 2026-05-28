// Gaussian exploration noise on shot parameters during training evaluation only.
// Decays over comparisons so early search is broad, late search refines.
class ExplorationNoise {
    boolean enabled = true;
    float strength = 1.0f; // UI multiplier on all sigmas

    float curlSigmaBase      = 0.18f;  // on curl [-1, 1]
    float speedSigmaBase     = 4.0f;   // ft/s
    float angleSigmaDegBase  = 2.5f;   // degrees

    float scaleStart = 1.0f;
    float scaleEnd   = 0.15f;
    int   decayComparisons = 400;

    int comparisonsDone = 0;

    void reset() {
        comparisonsDone = 0;
    }

    void onComparison() {
        comparisonsDone++;
    }

    float decayScale() {
        if (!enabled) return 0;
        float t = constrain(comparisonsDone / (float) decayComparisons, 0, 1);
        return lerp(scaleStart, scaleEnd, t);
    }

    Shot perturb(Shot base, NeuralPolicy policy) {
        if (!enabled) return base;
        float s = decayScale() * strength;
        if (s <= 0) return base;

        float curl = base.curl + randomGaussian() * curlSigmaBase * s;
        float speed = base.speed + randomGaussian() * speedSigmaBase * s;
        float angleDeg = degrees(base.angle) + randomGaussian() * angleSigmaDegBase * s;

        curl  = constrain(curl, policy.minCurl, policy.maxCurl);
        speed = constrain(speed, policy.minSpeed, policy.maxSpeed);
        angleDeg = constrain(angleDeg, policy.minAngleDeg, policy.maxAngleDeg);

        return new Shot(curl, speed, radians(angleDeg));
    }
}
