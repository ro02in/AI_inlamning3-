// Penalises policies that fire nearly the same shot on every layout in one comparison.
// Curl and angle (especially same sign/direction) are penalised fully; speed at 1/5 weight.
class ShotSimilarityPenalty {
    static final float AXIS_WEIGHT        = 1.5f;
    static final float SPEED_WEIGHT       = 0.2f;
    static final float CURL_RANGE_THRESH  = 0.25f;
    static final float ANGLE_RANGE_THRESH = 0.12f; // ~7 degrees in radians
    static final float SPEED_RANGE_THRESH = 6.0f;
    static final float SIGN_EPS           = 0.05f;
    static final int   MIN_SHOTS          = 3;

    float penalty(ArrayList<Shot> shots) {
        if (shots == null || shots.size() < MIN_SHOTS) return 0;

        float[] curl   = new float[shots.size()];
        float[] angle  = new float[shots.size()];
        float[] speed  = new float[shots.size()];
        for (int i = 0; i < shots.size(); i++) {
            Shot s = shots.get(i);
            curl[i]  = s.curl;
            angle[i] = s.angle;
            speed[i] = s.speed;
        }

        return axisPenalty(curl, CURL_RANGE_THRESH, true)
             + axisPenalty(angle, ANGLE_RANGE_THRESH, true)
             + axisPenalty(speed, SPEED_RANGE_THRESH, false) * SPEED_WEIGHT;
    }

    float axisPenalty(float[] values, float rangeThresh, boolean penaliseSameSign) {
        float vMin = values[0], vMax = values[0];
        int positive = 0, negative = 0;
        for (float v : values) {
            vMin = min(vMin, v);
            vMax = max(vMax, v);
            if (v > SIGN_EPS)       positive++;
            else if (v < -SIGN_EPS) negative++;
        }

        float range = vMax - vMin;
        // 0 = varied shots, 1 = all nearly identical values
        float rangeScore = constrain(1.0f - range / rangeThresh, 0, 1);

        float signScore = 0;
        if (penaliseSameSign) {
            if (positive == values.length || negative == values.length) {
                signScore = 1.0f; // always curl/angle the same direction
            } else if (positive == 0 && negative == 0) {
                signScore = 0.3f; // always near zero — mild penalty
            }
        }

        return AXIS_WEIGHT * (rangeScore + signScore);
    }
}
