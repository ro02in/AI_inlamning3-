// Save / load trained NeuralPolicy weights and active training configuration.
// File format: text .curlmodel with header metadata + @policy blocks.
class ModelStorage {
    static final String FORMAT_VERSION = "curling_model_v1";
    static final String DEFAULT_DIR    = "data/models";

    String lastMessage = "";

    boolean saveActiveModel(String path) {
        ArrayList<String> lines = new ArrayList<String>();
        lines.add(FORMAT_VERSION);
        lines.add("use_ensemble=" + (useEnsemble ? 1 : 0));
        lines.add("use_gradients=" + (useGradients ? 1 : 0));
        lines.add("training_done=" + trainingDone);

        StringBuilder body = new StringBuilder();
        int count = appendPolicies(body);
        lines.add("policy_count=" + count);
        lines.add("");

        String[] bodyLines = split(body.toString(), "\n");
        for (String l : bodyLines) {
            if (l.length() > 0) lines.add(l);
        }

        String[] out = lines.toArray(new String[lines.size()]);
        ensureParentDir(path);
        saveStrings(path, out);
        lastMessage = "Sparad (" + count + " policy) → " + path;
        println(lastMessage);
        return true;
    }

    boolean loadActiveModel(String path) {
        String[] lines = loadStrings(path);
        if (lines == null || lines.length == 0) {
            lastMessage = "Kunde inte läsa: " + path;
            return false;
        }

        boolean fileEnsemble  = useEnsemble;
        boolean fileGradients = useGradients;
        int fileTrainingDone  = trainingDone;
        int policyCount       = 0;

        for (String raw : lines) {
            String line = trim(raw);
            if (line.length() == 0 || line.startsWith("#")) continue;
            if (line.startsWith("@")) break;
            int eq = line.indexOf('=');
            if (eq <= 0) continue;
            String key = line.substring(0, eq);
            String val = line.substring(eq + 1);
            if      (key.equals("use_ensemble"))   fileEnsemble  = parseInt(val) != 0;
            else if (key.equals("use_gradients"))  fileGradients = parseInt(val) != 0;
            else if (key.equals("training_done"))  fileTrainingDone = parseInt(val);
            else if (key.equals("policy_count"))     policyCount = parseInt(val);
        }

        useEnsemble  = fileEnsemble;
        useGradients = fileGradients;
        trainingDone = fileTrainingDone;

        ArrayList<LoadedPolicy> loaded = new ArrayList<LoadedPolicy>();
        int idx = 0;
        while (idx < lines.length) {
            if (trim(lines[idx]).startsWith("@policy ")) {
                LoadedPolicy lp = new LoadedPolicy();
                lp.name = trim(lines[idx]).substring(8);
                lp.policy = new NeuralPolicy();
                idx = lp.policy.loadFromLines(lines, idx);
                lp.updateCount = lp.policy.lastLoadedUpdateCount;
                loaded.add(lp);
            } else {
                idx++;
            }
        }

        if (loaded.isEmpty()) {
            lastMessage = "Ingen policy hittades i " + path;
            return false;
        }

        applyLoadedPolicies(loaded);
        lastMessage = "Laddad (" + loaded.size() + " policy, "
            + modeLabel() + ") ← " + path;
        println(lastMessage);
        return true;
    }

    void applyLoadedPolicies(ArrayList<LoadedPolicy> loaded) {
        if (useGradients && useEnsemble) {
            int n = min(gradientEnsemble.count, loaded.size());
            for (int i = 0; i < n; i++) {
                gradientEnsemble.trainers[i].setPolicy(loaded.get(i).policy,
                    loaded.get(i).updateCount);
            }
        } else if (useGradients) {
            LoadedPolicy lp = loaded.get(0);
            pgTrainer.setPolicy(lp.policy, lp.updateCount);
        } else if (useEnsemble) {
            int n = min(ensemble.count, loaded.size());
            for (int i = 0; i < n; i++) {
                ensemble.trainers[i].setPolicy(loaded.get(i).policy);
            }
        } else {
            singleTrainer.setPolicy(loaded.get(0).policy);
        }
    }

    int appendPolicies(StringBuilder sb) {
        if (useGradients && useEnsemble) {
            for (int i = 0; i < gradientEnsemble.count; i++) {
                PolicyGradientTraining t = gradientEnsemble.trainers[i];
                t.policy.appendSave(sb, gradientEnsemble.names[i], t.updateCount);
            }
            return gradientEnsemble.count;
        }
        if (useGradients) {
            pgTrainer.policy.appendSave(sb, "Enkel", pgTrainer.updateCount);
            return 1;
        }
        if (useEnsemble) {
            for (int i = 0; i < ensemble.count; i++) {
                ensemble.trainers[i].current.appendSave(sb, ensemble.names[i], -1);
            }
            return ensemble.count;
        }
        singleTrainer.current.appendSave(sb, "Enkel", -1);
        return 1;
    }

    String modeLabel() {
        String m = useEnsemble ? "ensemble" : "enkel";
        String a = useGradients ? "gradient" : "sökning";
        return m + "+" + a;
    }

    void ensureParentDir(String path) {
        File f = new File(path);
        File parent = f.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
    }

    String defaultSavePath() {
        File dir = new File(sketchPath(DEFAULT_DIR));
        if (!dir.exists()) dir.mkdirs();
        return sketchPath(DEFAULT_DIR + "/model_"
            + year() + nf(month(), 2) + nf(day(), 2)
            + "_" + nf(hour(), 2) + nf(minute(), 2) + nf(second(), 2)
            + ".curlmodel");
    }

    class LoadedPolicy {
        String name;
        NeuralPolicy policy;
        int updateCount = -1;
    }
}
