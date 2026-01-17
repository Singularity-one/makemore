package com.makemore;

import com.makemore.mlp.Tensor;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Data loader for names dataset
 * Same as Lecture 3 but extracted into separate class
 */
public class DataLoader {

    private List<String> words;
    private Map<Character, Integer> charToIdx;
    private Map<Integer, Character> idxToChar;
    private int vocabSize;

    private Tensor Xtr, Ytr;
    private Tensor Xdev, Ydev;
    private Tensor Xte, Yte;

    /**
     * Load words from file and build vocabulary
     */
    public void loadData(String filename) throws IOException {
        InputStream inputStream = Main.class.getResourceAsStream("/names.txt");
        if (inputStream == null) {
            throw new FileNotFoundException("Cannot find names.txt in resources folder");
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        words = reader.lines().toList();

        // Build vocabulary
        Set<Character> chars = new TreeSet<>();
        for (String word : words) {
            for (char c : word.toCharArray()) {
                chars.add(c);
            }
        }

        // Create mappings (0 is reserved for '.')
        charToIdx = new HashMap<>();
        idxToChar = new HashMap<>();

        charToIdx.put('.', 0);
        idxToChar.put(0, '.');

        int idx = 1;
        for (Character c : chars) {
            charToIdx.put(c, idx);
            idxToChar.put(idx, c);
            idx++;
        }

        vocabSize = idx;

        System.out.println("Loaded " + words.size() + " words");
        System.out.println("Vocabulary size: " + vocabSize);
    }

    /**
     * Build training dataset with given block size
     */
    public void buildDataset(int blockSize) {
        List<int[]> X = new ArrayList<>();
        List<Integer> Y = new ArrayList<>();

        for (String word : words) {
            int[] context = new int[blockSize];
            Arrays.fill(context, 0);

            String extWord = word + ".";
            for (char ch : extWord.toCharArray()) {
                int ix = charToIdx.get(ch);
                X.add(Arrays.copyOf(context, blockSize));
                Y.add(ix);

                // Shift context window
                System.arraycopy(context, 1, context, 0, blockSize - 1);
                context[blockSize - 1] = ix;
            }
        }

        // Convert to tensors
        int n = X.size();
        double[] xData = new double[n * blockSize];
        double[] yData = new double[n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < blockSize; j++) {
                xData[i * blockSize + j] = X.get(i)[j];
            }
            yData[i] = Y.get(i);
        }

        Tensor allX = new Tensor(xData, new int[]{n, blockSize});
        Tensor allY = new Tensor(yData, new int[]{n});

        // Split: 80% train, 10% dev, 10% test
        int n1 = (int)(0.8 * n);
        int n2 = (int)(0.9 * n);

        Xtr = sliceTensor(allX, 0, n1);
        Ytr = sliceTensor(allY, 0, n1);

        Xdev = sliceTensor(allX, n1, n2);
        Ydev = sliceTensor(allY, n1, n2);

        Xte = sliceTensor(allX, n2, n);
        Yte = sliceTensor(allY, n2, n);

        System.out.println("Training examples: " + Xtr.getShape()[0]);
        System.out.println("Dev examples: " + Xdev.getShape()[0]);
        System.out.println("Test examples: " + Xte.getShape()[0]);
    }

    private Tensor sliceTensor(Tensor t, int start, int end) {
        int[] shape = t.getShape();
        int dim = shape.length == 1 ? 1 : shape[1];
        int count = end - start;

        double[] sliceData = new double[count * dim];
        System.arraycopy(t.getData(), start * dim, sliceData, 0, count * dim);

        int[] sliceShape = shape.length == 1 ? new int[]{count} : new int[]{count, dim};
        return new Tensor(sliceData, sliceShape);
    }

    // Getters
    public int getVocabSize() { return vocabSize; }
    public Map<Character, Integer> getCharToIdx() { return charToIdx; }
    public Map<Integer, Character> getIdxToChar() { return idxToChar; }
    public Tensor getXtr() { return Xtr; }
    public Tensor getYtr() { return Ytr; }
    public Tensor getXdev() { return Xdev; }
    public Tensor getYdev() { return Ydev; }
    public Tensor getXte() { return Xte; }
    public Tensor getYte() { return Yte; }
}