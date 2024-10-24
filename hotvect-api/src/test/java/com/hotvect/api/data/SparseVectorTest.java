package com.hotvect.api.data;

import com.google.common.collect.ImmutableList;
import org.junit.jupiter.api.Test;
import org.quicktheories.api.Pair;
import org.quicktheories.core.Gen;

import java.util.Arrays;
import java.util.function.BiConsumer;

import static com.hotvect.testutils.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.quicktheories.QuickTheory.qt;
import static org.quicktheories.generators.Generate.*;
import static org.quicktheories.generators.SourceDSL.integers;

class SparseVectorTest {
    private final Gen<Double> doubles = pick(ImmutableList.of(Double.MIN_VALUE, -1.0, 0.0, 1.0, Double.MAX_VALUE));
    private final Gen<Pair<int[], double[]>> vectors =  intArrays(integers().between(0, 3), integers().all())
            .mutate((is, rand) -> Pair.of(is, doubleArrays(constant(is.length), doubles).generate(rand)));

    @Test
    void correctlyInitializes() {
        qt().withFixedSeed(10).forAll(vectors).checkAssert(x -> {
            int[] names = x._1;
            double[] values = x._2;
            assert names.length == values.length;

            SparseVector subject = new SparseVector(names, values);

            assertArrayEquals(names, subject.getNumericalIndices());
            assertArrayEquals(values, subject.getNumericalValues());
        });
    }

    @Test
    void correctlyInitializesWithNamesOnly() {
        qt().withFixedSeed(10).forAll(intArrays(integers().between(0, 3), integers().all())).checkAssert(x -> {
            SparseVector subject = new SparseVector(x);

            assertArrayEquals(x, subject.getCategoricalIndices());
        });
    }

    @Test
    void invalidInputs() {
        assertThrows(NullPointerException.class, () -> new SparseVector((int[])null));
        assertThrows(NullPointerException.class, () -> new SparseVector((double[])null));
        assertThrows(NullPointerException.class, () -> new SparseVector(null, null));
        assertThrows(IllegalArgumentException.class, () -> new SparseVector(new int[1], new double[2]));
    }




    @Test
    void equality() {
        Gen<Pair<int[], double[]>> data = testVectors();
        BiConsumer<Pair<int[], double[]>, Pair<int[], double[]>> assertSparseVectors = (x, y) -> {
            SparseVector xp = new SparseVector(x._1, x._2);
            SparseVector yp = new SparseVector(y._1, y._2);

            boolean equal = Arrays.equals(x._1, y._1) && Arrays.equals(x._2, y._2);
            assertEquals(equal, xp.equals(yp));
            if (equal){
                assertEquals(xp.hashCode(), yp.hashCode());
            }
        };

        qt().withFixedSeed(10).forAll(data, data)
                .assuming((x, y) -> !Arrays.equals(x._1, y._1) || !Arrays.equals(x._2, y._2))
                .checkAssert(assertSparseVectors);

        qt().withFixedSeed(10).forAll(generateEquals(data)).checkAssert(x -> assertSparseVectors.accept(x._1, x._2));

    }
}