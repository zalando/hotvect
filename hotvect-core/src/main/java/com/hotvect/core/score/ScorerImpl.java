package com.hotvect.core.score;

import com.hotvect.api.algodefinition.scoring.ScoringVectorizer;
import com.hotvect.api.algorithms.Scorer;
import com.hotvect.api.data.DataRecord;
import com.hotvect.api.data.SparseVector;


/**
 * A {@link Scorer} which uses the specified {@link ScoringVectorizer} and {@link Estimator}
 * @param <RECORD>
 */
public class ScorerImpl<RECORD> implements Scorer<RECORD> {
    private final ScoringVectorizer<RECORD> scoringVectorizer;
    private final Estimator estimator;

    /**
     * Constructs a {@link ScorerImpl}
     * @param scoringVectorizer the {@link ScoringVectorizer} to use
     * @param estimator the {@link Estimator} to use
     */
    public ScorerImpl(ScoringVectorizer<RECORD> scoringVectorizer, Estimator estimator) {
        this.scoringVectorizer = scoringVectorizer;
        this.estimator = estimator;
    }

    /**
     * Given a raw {@link DataRecord}, calculate the score
     * @param input the raw {@link DataRecord} to score
     * @return the score
     */
    @Override
    public double applyAsDouble(RECORD input) {
        SparseVector vectorized = scoringVectorizer.apply(input);
        return estimator.applyAsDouble(vectorized);
    }
}
