package com.eshioji.hotvect.core.vectorization;

import com.eshioji.hotvect.api.data.DataRecord;
import com.eshioji.hotvect.api.data.SparseVector;
import com.eshioji.hotvect.api.data.hashed.HashedValue;
import com.eshioji.hotvect.api.data.raw.RawValue;
import com.eshioji.hotvect.core.TestFeatureNamespace;
import com.eshioji.hotvect.core.TestRawNamespace;
import com.eshioji.hotvect.core.combine.Combiner;
import com.eshioji.hotvect.core.hash.Hasher;
import com.eshioji.hotvect.core.transform.Transformer;
import com.eshioji.hotvect.core.util.AutoMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class VectorizerImplTest {

    @Test
    void apply() {
        var mapper = new AutoMapper<TestRawNamespace, TestFeatureNamespace, RawValue>(TestRawNamespace.class, TestFeatureNamespace.class);
        DataRecord<TestRawNamespace, RawValue> initialInput = new DataRecord<>(TestRawNamespace.class);

        Transformer<DataRecord<TestRawNamespace, RawValue>, TestFeatureNamespace> transformer = record -> {
            assertSame(initialInput, record);
            return mapper.apply(initialInput);
        };

        var hasher = new Hasher<>(TestFeatureNamespace.class);
        var combiner = new Combiner<TestFeatureNamespace>(){
            @Override
            public SparseVector apply(DataRecord<TestFeatureNamespace, HashedValue> toCombine) {
                assertEquals(hasher.apply(mapper.apply(initialInput)), toCombine);
                return new SparseVector(new int[]{1});
            }
        };
        var subject = new VectorizerImpl<>(transformer, hasher, combiner);

        assertEquals(new SparseVector(new int[]{1}), subject.apply(initialInput));
    }
}