package com.eshioji.hotvect.core.audit;

import com.eshioji.hotvect.api.data.FeatureNamespace;
import com.eshioji.hotvect.api.data.hashed.HashedValue;
import com.eshioji.hotvect.api.data.raw.RawValue;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class HasherAuditState {
    private final ConcurrentMap<HashedFeatureName, RawFeatureName> featureName2SourceRawValue = new ConcurrentHashMap<>();

    public void registerSourceRawValue(FeatureNamespace namespace, RawValue toHash, HashedValue hashed) {
        int[] featureNames = hashed.getCategoricals();
        for (int i = 0; i < featureNames.length; i++) {
            String sourceValue = extractSourceValue(toHash, i);
            int featureName = featureNames[i];
            featureName2SourceRawValue.put(new HashedFeatureName(namespace, featureName), new RawFeatureName(namespace, sourceValue));
        }
    }

    private String extractSourceValue(RawValue toHash, int index) {
        switch (toHash.getValueType()) {
            case SINGLE_STRING: return toHash.getSingleString();
            case STRINGS:
            case STRINGS_TO_NUMERICALS:
                return toHash.getStrings()[index];
            case SINGLE_NUMERICAL:
            case SINGLE_CATEGORICAL:
                return "0";
            case CATEGORICALS:
            case CATEGORICALS_TO_NUMERICALS:
                return String.valueOf(toHash.getCategoricals()[index]);
            default: throw new AssertionError();
        }

    }

    public ConcurrentMap<HashedFeatureName, RawFeatureName> getFeatureName2SourceRawValue(){
        return this.featureName2SourceRawValue;
    }
}
