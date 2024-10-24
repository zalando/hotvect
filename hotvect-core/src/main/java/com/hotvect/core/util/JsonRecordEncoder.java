package com.hotvect.core.util;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ImmutableMap;
import com.hotvect.api.codec.scoring.ScoringExampleEncoder;
import com.hotvect.api.data.DataRecord;
import com.hotvect.api.data.Namespace;
import com.hotvect.api.data.RawValue;
import com.hotvect.api.data.SparseVector;

import java.util.*;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

/**
 * A {@link ScoringExampleEncoder} that encodes a {@link DataRecord} into a JSON String
 * @param <K>
 */
public class JsonRecordEncoder<K extends Enum<K> & Namespace> implements Function<DataRecord<K, RawValue>, String> {
    private static final ObjectMapper OM = new ObjectMapper();

    private Map<String, ?> pojonize(DataRecord<?, RawValue> input) {
        return input.asEnumMap().entrySet().stream().map(x -> {
            String k = x.getKey().toString();
            Object value = pojonize(x.getValue());
            return new AbstractMap.SimpleEntry<>(k, value);
        }).collect(toMap(AbstractMap.SimpleEntry::getKey, AbstractMap.SimpleEntry::getValue));
    }

    private static Object pojonize(RawValue value) {
        switch (value.getValueType()) {
            case SINGLE_STRING:
                return value.getSingleString();
            case STRINGS:
                return Arrays.asList(value.getStrings());
            case SINGLE_CATEGORICAL:
                return value.getSingleCategorical();
            case CATEGORICALS:
                return Arrays.stream(value.getCategoricals()).boxed().collect(toList());
            case SINGLE_NUMERICAL:
                return value.getSingleNumerical();
            case DENSE_VECTOR:{
                List<Double> ret = new ArrayList<>();
                for (double v : value.getNumericals()) {
                    ret.add(v);
                }
                return ret;
            }
            case SPARSE_VECTOR:
            case CATEGORICALS_TO_NUMERICALS: {
                SparseVector vector = value.getSparseVector();
                int[] names = vector.getNumericalIndices();
                double[] values = vector.getNumericalValues();
                ImmutableMap.Builder<String, Double> ret = ImmutableMap.builder();
                for (int i = 0; i < names.length; i++) {
                    ret.put(String.valueOf(names[i]), values[i]);
                }
                return ret.build();
            }
            case STRINGS_TO_NUMERICALS:
                String[] names = value.getStrings();
                double[] values = value.getNumericals();
                ImmutableMap.Builder<String, Double> ret = ImmutableMap.builder();
                for (int i = 0; i < names.length; i++) {
                    ret.put(String.valueOf(names[i]), values[i]);
                }
                return ret.build();
            default:
                throw new AssertionError();
        }
    }

    @Override
    public String apply(DataRecord<K, RawValue> toEncode) {
        try {
            return OM.writeValueAsString(pojonize(toEncode));
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }


}
