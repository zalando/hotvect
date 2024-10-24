package com.hotvect.api.transformation.ranking;

import com.hotvect.api.data.RawValue;

import java.util.function.Function;

public interface ActionTransformation<ACTION> extends Function<ACTION, RawValue> {
}
