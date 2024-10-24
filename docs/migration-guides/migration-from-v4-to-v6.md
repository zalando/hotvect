# Migrating from Hotvect v4 to Hotvect v6

## Why you should migrate
### API improvements
 - Since Hotvect v5, algorithms must be referred to in configuration files / codes by their algorithm name, rather than by the algorithm name AND algorithm version. This removes the headache of having to update configuration files / code / commands everytime you increment the algorithm version.
 - Similarly, algorithms can use parameters generated by a different algorithm version. For safety, by default online operations fail unless the algorithm version match. This can be turned off by specifying `strictVersionCheck` to false in the `AlgorithmInstanceFactory`.
 - Data source files can now be specified as directories, instead of having to list every file comma separated. This improves the readability of logs and also prevents "Too many arguments" errors when the number of files grow too large. Source files within a directory are read in order of their file names (alphabetical order).

### Performance improvements
 - Encoding performance is improved via the use of unordered processing. Previously, the order of data in the encoded file matched the order in the source file. From v5 on, by default the order is not enforced. Previous behavior can be restored by specifying `transformer_parameter.ordering: "ordered"` in the algorithm definition file. You can also supply `--ordered` argument when invoking the encoding command. In practice the order tends to generally match the order in the source file even if it's not `ordered`. Importantly, the order is not shuffled in any way - it entirely depends on the running condition at the time. For GBT algorithms, `unordered` is the appropriate mode as the training is mostly unaffected by the order of data.

### Meta learning
 - Hotvect allowed "composition" of algorithms in which an algorithm could use other algorithms as its dependencies (components). For example, a greedy ranker could use another algorithm (e.g. an add2cart classifier) as its scoring algorithm. Or, it could use multiple algorithms and average their predictions etc. Since v6 this feature is expanded to allow the following:
   - (Virtually infinite) level of dependence. In v4, only one level of nesting was allowed (an algorithm that is a dependency of another algorithm could not have its own dependencies). Since v6 this restriction is lifted. For example, a greedy ranker can have a model as its dependency, which in turn can have other models as its dependencies and so on.
   - Before v6, only Algorithms (like Rankers or Scorers) could have dependencies. Since v6, Encoders can also be a composite, which allows meta learning like stacking. A composite Encoder can use other algorithms (like a model) to produce a feature value which is then encoded as part of training data.

### Caching of feature transformations
Since v6, "Memoized" variants of Rankers and BulkScorers are available. These variants come with memoization feature that allows arbitrary transformations to be memoized (cached). This cache is valid across nested algorithms and threads. That is, result of computation that was done by an algorithm can be re-used by an algorithm that depends on it without having to perform the calculation again. Similarly, if one thread had already calculated a value for an operation, subsequent calculation by any thread can re-use that cached result. For example, if we have an add2cart model that uses a click model as its dependency, the feature extraction that was done for the click model can be reused by the add2cart model.

Note it is not only the feature values (i.e. the `RawValue` object) that can be cached. Rather, any arbitrary java object can be cached. This eliminates the need for intermediate objects to cache computation results (so called "pre-processed" objects) that was in use since hotvect v4.

Importantly: usually the cached object is accessed by multiple threads. Therefore, the cached object should either be immutable or effectively immutable. For example, if arrays are cached, the caller MUST NOT mutate the array.

#### How is caching implemented
Caching is implemented via memoization pattern. When a function is invoked for the second time with the same argument, instead of re-doing the calculation, the saved calculation result is returned. For efficiency reasons, the cached value is stored into a context object that accompanies the original input. This context is passed across the nested algorithms and the threads that work on the calculation. The actual cache is an optimized hashmap.

#### What is registered transformations?
Registered transformations are pairs of `(RankingFeatureComputationId, Function[ACTION|SHARED|(SHARED, ACTION), Any]>`, and are used to define feature extraction for the memoized variants of Transformers. Any transformation can "lookup" the computation result of another registered transformation. If caching is enabled for that transformation, and the results are already available, the cached result is returned. If not, the value is computed at that point.  

If memoized variants of Transformers are used, the final feature transformation must be registered as a transformation (in other words, the final feature extractions are a subset of registered transformation). In addition to being registered transformations, final feature transformations must 1) use a special subinterface of `RankingFeatureComputationId`, namely `RankingFeatureNamespace`, or even more specialized interfaces like `CatBoostRankingFeatureNamespace` if ML-library plugin specific feature types are supported, and, 2) the return type of the transformation must be `RawValue` (although some ML-library plugins may allow or even require other return types - for this please refer to the documentation of the ML-library plugin). Then, these transformations can be declared as features to use in e.g. the algorithm definition json.

#### Note on implementation
For efficiency reasons, `RankingFeatureComputationId` must be an enum that implements this interface (so that includes any non-final-feature transformation and final-feature transformations). The `Function` can have three types of argument, `SHARED`, `ACTION` or a pair of `(SHARED, ACTION)`. The last one (pair of shared and action) is referred to as `interaction`. The interface for these Functions have the naming pattern `Memoized(Action|Shared|Interaction)Transfromation`. When caching is enabled, it is done for each computation ID and for each distinct input (so that for `ACTION` and `interaction`, the cache is only valid for that action being processed, but for `SHARED` it is valid for all actions. Hence, shared transformations are generally more suited for caching.

 - Use caching sparingly:
   - A registered transformation (i.e. a computation) is by default not cached, but caching can be turned-on by specifying CachingMode=CACHED
   - Caching introduces overhead, and hence in most cases no-caching is faster. Cache only operations that are 1) re-used over and over, and 2) are generally expensive. You should start with no-cache, and then performance test adding caching.
   - Both final-feature transformations and non-final-feature transformations can be cached. For example, it may make sense to cache an expensive shared feature because it has to be written down for every action.
 - Do not excessively break up feature extractions into registered transformations:
   - Unlike normal functions, registered transformations cannot be inlined. Hence, excessively breaking down feature extractions into registered transformations may increase inefficiency.
 - Be weary of type-unsafety:
   - Unfortunately, registered transformations are not type-safe. The code implicitly casts the result for you so that you don't have to keep explicitly casting, but that means if you use the wrong types, it fails at run time. 

### Transformations builder as a replacement of TransformationsFactory
Until hotvect v5, there was a utility to help declaring feature transformations called `TransformationsFactory`. Since v6, this is deprecated. A replacement is supplied for the CatBoost integration plugin as `CatBoostRankingTransformation`, which provides a builder like interface.  

#### Note on CatBoostRankingTransformation
Unfortunately, a) the actual feature transformations are often "parameterized" using the algorithm definition and/or using "encoded" or "prediction" parameter files, and at the same time b) the CatBoost library requires the feature extraction process to separately write a ColumnDescription file. This was a problem because in order to have a complete feature transformation, we need to read these parameters, but if we only wanted to write the ColumnDescription file, all this reading and parsing of parameters would be wasted (as we don't need to perform any transformation yet - we just need to know the name of features and their type). In order to solve this, the CatBoostRankingTransformation has two phases. Once is the "stub" stage where all information about the input class and the available features are provided but the actual transformation is not declared yet, and the "builder" stage where the missing, actual transformation is provided so that an actual instance of Transformer can be instantiated.

The "stub" is then used by the ColumnDescriptionWriter.


### Forkjoin enabled CatBoost bulk scorer
A new CatBoostBulkScorer is introduced, which uses ForkJoin framework to reduce the latency at high percentiles. 

## Backward compatibility and known compatibility issues
 - Algorithms developed/trained using hotvect v4 can be ran (used in production) on hotvect v6, unless they use deprecated hotvect v3 APIs
 - Algorithms developed using hotvect v4 cannot be trained using hotvect v6
 - There are a number of hotvect v4 APIs that are deprecated for which replacements are provided in a dedicated "v6" package. The aim is to replace the v4 package with these in future (so that eventually the dedicated "v6" packages are removed as well)
 - Offline test results created using hotvect v4 can not be downloaded using hotvect v5. So you have to switch versions to download
 - Hotvect v6 can analyze offline test results produced with v4 provided the the results were downloaded, but due to a bug if the algorithm name pattern is set to the default `.*`, it fails to extract offline test results created by v4. To work around this issue, provide as algorithm name pattern `<algorithm-name>$` .
