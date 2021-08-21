package com.eshioji.hotvect.commandline;

import com.codahale.metrics.MetricRegistry;
import com.eshioji.hotvect.hotdeploy.CloseableAlgorithmHandle;
import com.eshioji.hotvect.util.VerboseCallable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public abstract class Task<R> extends VerboseCallable<Map<String, String>> {
    protected static final Logger LOGGER = LoggerFactory.getLogger(Task.class);
    protected final Options opts;
    protected final MetricRegistry metricRegistry;
    protected final CloseableAlgorithmHandle<R> closeableAlgorithmHandle;

    public Task(Options opts, MetricRegistry metricRegistry, CloseableAlgorithmHandle<R> closeableAlgorithmHandle) throws Exception {
        this.opts = opts;
        this.metricRegistry = metricRegistry;

        this.closeableAlgorithmHandle = closeableAlgorithmHandle;
    }

    protected abstract Map<String, String> perform() throws Exception;

    @Override
    protected Map<String, String> doCall() throws Exception {
        LOGGER.info("Running {} from {} to {}", this.getClass().getSimpleName(), opts.sourceFile, opts.destinationFile);
        var metadata = perform();
        metadata.put("task_type", opts.encode ? "encode" : "predict");
        metadata.put("metadata_location", opts.metadataLocation.toString());
        metadata.put("destination_file", opts.destinationFile.toString());
        metadata.put("source_file", opts.sourceFile.toString());
        metadata.put("sample_pct", String.valueOf(opts.samplePct));
        metadata.put("sample_seed", String.valueOf(opts.sampleSeed));
        metadata.put("algorithm_name", closeableAlgorithmHandle.getMetadata().getName());
        metadata.put("instance_id", closeableAlgorithmHandle.getMetadata().getInstanceId());
        return metadata;
    }
}
