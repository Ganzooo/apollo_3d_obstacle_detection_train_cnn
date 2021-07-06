# Static shape
./trtexec --explicitBatch \
          --onnx=./bin/bcnn_latestmodel_0628.onnx \
          --verbose=true \
          --workspace=4096 \
          --fp16 \
          --saveEngine=./bin/bcnn_latestmodel_0628.engine