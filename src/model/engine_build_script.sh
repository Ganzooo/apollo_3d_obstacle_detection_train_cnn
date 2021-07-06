# Static shape
./trtexec --explicitBatch \
          --onnx=bcnn_latestmodel_0628.onnx \
          --verbose=true \
          --workspace=4096 \
          --fp16 \
          --saveEngine=./bcnn_latestmodel_0628.engine