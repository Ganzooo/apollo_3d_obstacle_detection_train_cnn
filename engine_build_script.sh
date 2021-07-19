# Static shape
./bin/trtexec --explicitBatch \
          --onnx=./bin/bcnn_bestmodel_0712_mid_new.onnx \
          --verbose=true \
          --workspace=4096 \
          --fp16 \
          --saveEngine=./bin/bcnn_bestmodel_0712_mid_new.engine