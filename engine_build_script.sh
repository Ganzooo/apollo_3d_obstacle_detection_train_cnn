# Static shape
./bin/trtexec --explicitBatch \
          --onnx=./bin/unet_bestmodel_val_512_new.onnx \
          --verbose=true \
          --workspace=4096 \
          --fp16 \
          --saveEngine=./bin/unet_bestmodel_val_512_new.engine