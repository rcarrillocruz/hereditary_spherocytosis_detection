FROM tensorflow/serving:latest

RUN mkdir -p /models/hs-retinanet/1

COPY notebooks/exported_model /models/hs-retinanet/1

ENV MODEL_NAME=hs-retinanet
ENV TF_CPP_MIN_VLOG_LEVEL=4
ENV TF_CPP_VMODULE=http_server=1

ENTRYPOINT tensorflow_model_server --port=8500 --rest_api_port=${PORT:-8051} --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
