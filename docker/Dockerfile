# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
# edited the Jupyter Scipy Notebook
ARG BASE_CONTAINER=jupyter/scipy-notebook@sha256:bda04776cd8016f7d55fecb8576a43b9c3fee6cc25ef6de25baae24280272ce2
FROM $BASE_CONTAINER

LABEL maintainer="Eva Fast <evaisfast@gmail.com>"


RUN pip install \
  pandas_profiling \
  pyarrow \
  openpyxl \
  seaborn \
  xgboost \
  hgboost \
  lckr-jupyterlab-variableinspector \
  shap 
