FROM julia:latest

USER root

#Julia Dependencies
ENV JULIA_PKGDIR /opt/julia
ENV JULIA_DEPOT_PATH /opt/julia

RUN mkdir -p /opt/julia && chmod 777 /opt/julia
RUN mkdir -p /app && chmod 777 /app

COPY requirements.jl /app/
COPY BNN.jl /app/
COPY utils.jl /app/

RUN julia /app/requirements.jl
RUN julia /app/BNN.jl

COPY . /app/

CMD julia /app/main.jl
