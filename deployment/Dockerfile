# Multi-stage build: compile the C++ pricing engine (csrc/) in a full
# build image, then ship only the compiled extension + Python runtime in a
# slim final image -- keeps build-essential/cmake out of the deployed
# container entirely.

# ---------------------------------------------------------------------------
# Stage 1: build the qengine C++ extension
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pybind11

COPY csrc/ csrc/
RUN cmake -S csrc -B csrc/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build csrc/build -j"$(nproc)"
# csrc/CMakeLists.txt sets LIBRARY_OUTPUT_DIRECTORY to ../core, so the
# compiled extension lands at /build/core/qengine*.so

# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN groupadd --system quant && useradd --system --gid quant --create-home --home-dir /app quant

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code (core/'s own *.so is excluded via .dockerignore -- the
# only qengine extension shipped is the one just compiled in Stage 1).
COPY core/ core/
COPY alpha/ alpha/
COPY risk/ risk/
COPY backtest/ backtest/
COPY data/ data/
COPY main.py .
COPY deployment/entrypoint.sh deployment/entrypoint.sh

COPY --from=build /build/core/qengine*.so core/

RUN chmod +x deployment/entrypoint.sh && \
    mkdir -p /app/data/db && \
    chown -R quant:quant /app

USER quant

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

VOLUME ["/app/data/db"]

ENTRYPOINT ["deployment/entrypoint.sh"]
CMD ["--mode", "prototype", "--fixture"]
