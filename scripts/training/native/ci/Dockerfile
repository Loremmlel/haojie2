# 只复制源码交付包；训练运行根没有宿主仓库或 JS 运行时挂载。
FROM rust:1.98.1-bookworm AS engine
WORKDIR /source
COPY source.tar.gz /tmp/source.tar.gz
RUN tar -xzf /tmp/source.tar.gz -C /source && \
    cargo build --locked --release --manifest-path native/engine/Cargo.toml

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends strace && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY --from=engine /source /work/source
RUN python -m venv /work/venv && \
    /work/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 && \
    strace -f -e trace=execve -o /work/install-processes.log /work/venv/bin/python -m pip install /work/source/training && \
    /work/venv/bin/python /work/source/scripts/training/native/ci/isolation.py --trace /work/install-processes.log --installation && \
    /work/venv/bin/python /work/source/scripts/training/native/ci/isolation.py
ENV HAOJIE_NATIVE=/work/source/native/engine/target/release/haojie-engine
CMD ["/bin/sh", "-ec", "strace -f -e trace=execve -o /evidence/processes.log /work/venv/bin/python /work/source/scripts/training/native/ci/installed.py --engine $HAOJIE_NATIVE --source /work/source --output /evidence/closure; /work/venv/bin/python /work/source/scripts/training/native/ci/isolation.py --trace /evidence/processes.log > /evidence/processes.json"]
