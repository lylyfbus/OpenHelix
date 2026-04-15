FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        chromium \
        chromium-driver \
        curl \
        git \
        nodejs \
        npm \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir selenium

COPY helix_exec.sh /usr/local/bin/helix_exec
RUN chmod +x /usr/local/bin/helix_exec

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER=/usr/bin/chromedriver

ENTRYPOINT ["/usr/local/bin/helix_exec"]
