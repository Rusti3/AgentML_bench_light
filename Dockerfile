FROM python@sha256:520153e2deb359602c9cffd84e491e3431d76e7bf95a3255c9ce9433b76ab99a
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/benchsite/lib/python3.12/site-packages
WORKDIR /workspace
