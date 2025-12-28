FROM pgvector/pgvector:0.8.1-pg17-trixie

RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends ca-certificates lsb-release wget; \
  wget -O /tmp/groonga-apt-source.deb "https://packages.groonga.org/debian/groonga-apt-source-latest-$(lsb_release --codename --short).deb"; \
  apt-get install -y --no-install-recommends /tmp/groonga-apt-source.deb; \
  rm -f /tmp/groonga-apt-source.deb; \
  apt-get update; \
  apt-get install -y --no-install-recommends postgresql-17-pgdg-pgroonga; \
  rm -rf /var/lib/apt/lists/*
