ARG TEMPO_IMAGE_TAG=2.9.0
FROM busybox:1.37.0-uclibc AS probe
RUN mkdir /tempo-data && chown 10001:10001 /tempo-data
FROM grafana/tempo:${TEMPO_IMAGE_TAG}
COPY --from=probe /bin/busybox /opt/probe/busybox
COPY --from=probe --chown=10001:10001 /tempo-data /var/tempo
USER 10001:10001
