FROM grafana/grafana:10.0.3

# Set environment variables
ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=admin
ENV GF_INSTALL_PLUGINS="grafana-piechart-panel,grafana-worldmap-panel"
ENV GF_PATHS_PROVISIONING="/etc/grafana/provisioning"
ENV GF_FEATURE_TOGGLES_ENABLE="publicDashboards"

# Create directory structure
RUN mkdir -p /etc/grafana/provisioning/datasources \
    && mkdir -p /etc/grafana/provisioning/dashboards \
    && mkdir -p /var/lib/grafana/dashboards/pulsar

# Copy dashboard provider configuration
COPY provider.yml /etc/grafana/provisioning/dashboards/provider.yml

# Copy dashboards
COPY pulsar_overview_dashboard.json /var/lib/grafana/dashboards/pulsar_overview_dashboard.json
COPY pulsar_broker_dashboard.json /var/lib/grafana/dashboards/pulsar_broker_dashboard.json
COPY pulsar_messaging_dashboard.json /var/lib/grafana/dashboards/pulsar_messaging_dashboard.json

# Create datasource configuration for Prometheus
RUN echo '{\
  "apiVersion": 1,\
  "datasources": [\
    {\
      "name": "Prometheus",\
      "type": "prometheus",\
      "access": "proxy",\
      "url": "http://prometheus:9090",\
      "isDefault": true,\
      "version": 1,\
      "editable": true\
    }\
  ]\
}' > /etc/grafana/provisioning/datasources/prometheus.yaml

# Expose the Grafana port
EXPOSE 3000

# Set health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD wget -q --no-check-certificate -O- http://localhost:3000/api/health || exit 1