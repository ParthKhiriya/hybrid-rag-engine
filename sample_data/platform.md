# AI Inference Platform

The Inference Gateway is the north-south entry point for model requests. The Inference Gateway depends on the GPU Cluster. The Inference Gateway is monitored by Prometheus. The Inference Gateway is owned by Platform Engineering.

The GPU Cluster runs on Kubernetes. The GPU Cluster is monitored by NVIDIA DCGM. Kubernetes is monitored by Prometheus. Model-serving pods use vLLM for high-throughput token generation and NVIDIA Triton for standardized inference endpoints where appropriate.

Prometheus forwards SLO-relevant metrics to Grafana dashboards. Alerts cover p95 request latency, token throughput, GPU memory pressure, error rate, queue depth, and pod restart frequency.
