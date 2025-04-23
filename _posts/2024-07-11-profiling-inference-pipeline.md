---
title: Profiling an inference pipeline
date: 2024-07-11 19:04:00 +0530
categories: [Optimization, Benchmarking]
tags: [ML, GPU]
math: true
pin: false
image:
  path: https://www.shutterstock.com/image-photo/estimate-approximate-symbol-businessman-turns-600nw-2187747311.jpg
  alt: Profiling an inference pipeline
---

## Summary

- Profile a sample inference pipeline
- Identify the bottlenecks: CPU vs GPU
- When we are CPU bound: Eg. Use NeMO GPU dataloader
- When we are GPU bound: Eg. ??
- Ensure overlap of CPU and GPU: Eg. Ensure CPU is ready to serve to GPU 
- Hidden I/O costs of moving back and forth from GPU