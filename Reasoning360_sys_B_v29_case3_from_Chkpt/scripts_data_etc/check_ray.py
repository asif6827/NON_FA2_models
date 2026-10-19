import ray
ray.init()  # no address, starts a local cluster
print(ray.cluster_resources())
