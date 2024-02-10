from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
import inspect


class ClusteringFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_clustering_algorithm(cluster_name):
        clustering_type = ClusteringFactory.get_clustering_type(cluster_name)
        param_dict = ClusteringFactory.get_cluster_param_dict(cluster_name)

        return clustering_type.__init__(**param_dict)

    @staticmethod
    def get_cluster_param_dict(cluster_name):
        clustering_type = ClusteringFactory.get_clustering_type(cluster_name)
        signature = inspect.signature(clustering_type.__init__)
        param_string = str(signature).replace("(", "").replace(")", "")
        param_list = param_string.split(", ")
        param_list.remove("self")
        if "**kwargs" in param_list:
            param_list.remove("**kwargs")
        if "*args" in param_list:
            param_list.remove("*args")
        if "*" in param_list:
            param_list.remove("*")

        param_dict = {p.split("=")[0]: p.split("=")[1] for p in param_list}
        return param_dict

    @staticmethod
    def get_clustering_type(cluster_name):
        if cluster_name.lower() == "kmeans":
            clustering_type = KMeans
        elif cluster_name.lower() == "dbscan":
            clustering_type = DBSCAN
        elif cluster_name.lower() == "hdbscan":
            clustering_type = HDBSCAN
        elif cluster_name.lower() == "spectral":
            clustering_type = SpectralClustering
        elif cluster_name.lower() == "gmm":
            clustering_type = GaussianMixture
        else:
            raise Exception(f"Invalid algorithm name {cluster_name}")

        return clustering_type



