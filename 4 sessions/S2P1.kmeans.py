import random
import numpy as np
from collections import defaultdict
import os
import sys
from time import time


class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id


class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []

    def reset_members(self):
        self._members = []

    def add_member(self, member):
        self._members.append(member)


class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        self._E = []
        self._S = 0

    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(":")[0])
                tfidf = float(index_tfidf.split(":")[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()

        with open(os.getcwd()+"/20news-bydate/20news-full-words-idfs.txt") as f:
            vocab_size = len(f.read().splitlines())

        self._data = []
        self._label_count = defaultdict(int)
        for d in d_lines:
            features = d.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            print(f"Loading {label} {doc_id}")
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))

    def compute_similarity(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def random_init(self, seed_value_id):
        start_time = time()
        set_candidates = set(range(len(self._data)))
        set_candidates.remove(seed_value_id)
        self._E.append(self._data[seed_value_id])
        while len(self._E) < self._num_clusters:
            new_centroid_id = None
            min_similarity_val = 1
            for i in set_candidates:
                local_max_similarity = -1
                for centroid in self._E:
                    local_max_similarity = max(local_max_similarity, self.compute_similarity(self._data[i]._r_d, centroid._r_d))
                if local_max_similarity < min_similarity_val:
                    new_centroid_id = i
                    min_similarity_val = local_max_similarity
            if new_centroid_id:
                print(f"Got new centroid {len(self._E)}: {new_centroid_id}")
                set_candidates.remove(new_centroid_id)
                self._E.append(self._data[new_centroid_id])
            else:
            	raise Exeption("Could not find enough centroids")
        for i in range(len(self._clusters)):
            self._clusters[i]._centroid = self._E[i]._r_d
        end_time = time()
        print("Execution time: ", end_time - start_time)

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member._r_d, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity

    def update_centroid_of(self, cluster):
        aver_r_d = np.mean([member._r_d for member in cluster._members], axis=0)
        new_centroid = aver_r_d/np.linalg.norm(aver_r_d)
        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        criteria = ["centroid", "similarity", "max_iters"]
        assert criterion in criteria
        if criterion == "max_iters":
            print(f"Iterations: {self._iteration}")
            if self._iteration >= threshold:
                return True
            return False

        elif criterion == "centroid":        
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
            print(f"Number centroid changed: {len(E_new_minus_E)}")
            self._E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            return False
            
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            print(f"S changed: {new_S_minus_S}")
            if abs(new_S_minus_S) <= threshold:
                return True
            return False

    def run(self, seed_value_id, criterion, threshold):
        self.random_init(seed_value_id)
        self._iteration = 0
        while True:
            self._iteration += 1
            print("Processing iteration:", self._iteration)
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            if self.stopping_condition(criterion, threshold):
                print("Catch stopping condition by criterion:", criterion)
                break

    def compute_purity(self, num_clusters):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(num_clusters)])
            majority_sum += max_count
        return majority_sum * 1 / len(self._data)

    def compute_NMI(self, num_clusters):
        I_value, H_O, H_C, N = 0.0, 0.0, 0.0, len(self._data)

        for cluster in self._clusters:
            wk = len(cluster._members) * 1.0
            H_O += -wk / N * np.log10(wk / N)

        for label in range(num_clusters):
            cj = self._label_count[label] * 1.0
            H_C += -cj / N * np.log10(cj / N)

        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            for label in range(num_clusters):
                wk_cj = member_labels.count(label) * 1.0
                wk = len(cluster._members) * 1.0
                cj = self._label_count[label] * 1.0
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)

        return I_value * 2.0 / (H_C + H_O)

if __name__ == "__main__":
    # Create kmeans sample
    num_clusters = 20
    k = Kmeans(num_clusters)

    # Load data
    k.load_data(os.getcwd()+"/20news-bydate/20news-full-tf-idf.txt")

    # Clusters 
    print(f"Given number clusters: {num_clusters}")
    print(f"Number clusters: {len(k._label_count)}")
    print(f"Label dictionary: {k._label_count}")

    # Running
    k.run(seed_value_id=random.choice(range(len(k._data))), criterion="centroid", threshold=0)

    # Measurements
    print(f"Purity: {k.compute_purity(num_clusters)}")
    print(f"NMI: {k.compute_NMI(num_clusters)}")
