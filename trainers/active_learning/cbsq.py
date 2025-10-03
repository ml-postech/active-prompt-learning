from .AL import AL
import torch
import copy
import numpy as np

from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class CBSQ(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, val_set, n_class, device, idx, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.labeled_in_set = val_set
        self.device= device
        self.idx = idx
        
    def get_features(self):
        if self.idx:
            self.model.eval()
        labeled_features, unlabeled_features = None, None
        with torch.no_grad():
            labeled_in_loader = build_data_loader(
                self.cfg,
                data_source=self.labeled_in_set, 
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            labels_org, labels_pse = np.array([]), np.array([])
            scores = np.array([])
            # generate entire labeled_in features set
            for data in labeled_in_loader:
                inputs = data["img"].cuda()
                labels = data["label"].cuda()
                preds, img_features, txt_features = self.model(inputs, get_feature=True, get_text_feature=True)
                p_labels = torch.argmax(preds, dim=1)

                labels_org = np.append(labels_org, labels.cpu().numpy())
                labels_pse = np.append(labels_pse, p_labels.cpu().numpy())

                # confidence
                preds = torch.nn.functional.softmax(preds, dim=1)
                preds_np = preds.cpu().numpy()
                score = np.max(preds_np, axis=1)
                scores = np.append(scores, score)

                dot_features = torch.matmul(preds, txt_features)
                features = torch.cat([img_features, dot_features], axis=1)

                if labeled_features is None:
                    labeled_features = features
                else:
                    labeled_features = torch.cat((labeled_features, features), 0)

            # generate entire unlabeled features set
            for data in unlabeled_loader:
                inputs = data["img"].cuda()
                labels = data["label"]
                if self.idx:
                    preds, img_features, txt_features = self.model(inputs, get_feature=True, get_text_feature=True)
                else:
                    preds, img_features, txt_features = self.model.model_inference(inputs, get_feature=True)
                p_labels = torch.argmax(preds, dim=1)

                labels_org = np.append(labels_org, labels.cpu().numpy())
                labels_pse = np.append(labels_pse, p_labels.cpu().numpy())

                # confidence
                preds = torch.nn.functional.softmax(preds, dim=1)
                preds_np = preds.cpu().numpy()
                score = np.max(preds_np, axis=1)
                scores = np.append(scores, score)

                dot_features = torch.matmul(preds, txt_features)
                features = torch.cat([img_features, dot_features], axis=1)
                
                if unlabeled_features is None:
                    unlabeled_features = features
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)
            
        return labeled_features, unlabeled_features, scores, labels_org, labels_pse

    def k_means_uncert(self, labeled, unlabeled, scores, labels_org, labels_pse, n_query):
        if self.idx != 0:
            label_len = labeled.size()[0]
            concat = torch.cat([labeled, unlabeled], dim=0).cpu().numpy()

            thresholds = {c : 99 for c in range(self.n_class)}
            for c in range(self.n_class):
                idxs = np.where(labels_org[:label_len] == c)
                if len(idxs[0]) == 0: continue
                thresholds[c] = np.mean(scores[idxs])
        else:
            label_len = 0
            concat = unlabeled.cpu().numpy()
            thresholds = {c : 99 for c in range(self.n_class)}
        
        concat = concat / np.linalg.norm(concat, axis=1, keepdims=True)
        if self.idx == 0: kmeans = KMeans(n_clusters=self.n_class * 4, random_state=42)
        else: kmeans = KMeans(n_clusters=self.n_class * 4 + label_len, random_state=42) # self.n_class + label_len
        kmeans.fit(concat)
        clusters = kmeans.labels_
        centroids = kmeans.cluster_centers_
        uniques = np.unique(np.array(clusters))

        corr = 0
        budget_saving = 0
        if self.idx == 0:
            for i, c in enumerate(uniques):
                c_indices = np.where(clusters == c)[0]
                c_points = concat[c_indices]
                centroid = centroids[c].reshape(1, -1)
                dists = cdist(c_points, centroid, 'cosine')
                sort = np.argsort(dists.squeeze())[0]
                selection = torch.tensor([torch.tensor(c_indices[sort])])
                if i == 0: final_indices = selection
                else: final_indices = torch.cat((final_indices, selection), 0)
        else:
            ratio_per_clusters, size_per_clusters = {}, {}
            num_per_clusters = {c : 0 for c in uniques}
            for c in uniques:
                c_indices = np.where(clusters == c)[0]
                size_per_clusters[c] = len(c_indices)

                for idx in c_indices:
                    if idx < label_len:
                        num_per_clusters[c] += 1
                ratio_per_clusters[c] = num_per_clusters[c] / size_per_clusters[c]

            budget_per_clusters = {c : 0 for c in uniques}
            for i in range(n_query):
                min_value = min(ratio_per_clusters.values())
                min_keys = [key for key, value in ratio_per_clusters.items() if value == min_value]
                max_key = max(min_keys, key=size_per_clusters.get)
                budget_per_clusters[max_key] += 1
                num_per_clusters[max_key] += 1
                ratio_per_clusters[max_key] = num_per_clusters[max_key] / size_per_clusters[max_key]
        
            selects = []
            for i, c in enumerate(uniques):
                if budget_per_clusters[c] == 0: continue

                c_indices = np.where(clusters == c)[0]
                c_points = concat[c_indices]
                
                centroid = centroids[c].reshape(1, -1)
                dists = cdist(c_points, centroid, 'cosine')
                sorts = np.argsort(dists.squeeze())
                for idx in c_indices[sorts]:
                    if idx < label_len or idx in selects: continue
                    
                    if scores[idx] >= thresholds[labels_pse[idx]]:
                        if labels_pse[idx] == labels_org[idx]: corr += 1
                        budget_saving += 1
                        selection = torch.tensor([(torch.tensor(idx - label_len), labels_pse[idx], 1)])
                    else:
                        selection = torch.tensor([(torch.tensor(idx - label_len), labels_org[idx], 1)])
                    if len(selects) == 0: final_indices = selection
                    else: final_indices = torch.cat((final_indices, selection), 0)
                    selects.append(idx)
                    break

        if budget_saving == 0: corr_ratio = 0
        else: corr_ratio = corr / budget_saving 

        return final_indices.cpu().numpy(), budget_saving, corr_ratio

    def select(self, n_query, **kwargs):
        labeled_features, unlabeled_features, scores, labels_org, labels_pse = self.get_features()
        selected_indices, budget_saving, corr = self.k_means_uncert(labeled_features, unlabeled_features, scores, labels_org, labels_pse, n_query)
        scores = list(np.ones(len(selected_indices)))
        Q_index = [(self.U_index[int(idx)], int(val), int(flag)) for idx, val, flag in selected_indices]

        return Q_index, budget_saving, corr