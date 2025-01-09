import torch.nn.functional as F
import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """
    to the cell classification task
        features:[batch_size,num_classes]
        labels:[batch_size,num_classes]
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        return self.loss(features, labels)


class ContrastiveLoss(nn.Module):
    """
        original:[batch_size,contrastive_out_feature]
        reconstructed:[batch_size,n_token,contrastive_out_feature]
    """
    def to_device(self, t1, t2):
        return t1.to(t2.device)

    def __init__(self, temperature=0.07, contrastive_mode='unsupervised'):
        super().__init__()
        self.temperature = temperature
        self.contrastive_mode = contrastive_mode

    def unsupervised_contrastive_loss(self, anchor_feature, contrast_feature):
        batch_size = anchor_feature.shape[0]
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float()
        anchor_feature = anchor_feature.flatten(1)
        contrast_feature = contrast_feature.flatten(1)
        anchor_feature = F.normalize(anchor_feature, dim=1)
        contrast_feature = F.normalize(contrast_feature, dim=1)
        feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        similarity = F.cosine_similarity(feature.unsqueeze(1), feature.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        numerator = torch.exp(positives / self.temperature)
        denominator = self.to_device(mask, similarity) * torch.exp(similarity / self.temperature)
        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

    def supervised_contrastive_loss(self, feature, contrast_feature, label):
        batch_size = feature.shape[0]
        feature = F.normalize(feature.flatten(1), dim=1)
        contrast_feature = F.normalize(contrast_feature.flatten(1), dim=1)
        total_loss = 0.0
        total_pairs = 0

        for i in range(batch_size):
            positive_similarities = []
            negative_similarities = []

            for j in range(batch_size):
                if i == j:
                    sim_with_contrast = F.cosine_similarity(feature[i].unsqueeze(0), contrast_feature[j].unsqueeze(0))
                    positive_similarities.append(sim_with_contrast)
                    continue

                sim_with_feature = F.cosine_similarity(feature[i].unsqueeze(0), feature[j].unsqueeze(0))
                sim_with_contrast = F.cosine_similarity(feature[i].unsqueeze(0), contrast_feature[j].unsqueeze(0))

                if label[i] == label[j]:
                    positive_similarities.extend([sim_with_feature, sim_with_contrast])
                else:
                    negative_similarities.extend([sim_with_feature, sim_with_contrast])

            if positive_similarities:
                positive_similarity = torch.mean(torch.cat(positive_similarities))
                numerator = torch.exp(positive_similarity / self.temperature)
                if negative_similarities:
                    negative_similarity = torch.mean(torch.cat(negative_similarities))
                    denominator = numerator + torch.exp(negative_similarity / self.temperature)
                else:
                    denominator = numerator

                total_loss += -torch.log(numerator / denominator)
                total_pairs += 1

        return total_loss / total_pairs if total_pairs > 0 else torch.tensor(0.0).to(feature.device)

    def forward(self, anchor_feature, contrast_feature=None, label=None):
        if self.contrastive_mode == 'unsupervised':
            if contrast_feature is None:
                raise ValueError('contrast_feature is None in unsupervised mode')
            return self.unsupervised_contrastive_loss(anchor_feature, contrast_feature)
        elif self.contrastive_mode == 'supervised':
            if label is None:
                raise ValueError('label is None in supervised mode')
            return self.supervised_contrastive_loss(anchor_feature, contrast_feature, label)


class ReconstructionLoss(nn.Module):
    """
        original:[batch_size,n_token]
        reconstructed:[batch_size,n_token]
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, original, reconstructed):
        return self.loss(reconstructed, original)


class DistillLoss(nn.Module):
    """
    Distillation loss to encourage two predictions to be close.
    In self-distillation, we encourage the predictions from the original sample and the corrupted sample to be close to each other.
    """

    def __init__(self, temperature=1):
        """
        Parameters
        ----------
        temperature (float, optional):
            scaling factor of the similarity metric.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Parameters
        ----------
        z_i (torch.tensor)
            anchor batch of samples
        z_j (torch.tensor)
            positive batch of samples
        Returns:
            float: loss
        """
        z_i, z_j = z_i.flatten(1), z_j.flatten(1)

        if z_i.size(1) == 1:
            return F.mse_loss(z_i, z_j)
        else:
            z_i, z_j = z_i / self.temperature, z_j / self.temperature
            z_i = F.softmax(z_i, dim=-1)
            return F.cross_entropy(z_j, z_i)


def masked_mse_loss(input, target, mask):
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(input, target, mask):
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()