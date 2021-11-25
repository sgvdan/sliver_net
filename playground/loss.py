import torch


def calc_accuracy(pred_scores, gt_types, specific_label=None):
	batch_size = gt_types.size(0)
	_, pred_labels = pred_scores.max(dim=1)
	_, gt_labels = gt_types.max(dim=1)
	if specific_label is None:
		correct_predictions = (pred_labels == gt_labels).count_nonzero()
		batch_size = gt_labels.size(0)
	else:
		correct_predictions = ((pred_labels == gt_labels) & (gt_labels == specific_label)).count_nonzero()
		batch_size = (gt_labels == specific_label).count_nonzero()

	if batch_size != 0:
		acc = correct_predictions / batch_size
	else:
		acc = torch.tensor(1, dtype=pred_labels.dtype, device=pred_labels.device)

	return round(acc.item(), 2)
