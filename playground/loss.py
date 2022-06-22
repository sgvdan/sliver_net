import torch


def calc_accuracy(pred_scores, gt_types, specific_label=None):
	_, pred_labels = pred_scores.max(dim=1)
	_, gt_labels = gt_types.max(dim=1)

	if specific_label is None:
		correct_predictions = (pred_labels == gt_labels).count_nonzero()
		incorrect_predictions = (pred_labels != gt_types).count_nonzero()
		batch_size = gt_labels.size(0)
	else:
		_, specific_label = specific_label.max(dim=0)
		correct_predictions = ((pred_labels == specific_label) & (gt_labels == specific_label)).count_nonzero()
		incorrect_predictions = ((pred_labels == specific_label) & (gt_labels != specific_label)).count_nonzero()
		batch_size = (gt_labels == specific_label).count_nonzero()

	return incorrect_predictions, correct_predictions, batch_size
