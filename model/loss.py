import torch.nn.functional as F
import torch


def nll_loss(output_scores, target_matches):
    loss = []
    ## Making an assumption that batch size == 1
    for i in range(len(target_matches[0])):
        x = target_matches[0][i][0]
        y = target_matches[0][i][1]
        loss.append(-torch.log( output_scores[x][y].exp() )) # check batch size == 1 ?
    loss_mean = torch.mean(torch.stack(loss))
    loss_mean = torch.reshape(loss_mean, (1, -1))
    return loss_mean[0]
